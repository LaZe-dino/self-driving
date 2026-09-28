#include "ego_motion.hpp"
#include "cv_compat.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>

void EgoMotion::configure(const CameraModel& camera, double x_near_m, double x_far_m) {
    camera_ = camera;
    x_near_ = x_near_m;
    x_far_ = x_far_m;
    prev_.release();
}

void EgoMotion::build_mask(const RoadState& road, double k_left, double k_right, const cv::Mat& exclude) {
    mask_ = cv::Mat::zeros(camera_.image_size, CV_8UC1);
    std::vector<cv::Point> left, right;
    for (double X = x_near_; X <= x_far_ + 1e-9; X += 1.0) {
        // Road bounded by observed lines (plus their paint), or a narrow
        // corridor when no lane is tracked.
        double yl = road.valid ? road.lateral(X, k_left) + 0.3 : 1.5;
        double yr = road.valid ? road.lateral(X, k_right) - 0.3 : -1.5;
        cv::Point2d a = camera_.ground_to_image(X, yl), b = camera_.ground_to_image(X, yr);
        left.push_back(cv::Point(cvRound(a.x), cvRound(a.y)));
        right.push_back(cv::Point(cvRound(b.x), cvRound(b.y)));
    }
    left.insert(left.end(), right.rbegin(), right.rend());
    cv::fillPoly(mask_, std::vector<std::vector<cv::Point>>{left}, cv::Scalar(255));
    if (!exclude.empty()) {
        mask_.setTo(0, exclude);
    }
}

EgoMotionEstimate EgoMotion::estimate(const cv::Mat& gray, const RoadState& road, double k_left,
                                      double k_right, const cv::Mat& exclude) {
    EgoMotionEstimate e;
    build_mask(road, k_left, k_right, exclude);
    if (prev_.empty()) {
        prev_ = gray.clone();
        return e;
    }
    std::vector<cv::Point2f> p0, p1, back;
    cv::goodFeaturesToTrack(prev_, p0, 400, 0.005, 6, mask_);
    if (p0.size() < 10) {
        prev_ = gray.clone();
        return e;
    }
    std::vector<unsigned char> st, st_back;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_, gray, p0, p1, st, err, cv::Size(21, 21), 3);
    // Forward-backward check: track the result back to the previous frame; a
    // reliable track returns to where it started.
    cv::calcOpticalFlowPyrLK(gray, prev_, p1, back, st_back, err, cv::Size(21, 21), 3);
    prev_ = gray.clone();

    std::vector<cv::Point2f> g0, g1;
    for (std::size_t i = 0; i < p0.size(); ++i) {
        if (!st[i] || !st_back[i]) {
            continue;
        }
        cv::Point2f d = back[i] - p0[i];
        if (d.dot(d) > 1.0f) {
            continue;
        }
        double X0, Y0, X1, Y1;
        if (!camera_.image_to_ground(p0[i], X0, Y0) || !camera_.image_to_ground(p1[i], X1, Y1)) {
            continue;
        }
        if (X0 < x_near_ - 2 || X0 > x_far_ + 2) {
            continue;
        }
        g0.push_back(cv::Point2f(static_cast<float>(X0), static_cast<float>(Y0)));
        g1.push_back(cv::Point2f(static_cast<float>(X1), static_cast<float>(Y1)));
        e.from_px.push_back(p0[i]);
        e.to_px.push_back(p1[i]);
    }
    e.tracked = static_cast<int>(g0.size());
    if (e.tracked < 10) {
        return e;
    }

    // Road points are fixed in the world. If the car moves by D and turns by
    // phi, a road point p (car frame) becomes p' = R(phi)^T (p - D).
    // Fitting p' = A p + b gives A = R^T and b = -R^T D.
    // One pixel of flow error is ~0.1 m on the ground at 20 m and ~0.4 m at 40 m.
    cv::Mat A = cv::estimateAffinePartial2D(g0, g1, e.inlier_mask, cv::RANSAC, 0.25, 2000, 0.99);
    if (A.empty()) {
        return e;
    }
    e.inliers = cv::countNonZero(e.inlier_mask);
    double a = A.at<double>(0, 0), c = A.at<double>(1, 0);
    e.scale = std::hypot(a, c);
    double phi = -std::atan2(c, a);
    cv::Vec2d b(A.at<double>(0, 2), A.at<double>(1, 2));
    double cp = std::cos(phi), sp = std::sin(phi);
    e.forward_m = -(cp * b[0] - sp * b[1]);
    e.lateral_m = -(sp * b[0] + cp * b[1]);
    e.yaw_rad = phi;

    // Rotation is observed through how far points sit from their centroid:
    // a point at distance r moving sideways by e metres implies e / r radians.
    // With N inliers of residual s, sigma_yaw ~ s / (sqrt(N) * r_rms).
    cv::Point2d centroid(0, 0);
    int n = 0;
    for (std::size_t i = 0; i < g0.size(); ++i) {
        if (!e.inlier_mask[i]) continue;
        centroid += cv::Point2d(g0[i]);
        ++n;
    }
    centroid *= 1.0 / std::max(1, n);
    double sse = 0.0, r2 = 0.0;
    for (std::size_t i = 0; i < g0.size(); ++i) {
        if (!e.inlier_mask[i]) continue;
        cv::Point2d p(g0[i]);
        cv::Point2d q(a * p.x + A.at<double>(0, 1) * p.y + b[0], c * p.x + A.at<double>(1, 1) * p.y + b[1]);
        cv::Point2d r = q - cv::Point2d(g1[i]);
        sse += r.dot(r);
        cv::Point2d dc = p - centroid;
        r2 += dc.dot(dc);
    }
    double s = std::sqrt(sse / std::max(1, n));
    double r_rms = std::sqrt(r2 / std::max(1, n));
    e.yaw_sigma_rad = std::max(s, 0.02) / (std::sqrt(static_cast<double>(std::max(1, n))) * std::max(r_rms, 0.5));
    // A scale far from 1 means the flat-ground model does not fit this frame
    // (pitching over a bump, or mostly non-road features).
    e.valid = e.inliers >= 10 && std::fabs(e.scale - 1.0) < 0.05;
    return e;
}
