#include "test_main.hpp"
#include "../vision/lane_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <random>

TEST(polynomial_fit_recovers_known_curve) {
    std::mt19937 rng(1);
    std::normal_distribution<double> noise(0.0, 0.03);
    std::vector<cv::Point2d> pts;
    for (double X = 6.0; X < 40.0; X += 0.05) {
        pts.push_back({X, 1.8 + 0.02 * X + 0.001 * X * X + noise(rng)});
    }
    pts.push_back({20.0, 5.0});  // outlier
    cv::Vec3d c;
    cv::Matx33d cov;
    double rms = 0;
    std::vector<cv::Point2d> inliers;
    CHECK(fit_lane_polynomial(pts, 0.01, 10.0, c, cov, rms, &inliers));
    CHECK_NEAR(c[0], 1.8, 0.03);
    CHECK_NEAR(c[1], 0.02, 0.005);
    CHECK_NEAR(c[2], 0.001, 0.0002);
    CHECK(inliers.size() == pts.size() - 1);
}

TEST(short_segment_has_uncertain_curvature) {
    std::vector<cv::Point2d> longp, shortp;
    for (double X = 6.0; X < 40.0; X += 0.05) longp.push_back({X, 1.8});
    for (double X = 6.0; X < 10.0; X += 0.05) shortp.push_back({X, 1.8});
    cv::Vec3d c;
    cv::Matx33d cov_long, cov_short;
    double rms;
    CHECK(fit_lane_polynomial(longp, 0.01, 10.0, c, cov_long, rms, nullptr));
    CHECK(fit_lane_polynomial(shortp, 0.01, 10.0, c, cov_short, rms, nullptr));
    CHECK(cov_short(2, 2) > 20.0 * cov_long(2, 2));
    // The prior bounds curvature uncertainty instead of letting it explode.
    CHECK(std::sqrt(cov_short(2, 2)) <= 0.0101);
}

// Paints lane stripes onto a synthetic grey BEV with a hard shadow edge.
static cv::Mat synthetic_bev(const GroundGrid& g, double yl, double yr, double c1, double c2,
                             bool dashed_right) {
    cv::Mat bev(g.rows(), g.cols(), CV_8UC3, cv::Scalar(90, 90, 90));
    // Shadow over Y > 4.2 m: a strong one-sided edge that no lane line crosses.
    cv::rectangle(bev, cv::Rect(0, 0, g.cols() / 5, g.rows()), cv::Scalar(40, 40, 40), cv::FILLED);
    for (int r = 0; r < g.rows(); ++r) {
        double X = g.x_of_row(r);
        double off = c1 * X + c2 * X * X;
        int cl = static_cast<int>(g.col_of_y(yl + off) + 0.5);
        int cr = static_cast<int>(g.col_of_y(yr + off) + 0.5);
        cv::line(bev, {cl - 1, r}, {cl + 1, r}, cv::Scalar(40, 200, 230));  // yellow
        bool dash_on = std::fmod(X, 12.0) < 3.0;
        if (!dashed_right || dash_on) {
            cv::line(bev, {cr - 1, r}, {cr + 1, r}, cv::Scalar(220, 220, 220));
        }
    }
    return bev;
}

TEST(detector_finds_synthetic_lanes_and_ignores_shadow_edge) {
    GroundGrid g;
    cv::Mat bev = synthetic_bev(g, 1.9, -1.8, 0.01, 0.0008, true);
    cv::Mat valid(bev.size(), CV_8UC1, cv::Scalar(255));
    LaneDetector det;
    LaneDetection out;
    det.detect(bev, valid, g, nullptr, out);
    const LaneMeasurement* left = nullptr;
    const LaneMeasurement* right = nullptr;
    for (const LaneMeasurement& m : out.lanes) {
        if (m.slot == LaneSlot::Left) left = &m;
        if (m.slot == LaneSlot::Right) right = &m;
    }
    CHECK(left != nullptr);
    CHECK(right != nullptr);
    if (left && right) {
        CHECK_NEAR(left->coeffs[0], 1.9, 0.08);
        CHECK_NEAR(right->coeffs[0], -1.8, 0.08);
        CHECK_NEAR(left->coeffs[2], 0.0008, 0.0003);
        CHECK(left->yellow);
        CHECK(!right->yellow);
        CHECK(!left->dashed);
        CHECK(right->dashed);
    }
    // No paint pixels may appear along the shadow boundary.
    int shadow_col = g.cols() / 5;
    int hits = 0;
    for (int r = 0; r < out.binary.rows; ++r) {
        for (int c = shadow_col - 2; c <= shadow_col + 2; ++c) {
            hits += out.binary.at<uchar>(r, c) ? 1 : 0;
        }
    }
    CHECK(hits == 0);
}
