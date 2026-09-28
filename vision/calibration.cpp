#include "calibration.hpp"
#include "cv_compat.hpp"
#include "frame_source.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

bool calibrate_from_chessboards(const std::string& folder, int inner_cols, int inner_rows,
                                CameraModel& model, ChessboardResult& result,
                                std::string& error) {
    std::vector<cv::String> files;
    cv::glob(folder + "/*.jpg", files, false);
    result.images_total = static_cast<int>(files.size());
    if (files.empty()) {
        error = "no .jpg files in " + folder;
        return false;
    }

    std::vector<cv::Point3f> board;
    for (int r = 0; r < inner_rows; ++r) {
        for (int c = 0; c < inner_cols; ++c) {
            board.push_back(cv::Point3f(static_cast<float>(c), static_cast<float>(r), 0.0f));
        }
    }

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    cv::Size size;
    for (const cv::String& f : files) {
        cv::Mat gray = cv::imread(f, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            error = "cannot read " + f;
            return false;
        }
        if (size.width == 0) {
            size = gray.size();
        }
        if (gray.size() != size) {
            std::cout << "  skip " << f << ": size " << gray.cols << "x" << gray.rows
                      << " differs from " << size.width << "x" << size.height << "\n";
            continue;
        }
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, cv::Size(inner_cols, inner_rows), corners,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (!found) {
            std::cout << "  skip " << f << ": full board not visible\n";
            continue;
        }
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
        object_points.push_back(board);
        image_points.push_back(corners);
    }
    result.images_used = static_cast<int>(image_points.size());
    if (result.images_used < 5) {
        error = "only " + std::to_string(result.images_used) + " usable chessboard images (need 5+)";
        return false;
    }

    cv::Mat K, dist;
    std::vector<cv::Mat> rvecs, tvecs;
    result.rms_reprojection_px = cv::calibrateCamera(object_points, image_points, size, K, dist, rvecs, tvecs);
    model.image_size = size;
    model.K = cv::Matx33d(K);
    model.dist = dist;
    model.intrinsics_calibrated = true;
    model.update();
    return true;
}

namespace {

struct ImageLine {
    cv::Point2d point;
    cv::Point2d dir;
};

bool fit_side(const std::vector<cv::Vec4i>& segs, ImageLine& out) {
    std::vector<cv::Point2f> pts;
    for (const cv::Vec4i& s : segs) {
        double len = std::hypot(s[2] - s[0], s[3] - s[1]);
        int n = std::max(2, static_cast<int>(len / 5.0));
        for (int i = 0; i <= n; ++i) {
            double a = static_cast<double>(i) / n;
            pts.push_back(cv::Point2f(static_cast<float>(s[0] + a * (s[2] - s[0])),
                                      static_cast<float>(s[1] + a * (s[3] - s[1]))));
        }
    }
    if (pts.size() < 20) {
        return false;
    }
    cv::Vec4f l;
    cv::fitLine(pts, l, cv::DIST_HUBER, 0, 0.01, 0.01);
    out.dir = {l[0], l[1]};
    out.point = {l[2], l[3]};
    return true;
}

bool intersect(const ImageLine& a, const ImageLine& b, cv::Point2d& p) {
    double den = a.dir.x * b.dir.y - a.dir.y * b.dir.x;
    if (std::fabs(den) < 1e-9) {
        return false;
    }
    cv::Point2d d = b.point - a.point;
    double s = (d.x * b.dir.y - d.y * b.dir.x) / den;
    p = a.point + s * a.dir;
    return true;
}

cv::Point2d point_at_row(const ImageLine& l, double v) {
    double s = (v - l.point.y) / l.dir.y;
    return l.point + s * l.dir;
}

double median(std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

}  // namespace

bool estimate_mount_from_video(const std::string& video_path, double lane_width_m,
                               int max_frames, CameraModel& model, MountEstimate& estimate,
                               std::string& error) {
    cv::VideoCapture cap;
    if (!open_video_file(cap, video_path)) {
        error = "cannot open " + video_path;
        return false;
    }
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(model.K, model.dist, cv::Mat(), model.K, model.image_size,
                                CV_16SC2, map1, map2);

    std::vector<ImageLine> lefts, rights;
    std::vector<double> us, vs;
    cv::Mat frame, und, gray, edges;
    int w = model.image_size.width;
    int h = model.image_size.height;
    int row_top = static_cast<int>(h * 0.62);
    int row_bottom = static_cast<int>(h * 0.92);

    for (int i = 0; i < max_frames && cap.read(frame); ++i) {
        if (frame.size() != model.image_size) {
            error = "video frame size differs from calibrated camera size";
            return false;
        }
        cv::remap(frame, und, map1, map2, cv::INTER_LINEAR);
        cv::cvtColor(und, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
        cv::Canny(gray, edges, 50, 150);
        cv::Mat roi_mask = cv::Mat::zeros(edges.size(), CV_8UC1);
        roi_mask(cv::Rect(0, row_top, w, row_bottom - row_top)).setTo(255);
        edges &= roi_mask;

        std::vector<cv::Vec4i> segs, left_segs, right_segs;
        cv::HoughLinesP(edges, segs, 1, CV_PI / 180.0, 30, 30, 20);
        for (const cv::Vec4i& s : segs) {
            double angle = std::atan2(s[3] - s[1], s[2] - s[0]) * 180.0 / CV_PI;
            if (angle < 0) {
                angle += 180.0;
            }
            double mid_u = 0.5 * (s[0] + s[2]);
            // Left lane rises to the right (image angle ~ 120-160 deg),
            // right lane rises to the left (~20-60 deg).
            if (angle > 115 && angle < 160 && mid_u < w * 0.5) {
                left_segs.push_back(s);
            } else if (angle > 20 && angle < 65 && mid_u > w * 0.5) {
                right_segs.push_back(s);
            }
        }
        ImageLine L, R;
        cv::Point2d vp;
        if (!fit_side(left_segs, L) || !fit_side(right_segs, R) || !intersect(L, R, vp)) {
            continue;
        }
        if (vp.x < 0 || vp.x > w || vp.y < 0 || vp.y > row_top) {
            continue;
        }
        lefts.push_back(L);
        rights.push_back(R);
        us.push_back(vp.x);
        vs.push_back(vp.y);
    }
    estimate.frames_used = static_cast<int>(us.size());
    if (estimate.frames_used < 20) {
        error = "only " + std::to_string(estimate.frames_used) +
                " frames had both lane lines; need a straight, well-marked road";
        return false;
    }

    estimate.vp_u_px = median(us);
    estimate.vp_v_px = median(vs);
    std::vector<double> dev;
    for (std::size_t i = 0; i < us.size(); ++i) {
        dev.push_back(std::hypot(us[i] - estimate.vp_u_px, vs[i] - estimate.vp_v_px));
    }
    estimate.vp_spread_px = median(dev);

    // Vanishing point of the forward direction (derived in camera_model.cpp):
    //   v = cy - fy tan(pitch),   u = cx + fx tan(yaw) / cos(pitch)
    double fx = model.K(0, 0), fy = model.K(1, 1), cx = model.K(0, 2), cy = model.K(1, 2);
    model.pitch_rad = std::atan((cy - estimate.vp_v_px) / fy);
    model.yaw_rad = std::atan((estimate.vp_u_px - cx) * std::cos(model.pitch_rad) / fx);

    // Every ground distance scales linearly with camera height, so measure
    // the lane width with h = 1 and scale to the known real width.
    model.height_m = 1.0;
    model.update();
    std::vector<double> widths;
    double v_probe = row_bottom - 10.0;
    for (std::size_t i = 0; i < lefts.size(); ++i) {
        double xl, yl, xr, yr;
        if (!model.image_to_ground(point_at_row(lefts[i], v_probe), xl, yl) ||
            !model.image_to_ground(point_at_row(rights[i], v_probe), xr, yr)) {
            continue;
        }
        widths.push_back(std::fabs(yl - yr));
    }
    if (widths.empty()) {
        error = "lane lines did not back-project onto the ground";
        return false;
    }
    estimate.lane_width_at_unit_height = median(widths);
    model.height_m = lane_width_m / estimate.lane_width_at_unit_height;
    model.mount_calibrated = true;
    model.update();
    return true;
}
