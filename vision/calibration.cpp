#include "calibration.hpp"
#include "cv_compat.hpp"
#include "frame_source.hpp"
#include "video_prep.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool find_board(const cv::Mat& gray, const ChessboardOptions& o, bool fast_check,
                std::vector<cv::Point2f>& corners) {
    int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    if (fast_check) flags |= cv::CALIB_CB_FAST_CHECK;
    cv::Size pattern(o.inner_cols, o.inner_rows);
    if (!cv::findChessboardCorners(gray, pattern, corners, flags)) {
        return false;
    }
    // The refinement window must stay inside one square, or corners snap to
    // their neighbours on small / distant boards.
    double min_step = 1e9;
    for (int r = 0; r < o.inner_rows; ++r) {
        for (int c = 0; c + 1 < o.inner_cols; ++c) {
            cv::Point2f d = corners[r * o.inner_cols + c + 1] - corners[r * o.inner_cols + c];
            min_step = std::min(min_step, static_cast<double>(std::hypot(d.x, d.y)));
        }
    }
    int half = std::clamp(static_cast<int>(min_step * 0.4), 2, 11);
    cv::cornerSubPix(gray, corners, cv::Size(half, half), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
    return true;
}

double run_calibration(const std::vector<std::vector<cv::Point2f>>& image_points,
                       const std::vector<cv::Point3f>& board, cv::Size size, cv::Mat& K, cv::Mat& dist,
                       std::vector<double>& per_view) {
    std::vector<std::vector<cv::Point3f>> object_points(image_points.size(), board);
    std::vector<cv::Mat> rvecs, tvecs;
    cv::Mat std_int, std_ext, errors;
    K.release();
    dist.release();
    double rms = cv::calibrateCamera(object_points, image_points, size, K, dist, rvecs, tvecs, std_int,
                                     std_ext, errors);
    errors.convertTo(errors, CV_64F);
    per_view.assign(errors.begin<double>(), errors.end<double>());
    return rms;
}

// Calibrates, drops views with more than 2x the median error, recalibrates.
bool calibrate_views(std::vector<std::vector<cv::Point2f>> image_points, std::vector<std::string> names,
                     cv::Size size, const ChessboardOptions& o, CameraModel& model, ChessboardResult& result,
                     std::string& error) {
    if (image_points.size() < 5) {
        error = "only " + std::to_string(image_points.size()) + " usable chessboard views (need 5+, 15+ is better)";
        return false;
    }
    std::vector<cv::Point3f> board;
    float s = static_cast<float>(o.square_mm / 1000.0);
    for (int r = 0; r < o.inner_rows; ++r) {
        for (int c = 0; c < o.inner_cols; ++c) {
            board.push_back(cv::Point3f(c * s, r * s, 0.0f));
        }
    }
    cv::Mat K, dist;
    std::vector<double> per_view;
    result.rms_initial_px = run_calibration(image_points, board, size, K, dist, per_view);
    result.rms_reprojection_px = result.rms_initial_px;

    std::vector<int> bad = outlier_views(per_view, 2.0);
    if (!bad.empty() && image_points.size() - bad.size() >= 5) {
        for (int i : bad) {
            std::printf("  reject %s: %.3f px reprojection error (> 2x median)\n", names[i].c_str(), per_view[i]);
        }
        for (auto it = bad.rbegin(); it != bad.rend(); ++it) {
            image_points.erase(image_points.begin() + *it);
            names.erase(names.begin() + *it);
        }
        result.outliers_rejected = static_cast<int>(bad.size());
        result.rms_reprojection_px = run_calibration(image_points, board, size, K, dist, per_view);
    }
    result.images_used = static_cast<int>(image_points.size());
    result.view_names = names;
    result.per_view_error_px = per_view;

    model.image_size = size;
    model.K = cv::Matx33d(K);
    model.dist = dist;
    model.intrinsics_calibrated = true;
    model.update();
    return true;
}

}  // namespace

bool calibrate_from_chessboards(const std::string& folder, const ChessboardOptions& o, CameraModel& model,
                                ChessboardResult& result, std::string& error) {
    std::vector<std::string> files = list_files(folder, is_image_file_name);
    result.images_total = static_cast<int>(files.size());
    if (files.empty()) {
        error = "no .jpg/.jpeg/.png files in " + folder;
        return false;
    }

    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::string> names;
    cv::Size native, size;
    for (const std::string& f : files) {
        cv::Mat gray = cv::imread(f, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            error = "cannot read " + f;
            return false;
        }
        if (native.width == 0) {
            native = gray.size();
            size = processing_size(native, o.process_width);
            std::printf("images %dx%d, calibrating at %dx%d\n", native.width, native.height, size.width,
                        size.height);
        }
        if (gray.size() != native) {
            std::cout << "  skip " << f << ": size " << gray.cols << "x" << gray.rows
                      << " differs from " << native.width << "x" << native.height << "\n";
            continue;
        }
        gray = resize_for_processing(gray, size);
        std::vector<cv::Point2f> corners;
        if (!find_board(gray, o, false, corners)) {
            std::cout << "  skip " << f << ": full board not visible\n";
            continue;
        }
        image_points.push_back(corners);
        names.push_back(std::filesystem::path(f).filename().string());
    }
    result.boards_found = static_cast<int>(image_points.size());
    return calibrate_views(image_points, names, size, o, model, result, error);
}

bool calibrate_from_chessboard_video(const std::string& video_path, const ChessboardOptions& o,
                                     CameraModel& model, ChessboardResult& result, std::string& error) {
    cv::VideoCapture cap;
    if (!open_video_file(cap, video_path, error)) {
        return false;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::string orientation = describe_orientation(cap);
    FrameSampler sampler;
    sampler.interval_s = o.sample_interval_s;
    std::vector<ChessboardView> views;
    cv::Size native, size;
    cv::Mat frame, gray;
    for (long long idx = 0; cap.grab(); ++idx) {
        double t = fps > 0 ? idx / fps : cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        if (!sampler.due(t) || !cap.retrieve(frame) || frame.empty()) {
            continue;
        }
        if (native.width == 0) {
            native = frame.size();
            size = processing_size(native, o.process_width);
            std::printf("video %dx%d%s%s, calibrating at %dx%d, sampling every %.2f s\n", native.width,
                        native.height, orientation.empty() ? "" : ", ", orientation.c_str(), size.width,
                        size.height, o.sample_interval_s);
            if (native.height > native.width) {
                error = "video is portrait (" + std::to_string(native.width) + "x" + std::to_string(native.height) +
                        "); record the calibration and the drives in landscape";
                return false;
            }
        }
        if (frame.size() != native) {
            continue;
        }
        ++result.images_total;
        cv::cvtColor(resize_for_processing(frame, size), gray, cv::COLOR_BGR2GRAY);
        ChessboardView v;
        v.time_s = t;
        if (!find_board(gray, o, true, v.corners)) {
            continue;
        }
        v.sharpness = laplacian_variance(gray, cv::boundingRect(v.corners));
        views.push_back(v);
        if (views.size() % 10 == 0) {
            std::printf("  t=%.1fs: %zu boards found in %d sampled frames\n", t, views.size(), result.images_total);
            std::fflush(stdout);
        }
    }
    result.boards_found = static_cast<int>(views.size());
    if (result.images_total == 0) {
        error = "no frames decoded from " + video_path;
        return false;
    }

    ViewSelectParams params;
    params.max_views = o.max_views;
    ViewSelection sel = select_calibration_views(views, size, params);
    result.blurry = sel.blurry;
    result.duplicates = sel.duplicates;
    result.surplus = sel.surplus;
    std::printf("boards found in %d/%d sampled frames: %d blurry (sharpness < %.0f), %d near-duplicate poses, "
                "%d surplus, %zu selected\n",
                result.boards_found, result.images_total, sel.blurry, sel.sharpness_threshold, sel.duplicates,
                sel.surplus, sel.selected.size());

    std::sort(sel.selected.begin(), sel.selected.end());
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::string> names;
    for (int i : sel.selected) {
        image_points.push_back(views[i].corners);
        char name[32];
        std::snprintf(name, sizeof(name), "t=%.1fs", views[i].time_s);
        names.push_back(name);
    }
    return calibrate_views(image_points, names, size, o, model, result, error);
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
    if (!open_video_file(cap, video_path, error)) {
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
            if (!same_aspect_ratio(frame.size(), model.image_size)) {
                error = "video frames are " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) +
                        " but the camera model is " + std::to_string(w) + "x" + std::to_string(h) +
                        " (different aspect ratio)";
                return false;
            }
            frame = resize_for_processing(frame, model.image_size);
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
