#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Frame preparation shared by live runs, batch logging and calibration, so
// that intrinsics are always estimated at the resolution they are used at.

// Size frames are processed at. `process_width` <= 0, or >= the native width,
// keeps the native size (frames are never upscaled). The height keeps the
// native aspect ratio.
cv::Size processing_size(cv::Size native, int process_width);

// Resizes `in` to `size` (INTER_AREA). Returns `in` itself when the size
// already matches.
cv::Mat resize_for_processing(const cv::Mat& in, cv::Size size);

// .mov / .mp4 / .m4v, case-insensitive (iPhone writes IMG_1234.MOV).
bool is_video_file_name(const std::string& name);
// .jpg / .jpeg / .png, case-insensitive.
bool is_image_file_name(const std::string& name);
// Sorted full paths of the matching regular files directly inside `dir`.
std::vector<std::string> list_files(const std::string& dir, bool (*accept)(const std::string&));

// Picks frames every `interval_s` of media time.
struct FrameSampler {
    double interval_s = 0.5;
    double last_s = -1e18;
    bool due(double t_s) {
        if (t_s - last_s + 1e-6 < interval_s) return false;
        last_s = t_s;
        return true;
    }
};

// Sharpness as the variance of the Laplacian over `roi` of a grey image.
double laplacian_variance(const cv::Mat& gray, const cv::Rect& roi);

struct ChessboardView {
    std::vector<cv::Point2f> corners;
    double sharpness = 0.0;
    double time_s = 0.0;
};

struct ViewSelectParams {
    int max_views = 40;
    // Views below max(min_sharpness, relative_sharpness * median) are blurry.
    double min_sharpness = 15.0;
    double relative_sharpness = 0.5;
    // Views closer than this (mean corner distance / image diagonal) to an
    // already selected view are near-duplicate poses.
    double min_pose_distance = 0.03;
};

struct ViewSelection {
    std::vector<int> selected;
    int blurry = 0;
    int duplicates = 0;
    int surplus = 0;
    double sharpness_threshold = 0.0;
};

// Mean distance between corresponding corners divided by the image diagonal.
// The board may be detected with its corner order reversed, so the smaller of
// the two orderings is used.
double board_pose_distance(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b,
                           cv::Size image_size);

// Drops blurry views, then greedily picks the sharpest view followed by the
// view farthest (in pose) from everything picked so far, until `max_views`
// or until every remaining view is a near-duplicate.
ViewSelection select_calibration_views(const std::vector<ChessboardView>& views, cv::Size image_size,
                                       const ViewSelectParams& params);

// Indices of views whose reprojection error is more than `factor` x median.
std::vector<int> outlier_views(const std::vector<double>& per_view_errors, double factor = 2.0);
