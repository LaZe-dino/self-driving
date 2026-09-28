#pragma once

#include "camera_model.hpp"
#include <string>
#include <vector>

struct ChessboardOptions {
    int inner_cols = 9;
    int inner_rows = 6;
    double square_mm = 25.0;
    // Images / frames are resized to this width before corner detection, so
    // the intrinsics come out at the processing resolution (0 = native).
    int process_width = 0;
    // Video only.
    double sample_interval_s = 0.5;
    int max_views = 40;
};

struct ChessboardResult {
    int images_total = 0;   // images in the folder, or frames sampled from the video
    int images_used = 0;    // views in the final calibration
    int boards_found = 0;
    int blurry = 0;
    int duplicates = 0;
    int surplus = 0;
    int outliers_rejected = 0;
    double rms_initial_px = 0.0;
    double rms_reprojection_px = 0.0;
    std::vector<std::string> view_names;   // parallel to per_view_error_px (final calibration)
    std::vector<double> per_view_error_px;
};

// Estimates K and lens distortion from photos (.jpg/.jpeg/.png) of a
// chessboard with `inner_cols` x `inner_rows` inner corners.
bool calibrate_from_chessboards(const std::string& folder, const ChessboardOptions& options,
                                CameraModel& model, ChessboardResult& result, std::string& error);

// Same from a video of the chessboard: samples a frame every
// `sample_interval_s`, drops blurry frames and near-duplicate board poses,
// and calibrates on up to `max_views` views.
bool calibrate_from_chessboard_video(const std::string& video_path, const ChessboardOptions& options,
                                     CameraModel& model, ChessboardResult& result, std::string& error);

struct MountEstimate {
    int frames_used = 0;
    double vp_u_px = 0.0;
    double vp_v_px = 0.0;
    double vp_spread_px = 0.0;
    double lane_width_at_unit_height = 0.0;
};

// Estimates pitch and yaw from the vanishing point of the lane lines, then
// camera height from an assumed real lane width. Needs a mostly straight road.
// Frames are resized to model.image_size when they have the same aspect ratio.
bool estimate_mount_from_video(const std::string& video_path, double lane_width_m,
                               int max_frames, CameraModel& model, MountEstimate& estimate,
                               std::string& error);
