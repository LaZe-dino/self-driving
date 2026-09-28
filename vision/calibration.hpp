#pragma once

#include "camera_model.hpp"
#include <string>

struct ChessboardResult {
    int images_total = 0;
    int images_used = 0;
    double rms_reprojection_px = 0.0;
};

// Estimates K and lens distortion from photos of a chessboard with
// `inner_cols` x `inner_rows` inner corners.
bool calibrate_from_chessboards(const std::string& folder, int inner_cols, int inner_rows,
                                CameraModel& model, ChessboardResult& result,
                                std::string& error);

struct MountEstimate {
    int frames_used = 0;
    double vp_u_px = 0.0;
    double vp_v_px = 0.0;
    double vp_spread_px = 0.0;
    double lane_width_at_unit_height = 0.0;
};

// Estimates pitch and yaw from the vanishing point of the lane lines, then
// camera height from an assumed real lane width. Needs a mostly straight road.
bool estimate_mount_from_video(const std::string& video_path, double lane_width_m,
                               int max_frames, CameraModel& model, MountEstimate& estimate,
                               std::string& error);
