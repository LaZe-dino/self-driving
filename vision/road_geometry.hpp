#pragma once

#include "camera_model.hpp"
#include "lane_types.hpp"
#include <opencv2/core.hpp>
#include <vector>

struct RoadGeometry {
    bool valid = false;
    // Car position relative to the lane centre, + = car is left of centre.
    double lateral_offset_m = 0.0;
    double lateral_offset_sigma_m = 0.0;
    // Lane direction relative to the car's heading, + = lane points left.
    double heading_error_rad = 0.0;
    // Signed curvature of the lane centre at the car (1/m), + = turning left.
    double curvature_1pm = 0.0;
    double curvature_sigma_1pm = 0.0;
    double lane_width_m = 0.0;
    double lane_width_sigma_m = 0.0;
    // Where the lane, extended straight along its current direction, meets
    // the horizon in the image.
    cv::Point2d lane_vanishing_px;
    cv::Point2d horizon_left_px;
    cv::Point2d horizon_right_px;
};

struct PathPrediction {
    bool valid = false;
    std::vector<cv::Point2d> points;  // car frame, metres
    double steering_rad = 0.0;
    double lookahead_m = 0.0;
    double speed_mps = 0.0;
    bool speed_measured = false;
};

RoadGeometry compute_road_geometry(const RoadState& road, const CameraModel& camera);

// Rolls the kinematic bicycle model forward under Pure Pursuit steering toward
// the tracked lane centre. Stops at `max_range_m`, the far edge of what the
// camera actually observed, instead of extrapolating the polynomial.
PathPrediction predict_path(const RoadState& road, double speed_mps, bool speed_measured,
                            double max_range_m);
