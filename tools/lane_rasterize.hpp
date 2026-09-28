#pragma once

#include "../vision/camera_model.hpp"
#include "../vision/lane_types.hpp"
#include <vector>

// Turns the tracked road model into TuSimple-style lanes: one x pixel per
// image row ("h_sample"), -2 where the lane is not drawn. Same rule as the
// dashboard and the data logger: a point is drawn only if it lies between
// x_min and x_max metres ahead and its 1-sigma lateral uncertainty is below
// max_sigma_m.
struct RasterizeOptions {
    double x_min = 6.0;
    double x_max = 40.0;
    double max_sigma_m = 0.4;
};

constexpr int kNoLanePoint = -2;

// The rows the data logger samples: every 10 px from 20 px below the horizon
// to the bottom of the image.
std::vector<int> logger_h_samples(const CameraModel& camera);

// x in UNDISTORTED image pixels at each row (rows are undistorted rows too).
std::vector<int> rasterize_boundary(const RoadState& road, double width_multiple, const CameraModel& camera,
                                    const std::vector<int>& rows, const RasterizeOptions& opt);

// x in the ORIGINAL (lens-distorted) image at each original image row: what a
// human labelled. Equal to rasterize_boundary when the camera has no
// distortion.
std::vector<int> rasterize_boundary_raw(const RoadState& road, double width_multiple, const CameraModel& camera,
                                        const std::vector<int>& rows, const RasterizeOptions& opt);

bool has_distortion(const CameraModel& camera);
