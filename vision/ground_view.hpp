#pragma once

#include "camera_model.hpp"
#include <opencv2/core.hpp>

// Bird's-eye view: a metric grid on the road plane. Row 0 is the far edge
// (x_max), the bottom row is the near edge (x_min); column 0 is the left edge
// (+y_half). Each pixel covers `resolution_m` x `resolution_m` of road.
struct GroundGrid {
    double x_min = 6.0;
    double x_max = 40.0;
    double y_half = 7.0;
    double resolution_m = 0.05;

    int cols() const { return static_cast<int>(2.0 * y_half / resolution_m + 0.5); }
    int rows() const { return static_cast<int>((x_max - x_min) / resolution_m + 0.5); }
    double x_of_row(double row) const { return x_max - (row + 0.5) * resolution_m; }
    double y_of_col(double col) const { return y_half - (col + 0.5) * resolution_m; }
    double row_of_x(double x) const { return (x_max - x) / resolution_m - 0.5; }
    double col_of_y(double y) const { return (y_half - y) / resolution_m - 0.5; }
    // Homogeneous map from grid pixel (col, row, 1) to ground (X, Y, 1).
    cv::Matx33d grid_to_ground() const;
};

class GroundView {
public:
    void configure(const CameraModel& camera, const GroundGrid& grid);
    // `undistorted` must already have lens distortion removed.
    void warp(const cv::Mat& undistorted, cv::Mat& bev) const;

    const GroundGrid& grid() const { return grid_; }
    // 255 where the BEV pixel maps inside the camera image.
    const cv::Mat& valid_mask() const { return valid_; }

private:
    GroundGrid grid_;
    cv::Matx33d H_grid_to_image_;
    cv::Size image_size_;
    cv::Mat valid_;
};
