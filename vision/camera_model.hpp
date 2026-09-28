#pragma once

#include <opencv2/core.hpp>
#include <string>

// Pinhole camera looking at a flat road.
//
// Vehicle frame (metres): X forward, Y left, Z up, origin on the ground
// directly below the camera. Camera frame: x right, y down, z along the
// optical axis. The camera sits at height h, pitched down by `pitch` and
// yawed left by `yaw`.
//
// A 3D point P projects to pixel p ~ K (R P + t). For points on the road
// (Z = 0) the third column of R drops out and projection becomes a 3x3
// homography: p ~ K [r1 r2 t] (X, Y, 1)^T. That single matrix is what lets a
// monocular camera measure metres, as long as the flat-ground assumption holds.
struct CameraModel {
    cv::Size image_size;
    cv::Matx33d K = cv::Matx33d::eye();
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
    double height_m = 1.4;
    double pitch_rad = 0.0;
    double yaw_rad = 0.0;
    bool intrinsics_calibrated = false;
    bool mount_calibrated = false;
    std::string source;

    cv::Matx33d R;
    cv::Vec3d t;
    cv::Matx33d H_ground_to_image;
    cv::Matx33d H_image_to_ground;

    void update();

    cv::Point2d ground_to_image(double X, double Y) const;
    // False when the pixel is at or above the horizon (ray never hits the road).
    bool image_to_ground(const cv::Point2d& px, double& X, double& Y) const;
    // Pixel where parallel ground lines with direction (dx, dy) meet.
    cv::Point2d vanishing_point(double dx, double dy) const;
    double horizon_row_at(double u) const;
};

CameraModel default_camera_model(cv::Size image_size, double horizontal_fov_deg);
bool load_camera_model(const std::string& path, CameraModel& model, std::string& error);
bool save_camera_model(const std::string& path, const CameraModel& model, std::string& error);
