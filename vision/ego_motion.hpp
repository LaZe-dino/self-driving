#pragma once

#include "camera_model.hpp"
#include "lane_types.hpp"
#include <opencv2/core.hpp>
#include <vector>

struct EgoMotionEstimate {
    bool valid = false;
    double forward_m = 0.0;
    double lateral_m = 0.0;
    double yaw_rad = 0.0;
    // 1-sigma uncertainty from the inlier residuals of the rigid fit.
    double yaw_sigma_rad = 0.0;
    double scale = 1.0;
    int tracked = 0;
    int inliers = 0;
    // For visualisation: flow vectors in image pixels and whether each was a
    // RANSAC inlier (static road) or outlier (moving object, bad track).
    std::vector<cv::Point2f> from_px;
    std::vector<cv::Point2f> to_px;
    std::vector<unsigned char> inlier_mask;
};

// Visual odometry on the road plane: track corners with pyramidal
// Lucas-Kanade optical flow, project both ends to the ground with the camera
// homography, and fit one rigid 2D motion with RANSAC.
class EgoMotion {
public:
    void configure(const CameraModel& camera, double x_near_m, double x_far_m);
    // Updates the projection (e.g. live pitch) without dropping the previous frame.
    void set_camera(const CameraModel& camera) { camera_ = camera; }
    // `road` (may be invalid) restricts features to road surface bounded by
    // observed lane lines: `k_left` / `k_right` are the outermost boundaries
    // (in lane-width multiples) seen recently. Barriers, walls and verges are
    // not on the road plane and would corrupt the ground-motion fit.
    // `exclude` (same size as the image, may be empty) marks pixels to ignore,
    // e.g. detected vehicles.
    EgoMotionEstimate estimate(const cv::Mat& undistorted_gray, const RoadState& road, double k_left,
                               double k_right, const cv::Mat& exclude);

private:
    void build_mask(const RoadState& road, double k_left, double k_right, const cv::Mat& exclude);

    CameraModel camera_;
    cv::Mat mask_;
    cv::Mat prev_;
    double x_near_ = 5.0;
    double x_far_ = 30.0;
};
