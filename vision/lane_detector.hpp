#pragma once

#include "ground_view.hpp"
#include "lane_types.hpp"
#include <opencv2/core.hpp>
#include <vector>

struct LaneDetectorParams {
    double marking_width_m = 0.15;
    double side_offset_m = 0.30;
    double min_contrast_L = 8.0;
    double min_contrast_b = 5.0;
    double noise_k = 4.0;
    double fresh_half_width_m = 0.6;
    double min_guided_half_width_m = 0.35;
    double max_guided_half_width_m = 0.9;
    double window_height_m = 2.0;
    int min_points = 30;
    double min_span_m = 3.0;
    // Prior on c2: curvature radius mostly above ~50 m.
    double c2_prior_sigma = 0.01;
    // Neighbouring rows see the same smeared paint, so row samples are not
    // independent; the least-squares covariance is scaled up by this factor.
    double correlation_inflation = 10.0;
};

struct SearchWindow {
    cv::Rect rect;
    LaneSlot slot;
    int pixels;
};

struct LaneDetection {
    cv::Mat ridge_L;
    cv::Mat ridge_b;
    cv::Mat binary;
    double noise_L = 0.0;
    double noise_b = 0.0;
    std::vector<float> histogram;
    std::vector<SearchWindow> windows;
    std::vector<LaneMeasurement> lanes;
    bool used_prior = false;
};

class LaneDetector {
public:
    explicit LaneDetector(const LaneDetectorParams& p = LaneDetectorParams()) : p_(p) {}

    // `prior` (may be null) is the tracker's prediction for this frame. With a
    // prior, each boundary is searched in a band around its predicted curve;
    // without one, boundaries are seeded from a column histogram.
    void detect(const cv::Mat& bev_bgr, const cv::Mat& valid, const GroundGrid& grid,
                const RoadState* prior, LaneDetection& out) const;

private:
    void extract_features(const cv::Mat& bev_bgr, const cv::Mat& valid, const GroundGrid& grid,
                          LaneDetection& out) const;
    bool trace_guided(const LaneDetection& det, const GroundGrid& grid, const RoadState& prior,
                      LaneSlot slot, LaneMeasurement& m, std::vector<SearchWindow>& windows) const;
    bool trace_fresh(const LaneDetection& det, const GroundGrid& grid, double seed_col,
                     LaneSlot slot, LaneMeasurement& m, std::vector<SearchWindow>& windows) const;
    bool fit(const GroundGrid& grid, const cv::Mat& bev_lab, LaneMeasurement& m) const;

    LaneDetectorParams p_;
    mutable cv::Mat lab_;
};

// Weighted least squares Y = c0 + c1 X + c2 X^2 with a zero-mean Gaussian
// prior on c2. Two passes: the second drops points far from the first fit.
bool fit_lane_polynomial(const std::vector<cv::Point2d>& pts, double c2_prior_sigma,
                         double correlation_inflation, cv::Vec3d& coeffs, cv::Matx33d& cov,
                         double& rms, std::vector<cv::Point2d>* inliers);
