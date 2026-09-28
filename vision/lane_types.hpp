#pragma once

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

// Lane boundaries relative to the road model's lane centre, in multiples of
// the lane width: the ego lane's left line is at +0.5 w, the next line out
// on the left at +1.5 w, and so on.
enum class LaneSlot { Left = 0, Right = 1, LeftOuter = 2, RightOuter = 3 };
constexpr int kLaneSlots = 4;

inline double slot_width_multiple(LaneSlot s) {
    switch (s) {
        case LaneSlot::Left: return 0.5;
        case LaneSlot::Right: return -0.5;
        case LaneSlot::LeftOuter: return 1.5;
        case LaneSlot::RightOuter: return -1.5;
    }
    return 0.0;
}

inline const char* slot_name(LaneSlot s) {
    switch (s) {
        case LaneSlot::Left: return "left";
        case LaneSlot::Right: return "right";
        case LaneSlot::LeftOuter: return "left_outer";
        case LaneSlot::RightOuter: return "right_outer";
    }
    return "?";
}

// One boundary observed in one frame: Y = c0 + c1 X + c2 X^2 in vehicle
// coordinates (metres), with the least-squares covariance of (c0, c1, c2).
struct LaneMeasurement {
    LaneSlot slot = LaneSlot::Left;
    cv::Vec3d coeffs;
    cv::Matx33d cov;
    std::vector<cv::Point2d> points;
    double x_near = 0.0;
    double x_far = 0.0;
    double rms_m = 0.0;
    double coverage = 0.0;
    bool yellow = false;
    bool dashed = false;
    bool guided = false;
    // Filled by the tracker.
    bool accepted = false;
    double mahalanobis2 = 0.0;
};

// Road model tracked over time: a lane of width w whose centre line is
// Y = y_c + c1 X + c2 X^2. c1 ~ heading of the lane relative to the car,
// 2 c2 ~ curvature. All boundaries share c1 and c2 (they are parallel).
struct RoadState {
    bool valid = false;
    cv::Vec4d x;   // y_c, c1, c2, w
    cv::Matx44d P;

    double lateral(double X, double width_multiple) const {
        return x[0] + width_multiple * x[3] + x[1] * X + x[2] * X * X;
    }
    // 1-sigma uncertainty of `lateral` from the state covariance: J P J^T.
    double lateral_sigma(double X, double width_multiple) const {
        cv::Vec4d J(1.0, X, X * X, width_multiple);
        return std::sqrt(std::max(0.0, (J.t() * P * J)(0)));
    }
};
