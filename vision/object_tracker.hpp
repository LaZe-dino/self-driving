#pragma once

#include "camera_model.hpp"
#include "lane_types.hpp"
#include "object_detector.hpp"
#include <opencv2/core.hpp>
#include <vector>

struct TrackedObject {
    int id = 0;
    int class_id = -1;
    float score = 0.0f;
    cv::Rect2d box;
    cv::Point2d box_velocity;  // px/s
    int hits = 0;
    int misses = 0;
    bool confirmed = false;
    bool matched = false;  // detected in the current frame
    // Position relative to the car on the road plane, from the box's ground
    // contact point, filtered with a constant-velocity Kalman filter.
    bool ground_valid = false;
    cv::Vec4d g;  // X, Y, vX, vY (relative to the car)
    cv::Matx44d P;
    bool in_ego_lane = false;
    double ttc_s = -1.0;
};

class ObjectTracker {
public:
    void update(const std::vector<Detection>& dets, const CameraModel& camera, const RoadState& road,
                double dt_s);
    const std::vector<TrackedObject>& tracks() const { return tracks_; }

private:
    void update_ground(TrackedObject& t, const CameraModel& camera, double dt_s, bool measured);

    std::vector<TrackedObject> tracks_;
    int next_id_ = 1;
};
