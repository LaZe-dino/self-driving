#pragma once

#include "camera_model.hpp"
#include "ego_motion.hpp"
#include "frame_source.hpp"
#include "ground_view.hpp"
#include "horizon_estimator.hpp"
#include "lane_detector.hpp"
#include "lane_tracker.hpp"
#include "object_detector.hpp"
#include "object_tracker.hpp"
#include "road_geometry.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct StageTimings {
    double undistort_ms = 0.0;
    double objects_ms = 0.0;
    double bev_ms = 0.0;
    double ego_ms = 0.0;
    double lanes_ms = 0.0;
    double track_ms = 0.0;
    double total_ms = 0.0;
};

// Everything Atlas Vision knows about one frame. The dashboard and the data
// logger only read from this; they never compute perception themselves.
struct PerceptionFrame {
    long long frame_id = -1;
    double media_time_s = 0.0;
    double capture_time_s = 0.0;
    double dt_s = 0.0;
    cv::Mat raw;
    cv::Mat undistorted;
    cv::Mat bev;
    EgoMotionEstimate ego;
    double speed_mps = 0.0;
    bool speed_measured = false;
    bool yaw_used = false;
    double pitch_rad = 0.0;
    bool pitch_measured = false;
    bool objects_enabled = false;
    std::vector<Detection> detections;
    std::vector<TrackedObject> objects;
    RoadState prior;
    LaneDetection detection;
    RoadState road;
    TrackStatus status = TrackStatus::Searching;
    double confidence = 0.0;
    RoadGeometry geometry;
    PathPrediction path;
    std::vector<TrackEvent> events;
    StageTimings timings;
};

class VisionPipeline {
public:
    void configure(const CameraModel& camera, const GroundGrid& grid);
    // Optional neural object detection; returns false (with `error`) if the
    // model cannot be loaded.
    bool enable_objects(const ObjectDetectorParams& params, std::string& error);
    void process(const Frame& frame, PerceptionFrame& out);

    const CameraModel& camera() const { return camera_; }
    const GroundView& ground() const { return ground_; }

private:
    CameraModel camera_;
    GroundView ground_;
    LaneDetector detector_;
    LaneTracker tracker_;
    EgoMotion ego_;
    HorizonEstimator horizon_;
    ObjectDetector detector_objects_;
    ObjectTracker object_tracker_;
    cv::Mat map1_, map2_, gray_, object_mask_, object_mask_bev_, lane_valid_;
    double last_time_s_ = -1.0;
    double speed_mps_ = 0.0;
    double speed_age_s_ = 1e9;
    double left_outer_seen_s_ = -1e9;
    double right_outer_seen_s_ = -1e9;
};
