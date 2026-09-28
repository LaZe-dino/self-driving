#include "pipeline.hpp"
#include "cv_compat.hpp"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

void VisionPipeline::configure(const CameraModel& camera, const GroundGrid& grid) {
    camera_ = camera;
    cv::initUndistortRectifyMap(camera.K, camera.dist, cv::Mat(), camera.K, camera.image_size,
                                CV_16SC2, map1_, map2_);
    ground_.configure(camera, grid);
    horizon_.configure(camera.pitch_rad);
    // Near the car the road moves ~20+ px per frame at highway speed and is
    // motion-blurred; the strongest remaining corners there are windshield
    // reflections that move with the camera. 12-40 m stays sharp.
    ego_.configure(camera, 12.0, grid.x_max);
    tracker_ = LaneTracker();
    last_time_s_ = -1.0;
    speed_mps_ = 0.0;
    speed_age_s_ = 1e9;
    left_outer_seen_s_ = -1e9;
    right_outer_seen_s_ = -1e9;
}

bool VisionPipeline::enable_objects(const ObjectDetectorParams& params, std::string& error) {
    return detector_objects_.load(params, error);
}

void VisionPipeline::process(const Frame& frame, PerceptionFrame& out) {
    using clock = std::chrono::steady_clock;
    auto t_all = clock::now();
    out.frame_id = frame.id;
    out.media_time_s = frame.media_time_s;
    out.capture_time_s = frame.capture_time_s;
    out.dt_s = last_time_s_ < 0 ? 0.0 : frame.media_time_s - last_time_s_;
    last_time_s_ = frame.media_time_s;
    out.raw = frame.image;

    auto t = clock::now();
    cv::remap(frame.image, out.undistorted, map1_, map2_, cv::INTER_LINEAR);
    out.timings.undistort_ms = ms_since(t);

    // Objects: vehicles and people hide lane paint and move independently of
    // the road, so their boxes are removed from lane search and odometry.
    t = clock::now();
    out.objects_enabled = detector_objects_.loaded();
    object_mask_ = cv::Mat::zeros(out.undistorted.size(), CV_8UC1);
    if (out.objects_enabled) {
        detector_objects_.detect(out.undistorted, out.detections);
        object_tracker_.update(out.detections, camera_, tracker_.state(), out.dt_s);
        out.objects = object_tracker_.tracks();
        cv::Rect image(0, 0, object_mask_.cols, object_mask_.rows);
        for (const Detection& d : out.detections) {
            if (!is_vehicle_or_person(d.class_id)) continue;
            cv::Rect2d grown(d.box.x - 0.05 * d.box.width, d.box.y, 1.1 * d.box.width, 1.05 * d.box.height);
            cv::rectangle(object_mask_, cv::Rect(grown) & image, cv::Scalar(255), cv::FILLED);
        }
    }
    out.timings.objects_ms = ms_since(t);

    t = clock::now();
    ground_.warp(out.undistorted, out.bev);
    ground_.warp(object_mask_, object_mask_bev_);
    cv::bitwise_and(ground_.valid_mask(), ~object_mask_bev_, lane_valid_);
    out.timings.bev_ms = ms_since(t);

    t = clock::now();
    cv::cvtColor(out.undistorted, gray_, cv::COLOR_BGR2GRAY);
    double k_left = frame.media_time_s - left_outer_seen_s_ < 1.0 ? 1.5 : 0.5;
    double k_right = frame.media_time_s - right_outer_seen_s_ < 1.0 ? -1.5 : -0.5;
    out.ego = ego_.estimate(gray_, tracker_.state(), k_left, k_right, object_mask_);
    speed_age_s_ += out.dt_s;
    if (out.ego.valid && out.dt_s > 1e-3) {
        // Low-pass the per-frame speed (time constant ~0.3 s): single-frame
        // odometry is noisy, the car's real speed changes slowly.
        double v = out.ego.forward_m / out.dt_s;
        double alpha = out.dt_s / (0.3 + out.dt_s);
        speed_mps_ = speed_age_s_ > 1.0 ? v : speed_mps_ + alpha * (v - speed_mps_);
        speed_age_s_ = 0.0;
    }
    out.speed_measured = speed_age_s_ < 1.0;
    out.speed_mps = out.speed_measured ? speed_mps_ : 0.0;

    double forward = out.speed_mps * out.dt_s;
    double forward_var = 0.1 * forward * 0.1 * forward;
    // Tyre friction limits lateral acceleration to ~5 m/s^2, so the yaw rate
    // of a real car is at most a_lat / v. Faster "yaw" is a bad flow fit.
    double max_yaw_rate = std::min(0.6, 5.0 / std::max(out.speed_mps, 1.0));
    double yaw = 0.0;
    double yaw_var = std::pow(max_yaw_rate * out.dt_s, 2) / 3.0;
    out.yaw_used = out.ego.valid && std::fabs(out.ego.yaw_rad) <= max_yaw_rate * out.dt_s;
    if (out.yaw_used) {
        yaw = out.ego.yaw_rad;
        yaw_var = out.ego.yaw_sigma_rad * out.ego.yaw_sigma_rad;
    }
    out.timings.ego_ms = ms_since(t);

    t = clock::now();
    tracker_.predict(frame.media_time_s, forward, yaw, forward_var, yaw_var);
    out.prior = tracker_.state();
    detector_.detect(out.bev, lane_valid_, ground_.grid(), &out.prior, out.detection);
    out.timings.lanes_ms = ms_since(t);

    t = clock::now();
    tracker_.update(out.detection.lanes);
    for (const LaneMeasurement& m : out.detection.lanes) {
        if (m.accepted && m.slot == LaneSlot::LeftOuter) left_outer_seen_s_ = frame.media_time_s;
        if (m.accepted && m.slot == LaneSlot::RightOuter) right_outer_seen_s_ = frame.media_time_s;
    }
    out.road = tracker_.state();
    // The horizon update affects the next frame's warp; this frame's
    // measurements were made with the pitch in out.pitch_rad.
    out.pitch_rad = camera_.pitch_rad;
    out.pitch_measured = horizon_.update(camera_, out.detection.lanes, out.dt_s);
    if (std::fabs(horizon_.pitch() - camera_.pitch_rad) > 1e-4) {
        camera_.pitch_rad = horizon_.pitch();
        camera_.update();
        ground_.configure(camera_, ground_.grid());
        ego_.set_camera(camera_);
    }
    out.status = tracker_.status();
    out.confidence = tracker_.confidence();
    out.events = tracker_.take_events();
    out.geometry = compute_road_geometry(out.road, camera_);
    out.path = predict_path(out.road, out.speed_mps, out.speed_measured, ground_.grid().x_max);
    out.timings.track_ms = ms_since(t);
    out.timings.total_ms = ms_since(t_all);
}
