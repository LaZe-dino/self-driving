#pragma once

#include "pipeline.hpp"
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct DataLoggerConfig {
    std::string root = "data/sessions";
    double keyframe_interval_s = 1.0;
    double low_confidence = 0.5;
    double event_min_interval_s = 0.25;
    double max_session_mb = 500.0;
    double max_total_mb = 2000.0;
    int jpeg_quality = 90;
};

// Writes a session folder:
//   session.json    camera model, grid, schema description
//   frames.jsonl    one JSON record per logged frame
//   images/*.jpg    the undistorted frame each record refers to
class DataLogger {
public:
    ~DataLogger();

    bool start(const DataLoggerConfig& cfg, const std::string& source_name, const CameraModel& camera,
               const GroundGrid& grid, std::string& error);
    // Decides whether this frame is worth keeping and queues it.
    void consider(const PerceptionFrame& f);
    // Human label for the current frame ("good" / "bad"); forces a record.
    void label_next(const std::string& label);
    void stop();

    std::string summary() const;
    bool active() const { return running_; }
    const std::string& session_dir() const { return dir_; }

private:
    struct Job {
        std::string image_path;
        cv::Mat image;
        std::string json;
    };
    void worker();
    std::string record_json(const PerceptionFrame& f, const std::string& image_rel,
                            const std::vector<std::string>& reasons) const;
    void enforce_total_budget();

    DataLoggerConfig cfg_;
    CameraModel camera_;
    GroundGrid grid_;
    std::string dir_;
    bool running_ = false;

    double last_keyframe_s_ = -1e9;
    double last_event_s_ = -1e9;
    TrackStatus last_status_ = TrackStatus::Searching;
    std::string pending_label_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Job> queue_;
    std::thread thread_;
    bool stop_ = false;
    std::uint64_t bytes_ = 0;
    long long records_ = 0;
    long long dropped_ = 0;
    bool budget_hit_ = false;
};
