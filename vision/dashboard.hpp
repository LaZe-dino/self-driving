#pragma once

#include "frame_source.hpp"
#include "pipeline.hpp"
#include <opencv2/core.hpp>
#include <deque>
#include <string>

struct HistorySample {
    double t = 0.0;
    bool valid = false;
    TrackStatus status = TrackStatus::Searching;
    double offset = 0.0;
    double offset_sigma = 0.0;
    double curvature = 0.0;
    double curvature_sigma = 0.0;
    double width = 0.0;
    double confidence = 0.0;
    double speed = 0.0;
    bool speed_measured = false;
};

struct DashboardStatus {
    std::string source;
    std::string camera;
    FrameSourceStats source_stats;
    double loop_fps = 0.0;
    double memory_mib = 0.0;
    std::string logger;
    bool paused = false;
    bool show_flow = false;
    bool raw_only = false;
};

class Dashboard {
public:
    Dashboard();
    // Call once per processed frame: records history for the charts.
    void observe(const PerceptionFrame& f);
    // Renders one dashboard image from perception output only.
    const cv::Mat& render(const PerceptionFrame& f, const VisionPipeline& pipeline,
                          const DashboardStatus& status);

private:
    void draw_camera(const PerceptionFrame& f, const VisionPipeline& p, const DashboardStatus& s,
                     cv::Mat& panel);
    void draw_bev(const PerceptionFrame& f, const VisionPipeline& p, cv::Mat& panel, bool features);
    void draw_text(const PerceptionFrame& f, const DashboardStatus& s, cv::Mat& panel);
    void draw_charts(cv::Mat& panel);

    cv::Mat canvas_;
    std::deque<HistorySample> history_;
    std::deque<std::string> event_log_;
    double odometer_m_ = 0.0;
};
