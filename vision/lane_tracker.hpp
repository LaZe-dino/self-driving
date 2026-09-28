#pragma once

#include "lane_types.hpp"
#include <string>
#include <vector>

enum class TrackStatus { Searching, Locked, Partial, Coasting };

inline const char* status_name(TrackStatus s) {
    switch (s) {
        case TrackStatus::Searching: return "SEARCHING";
        case TrackStatus::Locked: return "LOCKED";
        case TrackStatus::Partial: return "PARTIAL";
        case TrackStatus::Coasting: return "COASTING";
    }
    return "?";
}

struct LaneTrackerParams {
    // Random-walk process noise per second (1 sigma) for y_c, c1, c2, w.
    double q_offset = 0.35;
    double q_heading = 0.03;
    double q_curv = 0.0015;
    double q_width = 0.08;
    // The fit covariance only describes pixel noise. The flat-road, fixed-pitch
    // model is itself wrong by this much (1 sigma for c0, c1, c2): a bounce
    // that changes pitch by 0.3 deg shifts and tilts every line in the BEV.
    double model_sigma_c0 = 0.10;
    double model_sigma_c1 = 0.008;
    double model_sigma_c2 = 0.00015;
    // Outer boundaries assume equal lane widths; trust them less.
    double outer_cov_scale = 4.0;
    // 99.9% chi-square gate for 3 degrees of freedom.
    double gate_chi2 = 16.27;
    double min_width = 2.5;
    double max_width = 5.0;
    double max_coast_s = 1.5;
    double lost_sigma_m = 0.8;
    int reinit_after_rejections = 8;
    // An inner line detected but rejected this many frames in a row, while
    // the other inner line is accepted, means the track itself is wrong.
    int reinit_after_split = 12;
};

struct TrackEvent {
    double time_s;
    std::string what;
};

class LaneTracker {
public:
    explicit LaneTracker(const LaneTrackerParams& p = LaneTrackerParams()) : p_(p) {}

    // Motion model. `forward_m` and `yaw_rad` are the ego-motion since the last
    // frame (0 when unknown; the process noise then covers the motion), with
    // their variances.
    void predict(double time_s, double forward_m, double yaw_rad, double forward_var = 0.0,
                 double yaw_var = 0.0);
    // Associates measurements with the road model, gates outliers and updates.
    // Sets accepted / mahalanobis2 on each measurement.
    void update(std::vector<LaneMeasurement>& measurements);

    const RoadState& state() const { return state_; }
    TrackStatus status() const { return status_; }
    double time_since_update(LaneSlot s) const;
    // 0..1 from the lateral uncertainty of the lane centre 15 m ahead.
    double confidence() const;
    std::vector<TrackEvent> take_events();
    void reset(const std::string& why);

private:
    bool try_initialise(const std::vector<LaneMeasurement>& ms);
    bool apply(LaneMeasurement& m);
    double mahalanobis2(const LaneMeasurement& m) const;
    cv::Matx33d measurement_noise(const LaneMeasurement& m) const;
    void handle_lane_change();

    LaneTrackerParams p_;
    RoadState state_;
    TrackStatus status_ = TrackStatus::Searching;
    double time_s_ = 0.0;
    double last_update_[kLaneSlots] = {-1e9, -1e9, -1e9, -1e9};
    double last_any_update_ = -1e9;
    int consecutive_rejections_ = 0;
    int split_frames_ = 0;
    bool has_time_ = false;
    std::vector<TrackEvent> events_;
};
