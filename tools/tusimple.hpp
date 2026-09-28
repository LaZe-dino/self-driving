#pragma once

#include "../vision/camera_model.hpp"
#include <string>
#include <vector>

// One line of a TuSimple label file (test_label.json, label_data_*.json).
struct TuSimpleLabel {
    std::string raw_file;                  // e.g. clips/0530/1492626760788443246_0/20.jpg
    std::vector<int> h_samples;            // image rows
    std::vector<std::vector<int>> lanes;   // x per h_sample, -2 = no lane at that row
};

bool parse_tusimple_label(const std::string& json_line, TuSimpleLabel& out, std::string& error);
bool load_tusimple_labels(const std::string& path, std::vector<TuSimpleLabel>& out, std::string& error);

// Port of the official TuSimple lane.py (LaneEval). Constants as in the
// official script.
struct TuSimpleParams {
    double pixel_thresh = 20.0;
    double pt_thresh = 0.85;
    double max_run_time_ms = 200.0;
};

struct TuSimpleScore {
    double accuracy = 0.0;
    double fp = 0.0;
    double fn = 0.0;
    int matched = 0;
    // Best point accuracy of each gt lane over all predictions.
    std::vector<double> line_accuracy;
    // True when the official early-outs fired (too slow or > #gt + 2 lanes).
    bool rejected = false;
};

// Angle (rad) of a lane from a least-squares fit x = k y + b over its valid
// points; 0 with fewer than 2 points.
double tusimple_lane_angle(const std::vector<int>& xs, const std::vector<int>& y_samples);
// Fraction of ALL h_samples (not only the labelled ones) where |pred - gt| <
// thresh after mapping missing points (< 0) of both to -100.
double tusimple_line_accuracy(const std::vector<int>& pred, const std::vector<int>& gt, double thresh);
// Per-image score. Every pred lane must have y_samples.size() entries
// (returns false otherwise, like the official "Format of lanes error").
bool tusimple_bench(const std::vector<std::vector<int>>& pred, const std::vector<std::vector<int>>& gt,
                    const std::vector<int>& y_samples, double run_time_ms, TuSimpleScore& score,
                    const TuSimpleParams& p = TuSimpleParams());

// Camera mount from the labelled lanes: the lanes of a straight road meet at
// the vanishing point of the forward direction, which fixes pitch and yaw;
// the ego lane's width in metres at unit camera height then fixes the height.
struct MountFit {
    int images_used = 0;
    int images_for_height = 0;
    double vp_u_px = 0.0;
    double vp_v_px = 0.0;
    double vp_spread_px = 0.0;        // median |v - median v|
    double lane_width_at_unit_height = 0.0;
};

// Keeps K and distortion; overwrites pitch, yaw and height. Uses at most
// max_images labels.
bool fit_mount_from_labels(const std::vector<TuSimpleLabel>& labels, int max_images, double lane_width_m,
                           CameraModel& camera, MountFit& fit, std::string& error);
