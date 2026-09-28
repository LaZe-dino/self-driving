#pragma once

#include "tusimple.hpp"
#include <string>
#include <vector>

// One line of an externally produced prediction file (official TuSimple
// submission format: raw_file, lanes, run_time). h_samples is optional; any
// other keys (e.g. training/predict.py's "slots", "frame_id") are ignored.
struct TuSimplePrediction {
    std::string raw_file;
    std::vector<int> h_samples;  // empty when the line has none
    std::vector<std::vector<int>> lanes;
    double run_time_ms = 0.0;
};

bool parse_tusimple_prediction(const std::string& json_line, TuSimplePrediction& out, std::string& error);
bool load_tusimple_predictions(const std::string& path, std::vector<TuSimplePrediction>& out, std::string& error);

struct ScoredImage {
    std::string raw_file;
    TuSimpleScore score;
    int n_pred = 0;
    int n_gt = 0;
    double run_time_ms = 0.0;
    // "ok", "missing" (no prediction line) or "format" (lane length or
    // h_samples differ from the label). Both failures score as no lanes.
    std::string status = "ok";
};

struct ScoreReport {
    std::vector<ScoredImage> images;  // one per label, in label order
    long long missing = 0;
    long long format_errors = 0;
    long long rejected = 0;           // official early-out (> 200 ms or > #gt + 2 lanes)
    long long duplicates = 0;         // repeated raw_file in the predictions (last one wins)
    std::vector<std::string> unknown; // predicted raw_files with no label
    double accuracy = 0.0, fp = 0.0, fn = 0.0;          // over all labelled images
    double accuracy_scored = 0.0, fp_scored = 0.0, fn_scored = 0.0;  // images with a valid prediction only
};

// Matches predictions to labels by raw_file and applies the official metric.
ScoreReport score_tusimple_predictions(const std::vector<TuSimpleLabel>& labels,
                                       const std::vector<TuSimplePrediction>& preds,
                                       const TuSimpleParams& p = TuSimpleParams());

int run_tusimple_score(const std::string& pred_path, const std::vector<std::string>& label_files,
                       const std::string& out_json);
