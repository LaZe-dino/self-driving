#pragma once

#include <map>
#include <string>
#include <vector>

// Human review of a logged session (atlas --log, then atlas --review):
// frames.jsonl holds one record per logged frame, labels.jsonl one
// {"frame_id":N,"human":"good"|"bad"} per key press (later lines win).
struct SessionRecord {
    long long frame_id = -1;
    std::string status;
    double confidence = 0.0;
    bool road_valid = false;
    std::vector<std::string> reasons;
    std::string human;  // "good", "bad" or empty
};

bool parse_session_record(const std::string& json_line, SessionRecord& out, std::string& error);
bool parse_label_line(const std::string& json_line, long long& frame_id, std::string& human, std::string& error);

struct GoodBad {
    long long records = 0;
    long long good = 0;
    long long bad = 0;
};

constexpr int kConfidenceBins = 5;

struct SessionReport {
    long long records = 0;
    long long good = 0;
    long long bad = 0;
    long long labels_without_record = 0;
    std::map<std::string, GoodBad> by_status;
    std::map<std::string, GoodBad> by_reason;
    GoodBad by_confidence[kConfidenceBins];  // [0, 0.2), [0.2, 0.4), ... [0.8, 1]
    double conf_sum_bin[kConfidenceBins] = {};  // over labelled records
    double mean_conf_good = 0.0;
    double mean_conf_bad = 0.0;
    double median_conf_good = 0.0;
    double median_conf_bad = 0.0;
    // P(confidence of a random good frame > that of a random bad frame),
    // ties count 1/2. 0.5 = confidence says nothing, 1 = perfect ranking.
    double auc = 0.5;
    // Treating confidence as P(good): mean (conf - y)^2 and expected
    // calibration error over the bins.
    double brier = 0.0;
    double ece = 0.0;
};

// Labels override the "human" field stored in the records.
SessionReport summarize_session(std::vector<SessionRecord> records, const std::map<long long, std::string>& labels);
void print_session_report(const SessionReport& r);
std::string session_report_json(const SessionReport& r, const std::string& dir);

// Reads DIR/frames.jsonl and DIR/labels.jsonl (optional), prints the report
// and writes it to out_json if not empty. Returns a process exit code.
int run_session_eval(const std::string& dir, const std::string& out_json);
