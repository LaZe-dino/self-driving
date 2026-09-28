#include "session_eval.hpp"
#include "json_out.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

namespace {

cv::FileStorage open_json(const std::string& line) {
    return cv::FileStorage(line, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    std::size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

int confidence_bin(double c) {
    return std::clamp(static_cast<int>(c * kConfidenceBins), 0, kConfidenceBins - 1);
}

void count(GoodBad& g, const std::string& human) {
    ++g.records;
    g.good += human == "good" ? 1 : 0;
    g.bad += human == "bad" ? 1 : 0;
}

std::string pct_good(const GoodBad& g) {
    long long n = g.good + g.bad;
    if (n == 0) return "    -";
    char b[16];
    std::snprintf(b, sizeof(b), "%5.1f", 100.0 * g.good / n);
    return b;
}

void print_table(const char* title, const std::map<std::string, GoodBad>& m) {
    std::printf("\n%-22s %8s %6s %6s %7s\n", title, "records", "good", "bad", "%good");
    for (const auto& e : m) {
        std::printf("%-22s %8lld %6lld %6lld %7s\n", e.first.c_str(), e.second.records, e.second.good,
                    e.second.bad, pct_good(e.second).c_str());
    }
}

void json_table(JsonOut& j, const char* k, const std::map<std::string, GoodBad>& m) {
    j.key(k).begin_obj();
    for (const auto& e : m) {
        j.key(e.first.c_str()).begin_obj();
        j.kvi("records", e.second.records).kvi("good", e.second.good).kvi("bad", e.second.bad);
        j.end_obj();
    }
    j.end_obj();
}

}  // namespace

bool parse_session_record(const std::string& line, SessionRecord& r, std::string& error) {
    try {
        cv::FileStorage f = open_json(line);
        if (!f.isOpened() || f["frame_id"].empty()) {
            error = "no frame_id";
            return false;
        }
        r = SessionRecord();
        r.frame_id = static_cast<long long>(static_cast<double>(f["frame_id"]));
        r.status = f["status"].isString() ? static_cast<std::string>(f["status"]) : "?";
        r.confidence = f["confidence"].isReal() || f["confidence"].isInt() ? static_cast<double>(f["confidence"]) : 0.0;
        r.road_valid = !f["road_state"]["valid"].empty() && static_cast<int>(f["road_state"]["valid"]) != 0;
        for (const cv::FileNode& n : f["reasons"]) r.reasons.push_back(static_cast<std::string>(n));
        cv::FileNode h = f["labels"]["human"];
        if (h.isString()) r.human = static_cast<std::string>(h);
    } catch (const cv::Exception& e) {
        error = e.what();
        return false;
    }
    return true;
}

bool parse_label_line(const std::string& line, long long& frame_id, std::string& human, std::string& error) {
    try {
        cv::FileStorage f = open_json(line);
        if (!f.isOpened() || f["frame_id"].empty() || !f["human"].isString()) {
            error = "expected {\"frame_id\":N,\"human\":\"good\"|\"bad\"}";
            return false;
        }
        frame_id = static_cast<long long>(static_cast<double>(f["frame_id"]));
        human = static_cast<std::string>(f["human"]);
    } catch (const cv::Exception& e) {
        error = e.what();
        return false;
    }
    return true;
}

SessionReport summarize_session(std::vector<SessionRecord> records, const std::map<long long, std::string>& labels) {
    SessionReport r;
    std::map<long long, bool> seen;
    for (SessionRecord& rec : records) {
        auto it = labels.find(rec.frame_id);
        if (it != labels.end()) rec.human = it->second;
        if (rec.human != "good" && rec.human != "bad") rec.human.clear();
        seen[rec.frame_id] = true;
    }
    for (const auto& l : labels) r.labels_without_record += seen.count(l.first) ? 0 : 1;

    std::vector<double> good, bad;
    for (const SessionRecord& rec : records) {
        ++r.records;
        count(r.by_status[rec.status], rec.human);
        for (const std::string& why : rec.reasons) count(r.by_reason[why], rec.human);
        int b = confidence_bin(rec.confidence);
        count(r.by_confidence[b], rec.human);
        if (rec.human.empty()) continue;
        r.conf_sum_bin[b] += rec.confidence;
        (rec.human == "good" ? good : bad).push_back(rec.confidence);
        double y = rec.human == "good" ? 1.0 : 0.0;
        r.brier += (rec.confidence - y) * (rec.confidence - y);
    }
    r.good = static_cast<long long>(good.size());
    r.bad = static_cast<long long>(bad.size());
    long long labelled = r.good + r.bad;
    if (labelled == 0) return r;
    r.brier /= labelled;
    auto mean = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x;
        return v.empty() ? 0.0 : s / v.size();
    };
    r.mean_conf_good = mean(good);
    r.mean_conf_bad = mean(bad);
    r.median_conf_good = median(good);
    r.median_conf_bad = median(bad);
    if (!good.empty() && !bad.empty()) {
        double wins = 0;
        for (double g : good) {
            for (double b : bad) wins += g > b ? 1.0 : (g == b ? 0.5 : 0.0);
        }
        r.auc = wins / (static_cast<double>(good.size()) * bad.size());
    }
    for (int b = 0; b < kConfidenceBins; ++b) {
        long long n = r.by_confidence[b].good + r.by_confidence[b].bad;
        if (n == 0) continue;
        double conf = r.conf_sum_bin[b] / n;
        double acc = static_cast<double>(r.by_confidence[b].good) / n;
        r.ece += static_cast<double>(n) / labelled * std::fabs(conf - acc);
    }
    return r;
}

void print_session_report(const SessionReport& r) {
    long long labelled = r.good + r.bad;
    std::printf("---- session evaluation (human labels) ----\n");
    std::printf("logged frames %lld, labelled %lld (good %lld, bad %lld), unlabelled %lld\n", r.records, labelled,
                r.good, r.bad, r.records - labelled);
    if (r.labels_without_record > 0) {
        std::printf("warning: %lld labels refer to frame ids not in frames.jsonl\n", r.labels_without_record);
    }
    print_table("tracker status", r.by_status);
    print_table("logging reason", r.by_reason);
    std::printf("  (a frame logged for several reasons counts once per reason)\n");

    std::printf("\n%-22s %8s %6s %6s %7s %9s\n", "confidence", "records", "good", "bad", "%good", "mean conf");
    for (int b = 0; b < kConfidenceBins; ++b) {
        char name[32];
        std::snprintf(name, sizeof(name), "[%.1f, %.1f%s", b / 5.0, (b + 1) / 5.0, b + 1 == kConfidenceBins ? "]" : ")");
        const GoodBad& g = r.by_confidence[b];
        long long n = g.good + g.bad;
        std::printf("%-22s %8lld %6lld %6lld %7s", name, g.records, g.good, g.bad, pct_good(g).c_str());
        if (n > 0) std::printf(" %9.3f\n", r.conf_sum_bin[b] / n);
        else std::printf(" %9s\n", "-");
    }
    if (labelled == 0) {
        std::printf("\nno human labels yet: run atlas --review SESSION_DIR and press g / b\n");
        return;
    }
    std::printf("\nconfidence of good frames: mean %.3f median %.3f (n=%lld)\n", r.mean_conf_good, r.median_conf_good,
                r.good);
    std::printf("confidence of bad frames:  mean %.3f median %.3f (n=%lld)\n", r.mean_conf_bad, r.median_conf_bad,
                r.bad);
    if (r.good > 0 && r.bad > 0) {
        std::printf("AUC (P(conf good > conf bad)) %.3f   (0.5 = uninformative, 1.0 = perfect ranking)\n", r.auc);
    }
    std::printf("as a probability of 'good': Brier %.3f, ECE %.3f   (0 = perfectly calibrated)\n", r.brier, r.ece);
}

std::string session_report_json(const SessionReport& r, const std::string& dir) {
    JsonOut j;
    j.begin_obj();
    j.kv("session", dir);
    j.kvi("records", r.records).kvi("good", r.good).kvi("bad", r.bad);
    j.kvi("labels_without_record", r.labels_without_record);
    json_table(j, "by_status", r.by_status);
    json_table(j, "by_reason", r.by_reason);
    j.key("by_confidence").begin_arr();
    for (int b = 0; b < kConfidenceBins; ++b) {
        const GoodBad& g = r.by_confidence[b];
        long long n = g.good + g.bad;
        j.begin_obj().kv("lo", b / 5.0).kv("hi", (b + 1) / 5.0);
        j.kvi("records", g.records).kvi("good", g.good).kvi("bad", g.bad);
        j.kv("mean_confidence", n > 0 ? r.conf_sum_bin[b] / n : NAN);
        j.end_obj();
    }
    j.end_arr();
    j.kv("mean_conf_good", r.mean_conf_good).kv("mean_conf_bad", r.mean_conf_bad);
    j.kv("median_conf_good", r.median_conf_good).kv("median_conf_bad", r.median_conf_bad);
    j.kv("auc", r.auc).kv("brier", r.brier).kv("ece", r.ece);
    j.end_obj();
    return j.text();
}

int run_session_eval(const std::string& dir, const std::string& out_json) {
    std::ifstream in(fs::path(dir) / "frames.jsonl");
    if (!in) {
        std::cerr << "no frames.jsonl in " << dir << "\n";
        return 1;
    }
    std::vector<SessionRecord> records;
    std::string line, error;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        SessionRecord r;
        if (!parse_session_record(line, r, error)) {
            std::cerr << "frames.jsonl:" << line_no << ": skipped (" << error << ")\n";
            continue;
        }
        records.push_back(r);
    }
    std::map<long long, std::string> labels;
    std::ifstream lin(fs::path(dir) / "labels.jsonl");
    bool have_labels_file = lin.is_open();
    line_no = 0;
    while (have_labels_file && std::getline(lin, line)) {
        ++line_no;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        long long id = 0;
        std::string human;
        if (!parse_label_line(line, id, human, error)) {
            std::cerr << "labels.jsonl:" << line_no << ": skipped (" << error << ")\n";
            continue;
        }
        labels[id] = human;
    }
    if (!have_labels_file) std::cout << "no labels.jsonl in " << dir << " (only labels pressed during the live run count)\n";

    SessionReport r = summarize_session(records, labels);
    print_session_report(r);
    if (!out_json.empty()) {
        std::ofstream(out_json) << session_report_json(r, dir) << "\n";
        std::cout << "wrote " << out_json << "\n";
    }
    return 0;
}
