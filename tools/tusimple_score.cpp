#include "tusimple_score.hpp"
#include "json_out.hpp"
#include <opencv2/core.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace {

std::vector<int> read_ints(const cv::FileNode& n) {
    std::vector<int> v;
    for (const cv::FileNode& e : n) v.push_back(static_cast<int>(std::lround(static_cast<double>(e))));
    return v;
}

}  // namespace

bool parse_tusimple_prediction(const std::string& json_line, TuSimplePrediction& out, std::string& error) {
    out = TuSimplePrediction();
    try {
        cv::FileStorage fs(json_line, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        if (!fs.isOpened()) {
            error = "not JSON";
            return false;
        }
        out.raw_file = static_cast<std::string>(fs["raw_file"]);
        out.h_samples = read_ints(fs["h_samples"]);
        for (const cv::FileNode& lane : fs["lanes"]) out.lanes.push_back(read_ints(lane));
        cv::FileNode rt = fs["run_time"];
        if (rt.isReal() || rt.isInt()) out.run_time_ms = static_cast<double>(rt);
    } catch (const cv::Exception& e) {
        error = e.what();
        return false;
    }
    if (out.raw_file.empty()) {
        error = "missing raw_file";
        return false;
    }
    return true;
}

bool load_tusimple_predictions(const std::string& path, std::vector<TuSimplePrediction>& out, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        TuSimplePrediction p;
        std::string e;
        if (!parse_tusimple_prediction(line, p, e)) {
            error = path + ":" + std::to_string(line_no) + ": " + e;
            return false;
        }
        out.push_back(std::move(p));
    }
    return true;
}

ScoreReport score_tusimple_predictions(const std::vector<TuSimpleLabel>& labels,
                                       const std::vector<TuSimplePrediction>& preds, const TuSimpleParams& p) {
    ScoreReport r;
    std::unordered_map<std::string, const TuSimplePrediction*> by_file;
    for (const TuSimplePrediction& q : preds) {
        auto ins = by_file.insert({q.raw_file, &q});
        if (!ins.second) {
            ++r.duplicates;
            ins.first->second = &q;
        }
    }
    std::unordered_map<std::string, bool> labelled;
    for (const TuSimpleLabel& l : labels) labelled[l.raw_file] = true;
    for (const auto& e : by_file) {
        if (!labelled.count(e.first)) r.unknown.push_back(e.first);
    }

    long long scored = 0;
    for (const TuSimpleLabel& l : labels) {
        ScoredImage s;
        s.raw_file = l.raw_file;
        s.n_gt = static_cast<int>(l.lanes.size());
        auto it = by_file.find(l.raw_file);
        bool ok = false;
        if (it == by_file.end()) {
            s.status = "missing";
            ++r.missing;
        } else {
            const TuSimplePrediction& q = *it->second;
            s.n_pred = static_cast<int>(q.lanes.size());
            s.run_time_ms = q.run_time_ms;
            bool rows_match = q.h_samples.empty() || q.h_samples == l.h_samples;
            ok = rows_match && tusimple_bench(q.lanes, l.lanes, l.h_samples, q.run_time_ms, s.score, p);
            if (!ok) {
                s.status = "format";
                ++r.format_errors;
            }
        }
        if (!ok) {
            tusimple_bench({}, l.lanes, l.h_samples, 0.0, s.score, p);
        } else {
            ++scored;
            r.rejected += s.score.rejected ? 1 : 0;
            r.accuracy_scored += s.score.accuracy;
            r.fp_scored += s.score.fp;
            r.fn_scored += s.score.fn;
        }
        r.accuracy += s.score.accuracy;
        r.fp += s.score.fp;
        r.fn += s.score.fn;
        r.images.push_back(std::move(s));
    }
    double n = static_cast<double>(labels.size());
    if (n > 0) {
        r.accuracy /= n;
        r.fp /= n;
        r.fn /= n;
    }
    if (scored > 0) {
        r.accuracy_scored /= scored;
        r.fp_scored /= scored;
        r.fn_scored /= scored;
    }
    return r;
}

int run_tusimple_score(const std::string& pred_path, const std::vector<std::string>& label_files,
                       const std::string& out_json) {
    std::vector<TuSimpleLabel> labels;
    for (const std::string& f : label_files) {
        std::string error;
        std::size_t before = labels.size();
        if (!load_tusimple_labels(f, labels, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cout << f << ": " << labels.size() - before << " labelled images\n";
    }
    if (labels.empty()) {
        std::cerr << "no labels\n";
        return 1;
    }
    std::vector<TuSimplePrediction> preds;
    std::string error;
    if (!load_tusimple_predictions(pred_path, preds, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << pred_path << ": " << preds.size() << " predictions\n";

    ScoreReport r = score_tusimple_predictions(labels, preds);
    const int kShow = 5;
    auto list = [&](const char* what, const std::string& status) {
        int shown = 0;
        for (const ScoredImage& s : r.images) {
            if (s.status != status) continue;
            if (shown++ == kShow) {
                std::printf("    ...\n");
                break;
            }
            std::printf("    %s %s\n", what, s.raw_file.c_str());
        }
    };
    std::printf("\n==== TuSimple score: %zu labelled images ====\n", r.images.size());
    std::printf("official metric (port of tusimple-benchmark lane.py); missing or malformed predictions score as "
                "no lanes\n");
    std::printf("  Accuracy  %6.2f %%\n", 100.0 * r.accuracy);
    std::printf("  FP        %.4f\n", r.fp);
    std::printf("  FN        %.4f\n", r.fn);
    if (r.missing > 0) {
        std::printf("  %lld labelled images have no prediction:\n", r.missing);
        list("missing", "missing");
    }
    if (r.format_errors > 0) {
        std::printf("  %lld predictions have lanes or h_samples that do not match the label rows:\n",
                    r.format_errors);
        list("format", "format");
    }
    if (r.missing + r.format_errors > 0) {
        std::printf("  over the %zu images with a valid prediction: Accuracy %6.2f %%  FP %.4f  FN %.4f\n",
                    r.images.size() - static_cast<std::size_t>(r.missing + r.format_errors),
                    100.0 * r.accuracy_scored, r.fp_scored, r.fn_scored);
    }
    if (r.rejected > 0) std::printf("  %lld images hit the official early-out (> 200 ms or > #gt+2 lanes)\n", r.rejected);
    if (r.duplicates > 0) std::printf("  %lld duplicate raw_file lines (last one used)\n", r.duplicates);
    if (!r.unknown.empty()) {
        std::printf("  %zu predictions have no label (ignored), e.g. %s\n", r.unknown.size(), r.unknown[0].c_str());
    }

    if (!out_json.empty()) {
        JsonOut j;
        j.begin_obj();
        j.kv("dataset", std::string("tusimple")).kv("predictions", pred_path);
        j.key("summary").begin_obj();
        j.kvi("images", static_cast<long long>(r.images.size())).kvi("predictions", static_cast<long long>(preds.size()));
        j.kvi("missing", r.missing).kvi("format_errors", r.format_errors).kvi("rejected", r.rejected);
        j.kvi("duplicates", r.duplicates).kvi("unknown", static_cast<long long>(r.unknown.size()));
        j.kv("accuracy", r.accuracy).kv("fp", r.fp).kv("fn", r.fn);
        j.kv("accuracy_scored", r.accuracy_scored).kv("fp_scored", r.fp_scored).kv("fn_scored", r.fn_scored);
        j.end_obj();
        j.key("images").begin_arr();
        for (const ScoredImage& s : r.images) {
            j.begin_obj().kv("raw_file", s.raw_file).kv("status", s.status);
            j.kv("accuracy", s.score.accuracy).kv("fp", s.score.fp).kv("fn", s.score.fn);
            j.kvi("n_pred", s.n_pred).kvi("n_gt", s.n_gt).kvi("matched", s.score.matched);
            j.kv("run_ms", s.run_time_ms).kvb("rejected", s.score.rejected);
            j.end_obj();
        }
        j.end_arr();
        j.end_obj();
        std::ofstream(out_json) << j.text() << "\n";
        std::cout << "wrote " << out_json << "\n";
    }
    return 0;
}
