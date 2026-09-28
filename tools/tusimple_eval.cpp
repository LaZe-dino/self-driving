#include "tusimple_eval.hpp"
#include "json_out.hpp"
#include "../vision/cv_compat.hpp"
#include "../vision/pipeline.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

namespace fs = std::filesystem;

namespace {

const double kLateralDistances[] = {10.0, 20.0, 30.0};
constexpr int kDistances = 3;
// The pipeline itself only searches for an outer line seen within 1 s.
const double kOuterMemoryS = 1.0;

void print_camera(const CameraModel& m) {
    std::printf("  image %dx%d  fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n", m.image_size.width, m.image_size.height,
                m.K(0, 0), m.K(1, 1), m.K(0, 2), m.K(1, 2));
    std::printf("  height=%.3f m  pitch=%.3f deg  yaw=%.3f deg\n", m.height_m, m.pitch_rad * 180.0 / CV_PI,
                m.yaw_rad * 180.0 / CV_PI);
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[static_cast<std::size_t>(std::lround(p * (v.size() - 1)))];
}

double mean(const std::vector<double>& v) {
    if (v.empty()) return NAN;
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

struct ImageResult {
    std::string raw_file;
    TuSimpleScore score;
    int n_pred = 0;
    int n_gt = 0;
    std::string status = "MISSING";
    double confidence = 0.0;
    bool locked_any = false;
    double run_ms = 0.0;
    // Per distance: [left, right] |error| in m, NaN when not measurable.
    double lat_err[kDistances][2];
};

struct StatusAcc {
    long long n = 0;
    double acc = 0.0;
};

// Frame number of a TuSimple clip image ("20.jpg" -> 20), 0 if not numeric.
int frame_number(const fs::path& p) {
    std::string s = p.stem().string();
    if (s.empty() || !std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; })) return 0;
    return std::stoi(s);
}

bool load_all(const std::vector<std::string>& files, std::vector<TuSimpleLabel>& out) {
    for (const std::string& f : files) {
        std::string error;
        std::size_t before = out.size();
        if (!load_tusimple_labels(f, out, error)) {
            std::cerr << error << "\n";
            return false;
        }
        std::cout << f << ": " << out.size() - before << " labelled images\n";
    }
    return true;
}

}  // namespace

std::vector<std::vector<int>> predict_tusimple_lanes(const RoadState& road, const CameraModel& camera,
                                                     const std::vector<int>& rows, const RasterizeOptions& opt,
                                                     const bool use_slot[kLaneSlots], int min_points) {
    std::vector<std::vector<int>> lanes;
    if (!road.valid) return lanes;
    for (int s = 0; s < kLaneSlots; ++s) {
        if (!use_slot[s]) continue;
        std::vector<int> xs = rasterize_boundary_raw(road, slot_width_multiple(static_cast<LaneSlot>(s)), camera,
                                                     rows, opt);
        int drawn = static_cast<int>(std::count_if(xs.begin(), xs.end(), [](int x) { return x >= 0; }));
        if (drawn >= std::max(min_points, 1)) lanes.push_back(std::move(xs));
    }
    return lanes;
}

bool label_lateral_at(const std::vector<int>& xs, const std::vector<int>& rows, const CameraModel& camera,
                      double X, double& Y) {
    std::vector<cv::Point2d> px;
    for (std::size_t i = 0; i < xs.size() && i < rows.size(); ++i) {
        if (xs[i] >= 0) px.push_back({static_cast<double>(xs[i]), static_cast<double>(rows[i])});
    }
    if (px.size() < 2) return false;
    if (has_distortion(camera)) {
        std::vector<cv::Point2d> und;
        cv::undistortPoints(px, und, camera.K, camera.dist, cv::noArray(), camera.K);
        px = und;
    }
    std::vector<cv::Point2d> g;
    for (const cv::Point2d& p : px) {
        double gx, gy;
        if (camera.image_to_ground(p, gx, gy)) g.push_back({gx, gy});
    }
    std::sort(g.begin(), g.end(), [](const cv::Point2d& a, const cv::Point2d& b) { return a.x < b.x; });
    for (std::size_t i = 0; i + 1 < g.size(); ++i) {
        if (g[i].x <= X && X <= g[i + 1].x && g[i + 1].x - g[i].x > 1e-9) {
            double a = (X - g[i].x) / (g[i + 1].x - g[i].x);
            Y = g[i].y + a * (g[i + 1].y - g[i].y);
            return true;
        }
    }
    return false;
}

int run_tusimple_eval(const TuSimpleEvalOptions& o) {
    std::vector<TuSimpleLabel> labels;
    if (!load_all(o.label_files, labels)) return 1;
    if (labels.empty()) {
        std::cerr << "no labels\n";
        return 1;
    }
    std::vector<TuSimpleLabel> fit_labels;
    if (!o.fit_label_files.empty() && !load_all(o.fit_label_files, fit_labels)) return 1;
    std::vector<TuSimpleLabel> eval = labels;
    if (o.limit > 0 && static_cast<long long>(eval.size()) > o.limit) eval.resize(static_cast<std::size_t>(o.limit));

    // ---- camera ----
    fs::path root(o.root);
    cv::Mat first = cv::imread((root / labels.front().raw_file).string());
    if (first.empty()) {
        std::cerr << "cannot read " << (root / labels.front().raw_file).string()
                  << " (is --tusimple the folder that contains clips/?)\n";
        return 1;
    }
    CameraModel camera;
    std::string error;
    if (o.camera_model_path.empty()) {
        camera = default_camera_model(first.size(), o.fov_deg);
    } else if (!load_camera_model(o.camera_model_path, camera, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (camera.image_size != first.size()) {
        std::cerr << "camera model is " << camera.image_size.width << "x" << camera.image_size.height
                  << " but images are " << first.cols << "x" << first.rows << "\n";
        return 1;
    }
    bool fit = o.fit_mount || (o.camera_model_path.empty() && !o.no_fit_mount);
    MountFit mf;
    bool fitted = false;
    if (fit) {
        const std::vector<TuSimpleLabel>& src = fit_labels.empty() ? labels : fit_labels;
        fitted = fit_mount_from_labels(src, o.fit_images, o.lane_width_m, camera, mf, error);
        if (fitted) {
            std::printf("mount fitted from %d labelled images (%d for height): vanishing point (%.1f, %.1f) px, "
                        "spread %.1f px, lane width at h=1: %.3f\n",
                        mf.images_used, mf.images_for_height, mf.vp_u_px, mf.vp_v_px, mf.vp_spread_px,
                        mf.lane_width_at_unit_height);
            camera.source += " + mount fitted to TuSimple labels";
        } else if (o.fit_mount) {
            std::cerr << "mount fit failed: " << error << "\n";
            return 1;
        } else {
            std::cerr << "warning: mount fit failed (" << error << "); using the default mount\n";
        }
    }
    std::cout << "camera: " << camera.source << "\n";
    print_camera(camera);
    if (!o.save_camera_path.empty()) {
        if (!save_camera_model(o.save_camera_path, camera, error)) {
            std::cerr << error << "\n";
            return 1;
        }
        std::cout << "wrote " << o.save_camera_path << "\n";
    }

    std::ofstream pred_out;
    if (!o.predictions_path.empty()) {
        pred_out.open(o.predictions_path);
        if (!pred_out) {
            std::cerr << "cannot write " << o.predictions_path << "\n";
            return 1;
        }
    }

    // ---- clips ----
    std::vector<ImageResult> results;
    long long missing = 0;
    long long frames_run = 0;
    double proc_ms = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t li = 0; li < eval.size(); ++li) {
        const TuSimpleLabel& l = eval[li];
        ImageResult r;
        r.raw_file = l.raw_file;
        r.n_gt = static_cast<int>(l.lanes.size());
        for (auto& d : r.lat_err) d[0] = d[1] = NAN;

        fs::path labelled = root / l.raw_file;
        int last = frame_number(labelled);
        VisionPipeline pipe;
        pipe.configure(camera, GroundGrid());
        if (o.objects && !pipe.enable_objects(o.object_params, error)) {
            std::cerr << "objects: " << error << "\n";
            return 1;
        }
        PerceptionFrame pf;
        double outer_seen[kLaneSlots] = {-1e9, -1e9, -1e9, -1e9};
        bool ok = true;
        // Warm the tracker up on the clip's unlabelled frames 1..N-1.
        for (int i = last > 0 ? 1 : 0; i <= last && ok; ++i) {
            fs::path p = last > 0 ? labelled.parent_path() / (std::to_string(i) + labelled.extension().string())
                                  : labelled;
            Frame f;
            f.id = i;
            f.media_time_s = f.capture_time_s = i / o.fps;
            f.image = cv::imread(p.string());
            if (f.image.empty()) {
                std::cerr << "missing " << p.string() << "\n";
                ok = false;
                break;
            }
            if (f.image.size() != camera.image_size) {
                std::cerr << p.string() << " is " << f.image.cols << "x" << f.image.rows << ", camera model is "
                          << camera.image_size.width << "x" << camera.image_size.height << "\n";
                return 1;
            }
            pipe.process(f, pf);
            ++frames_run;
            proc_ms += pf.timings.total_ms;
            r.locked_any = r.locked_any || pf.status == TrackStatus::Locked;
            for (const LaneMeasurement& m : pf.detection.lanes) {
                if (m.accepted) outer_seen[static_cast<int>(m.slot)] = f.media_time_s;
            }
        }

        std::vector<std::vector<int>> pred;
        if (!ok) {
            ++missing;
        } else {
            r.status = status_name(pf.status);
            r.confidence = pf.confidence;
            r.run_ms = pf.timings.total_ms;
            // The road state was measured with the pitch in effect for this
            // frame, before the horizon estimator's update.
            CameraModel cam = pipe.camera();
            cam.pitch_rad = pf.pitch_rad;
            cam.update();
            bool use[kLaneSlots] = {true, true, false, false};
            for (LaneSlot s : {LaneSlot::LeftOuter, LaneSlot::RightOuter}) {
                int k = static_cast<int>(s);
                use[k] = o.all_slots || pf.media_time_s - outer_seen[k] < kOuterMemoryS;
            }
            pred = predict_tusimple_lanes(pf.road, cam, l.h_samples, o.raster, use, o.min_points);

            for (int d = 0; d < kDistances; ++d) {
                double X = kLateralDistances[d];
                double gl = 1e9, gr = -1e9;
                for (const std::vector<int>& lane : l.lanes) {
                    double Y;
                    if (!label_lateral_at(lane, l.h_samples, cam, X, Y)) continue;
                    if (Y > 0) gl = std::min(gl, Y);
                    else gr = std::max(gr, Y);
                }
                const double gt[2] = {gl, gr};
                const double k[2] = {slot_width_multiple(LaneSlot::Left), slot_width_multiple(LaneSlot::Right)};
                for (int side = 0; side < 2; ++side) {
                    if (std::fabs(gt[side]) > 1e8) continue;
                    // Inf = labelled but not predicted (sigma too large or no track).
                    bool predicted = pf.road.valid && pf.road.lateral_sigma(X, k[side]) < o.raster.max_sigma_m;
                    r.lat_err[d][side] = predicted ? std::fabs(pf.road.lateral(X, k[side]) - gt[side]) : INFINITY;
                }
            }
        }
        r.n_pred = static_cast<int>(pred.size());
        if (!tusimple_bench(pred, l.lanes, l.h_samples, r.run_ms, r.score)) {
            std::cerr << "internal error: prediction length mismatch for " << l.raw_file << "\n";
            return 1;
        }
        if (pred_out) {
            JsonOut j;
            j.begin_obj().kv("raw_file", l.raw_file).key("lanes").begin_arr();
            for (const std::vector<int>& lane : pred) {
                j.begin_arr();
                for (int x : lane) j.integer(x);
                j.end_arr();
            }
            j.end_arr().kv("run_time", r.run_ms).end_obj();
            pred_out << j.text() << "\n";
        }
        results.push_back(r);

        if ((li + 1) % 100 == 0 || li + 1 == eval.size()) {
            double acc = 0;
            for (const ImageResult& x : results) acc += x.score.accuracy;
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("[eval] %zu/%zu images  accuracy so far %.2f%%  (%.1f s)\n", li + 1, eval.size(),
                        100.0 * acc / results.size(), el);
            std::fflush(stdout);
        }
    }

    // ---- summary ----
    double n = static_cast<double>(results.size());
    double acc = 0, fp = 0, fn = 0;
    long long locked_last = 0, locked_any = 0, rejected = 0, n_pred = 0, n_gt = 0;
    std::map<std::string, StatusAcc> by_status;
    std::vector<double> err[kDistances];
    long long lat_gt[kDistances] = {}, lat_pred[kDistances] = {};
    for (const ImageResult& r : results) {
        acc += r.score.accuracy;
        fp += r.score.fp;
        fn += r.score.fn;
        locked_last += r.status == "LOCKED" ? 1 : 0;
        locked_any += r.locked_any ? 1 : 0;
        rejected += r.score.rejected ? 1 : 0;
        n_pred += r.n_pred;
        n_gt += r.n_gt;
        StatusAcc& s = by_status[r.status];
        ++s.n;
        s.acc += r.score.accuracy;
        for (int d = 0; d < kDistances; ++d) {
            for (int side = 0; side < 2; ++side) {
                double e = r.lat_err[d][side];
                if (std::isnan(e)) continue;
                ++lat_gt[d];
                if (std::isfinite(e)) {
                    ++lat_pred[d];
                    err[d].push_back(e);
                }
            }
        }
    }
    std::printf("\n==== TuSimple evaluation: %zu labelled images (%lld clips missing frames) ====\n",
                results.size(), missing);
    std::printf("official metric (port of tusimple-benchmark lane.py)\n");
    std::printf("  Accuracy  %6.2f %%\n", 100.0 * acc / n);
    std::printf("  FP        %.4f\n", fp / n);
    std::printf("  FN        %.4f\n", fn / n);
    if (rejected > 0) std::printf("  %lld images hit the official early-out (> 200 ms or > #gt+2 lanes)\n", rejected);
    std::printf("  lanes per image: predicted %.2f, labelled %.2f\n", n_pred / n, n_gt / n);
    std::printf("tracker at the labelled frame:\n");
    for (const auto& e : by_status) {
        std::printf("  %-10s %6.1f %% of clips   accuracy %6.2f %%\n", e.first.c_str(), 100.0 * e.second.n / n,
                    100.0 * e.second.acc / e.second.n);
    }
    std::printf("  LOCKED at the labelled frame %.1f %%, LOCKED at any frame of the clip %.1f %%\n",
                100.0 * locked_last / n, 100.0 * locked_any / n);
    std::printf("ego-lane lateral error vs labels back-projected to the ground (flat road, camera above):\n");
    std::printf("  %6s %9s %9s %9s %9s %9s\n", "ahead", "labelled", "predicted", "mean m", "median m", "p90 m");
    for (int d = 0; d < kDistances; ++d) {
        std::printf("  %4.0f m %9lld %8.1f%% %9.3f %9.3f %9.3f\n", kLateralDistances[d], lat_gt[d],
                    lat_gt[d] ? 100.0 * lat_pred[d] / lat_gt[d] : 0.0, mean(err[d]), percentile(err[d], 0.5),
                    percentile(err[d], 0.9));
    }
    if (frames_run > 0) std::printf("pipeline %.2f ms per frame (mean over %lld frames)\n", proc_ms / frames_run,
                                    frames_run);

    if (!o.out_json.empty()) {
        JsonOut j;
        j.begin_obj();
        j.kv("dataset", std::string("tusimple")).kv("root", o.root);
        j.key("camera").begin_obj();
        j.kv("source", camera.source).kvi("width", camera.image_size.width).kvi("height", camera.image_size.height);
        j.kv("fx", camera.K(0, 0)).kv("fy", camera.K(1, 1)).kv("cx", camera.K(0, 2)).kv("cy", camera.K(1, 2));
        j.kv("height_m", camera.height_m).kv("pitch_rad", camera.pitch_rad).kv("yaw_rad", camera.yaw_rad);
        j.kvb("mount_fitted", fitted);
        if (fitted) {
            j.kvi("fit_images", mf.images_used).kv("vp_u_px", mf.vp_u_px).kv("vp_v_px", mf.vp_v_px);
            j.kv("vp_spread_px", mf.vp_spread_px);
        }
        j.end_obj();
        j.key("summary").begin_obj();
        j.kvi("images", static_cast<long long>(results.size())).kvi("missing_clips", missing);
        j.kv("accuracy", acc / n).kv("fp", fp / n).kv("fn", fn / n);
        j.kv("locked_at_label", locked_last / n).kv("locked_any", locked_any / n);
        j.kv("pred_lanes_per_image", n_pred / n).kv("gt_lanes_per_image", n_gt / n);
        j.key("lateral_error").begin_arr();
        for (int d = 0; d < kDistances; ++d) {
            j.begin_obj().kv("x_m", kLateralDistances[d]).kvi("labelled", lat_gt[d]).kvi("predicted", lat_pred[d]);
            j.kv("mean_m", mean(err[d])).kv("median_m", percentile(err[d], 0.5)).kv("p90_m", percentile(err[d], 0.9));
            j.end_obj();
        }
        j.end_arr();
        j.end_obj();
        j.key("images").begin_arr();
        for (const ImageResult& r : results) {
            j.begin_obj().kv("raw_file", r.raw_file);
            j.kv("accuracy", r.score.accuracy).kv("fp", r.score.fp).kv("fn", r.score.fn);
            j.kvi("n_pred", r.n_pred).kvi("n_gt", r.n_gt).kvi("matched", r.score.matched);
            j.kv("status", r.status).kv("confidence", r.confidence).kvb("locked_any", r.locked_any);
            j.kv("run_ms", r.run_ms);
            // null = no label at that distance; the string "missed" = labelled
            // but no confident prediction.
            j.key("lateral_error_m").begin_obj();
            for (int d = 0; d < kDistances; ++d) {
                char k[16];
                std::snprintf(k, sizeof(k), "%.0f", kLateralDistances[d]);
                j.key(k).begin_arr();
                for (int side = 0; side < 2; ++side) {
                    double e = r.lat_err[d][side];
                    if (std::isinf(e)) j.str("missed");
                    else j.num(e);
                }
                j.end_arr();
            }
            j.end_obj();
            j.end_obj();
        }
        j.end_arr();
        j.end_obj();
        std::ofstream(o.out_json) << j.text() << "\n";
        std::cout << "wrote " << o.out_json << "\n";
    }
    if (pred_out) std::cout << "wrote " << o.predictions_path << " (official submission format)\n";
    return 0;
}
