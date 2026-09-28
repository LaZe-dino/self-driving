#include "app.hpp"
#include "calibration.hpp"
#include "camera_model.hpp"
#include "dashboard.hpp"
#include "data_logger.hpp"
#include "frame_source.hpp"
#include "pipeline.hpp"
#include "process_stats.hpp"
#include "review.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string mode;
    SourceSpec source;
    std::string camera_model_path;
    std::string out_path;
    std::string calib_dir;
    std::string review_dir;
    bool headless = false;
    long long max_frames = -1;
    double lane_width_m = 3.7;
    double fov_deg = 70.0;
    std::string snapshot_dir;
    long long snapshot_every = 100;
    bool show_flow = false;
    bool objects = true;
    ObjectDetectorParams object_params;
    bool log = false;
    DataLoggerConfig logger;
};

void usage() {
    std::cout <<
        "Atlas Vision\n"
        "  atlas --webcam [--camera-index N]            live camera\n"
        "  atlas --video FILE [--loop]                  recorded video\n"
        "     --camera-model FILE.yml   calibration (default: --fov 70 deg, level, 1.4 m high)\n"
        "     --headless                no window; prints telemetry and a run summary\n"
        "     --max-frames N            stop after N frames (default: run until the source ends)\n"
        "     --snapshots DIR [--snapshot-every N]   save dashboard images\n"
        "     --object-model FILE.onnx | --no-objects   (default data/models/yolox_tiny.onnx)\n"
        "     --flow                    draw optical-flow vectors\n"
        "     --log [--log-dir DIR] [--log-budget-mb MB]   record training data\n"
        "  atlas --review SESSION_DIR [--snapshots DIR] browse / label logged training data\n"
        "  atlas --calibrate DIR --out FILE.yml         chessboard intrinsics (9x6 inner corners)\n"
        "  atlas --estimate-mount FILE [--camera-model IN.yml | --fov DEG] --out OUT.yml [--lane-width 3.7]\n";
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&](std::string& dst) -> bool {
            if (i + 1 >= argc) {
                std::cerr << a << " needs a value\n";
                return false;
            }
            dst = argv[++i];
            return true;
        };
        std::string v;
        if (a == "--webcam") {
            o.mode = "run";
            o.source.live = true;
        } else if (a == "--video") {
            o.mode = "run";
            if (!value(o.source.path)) return false;
        } else if (a == "--loop") {
            o.source.loop = true;
        } else if (a == "--camera-index") {
            if (!value(v)) return false;
            o.source.camera_index = std::stoi(v);
        } else if (a == "--camera-model") {
            if (!value(o.camera_model_path)) return false;
        } else if (a == "--headless") {
            o.headless = true;
        } else if (a == "--max-frames") {
            if (!value(v)) return false;
            o.max_frames = std::stoll(v);
        } else if (a == "--snapshots") {
            if (!value(o.snapshot_dir)) return false;
        } else if (a == "--snapshot-every") {
            if (!value(v)) return false;
            o.snapshot_every = std::stoll(v);
        } else if (a == "--no-objects") {
            o.objects = false;
        } else if (a == "--object-model") {
            if (!value(o.object_params.model_path)) return false;
        } else if (a == "--flow") {
            o.show_flow = true;
        } else if (a == "--log") {
            o.log = true;
        } else if (a == "--log-dir") {
            if (!value(o.logger.root)) return false;
        } else if (a == "--log-budget-mb") {
            if (!value(v)) return false;
            o.logger.max_total_mb = std::stod(v);
            o.logger.max_session_mb = std::min(o.logger.max_session_mb, 0.5 * o.logger.max_total_mb);
        } else if (a == "--review") {
            o.mode = "review";
            if (!value(o.review_dir)) return false;
        } else if (a == "--calibrate") {
            o.mode = "calibrate";
            if (!value(o.calib_dir)) return false;
        } else if (a == "--estimate-mount") {
            o.mode = "estimate-mount";
            if (!value(o.source.path)) return false;
        } else if (a == "--out") {
            if (!value(o.out_path)) return false;
        } else if (a == "--fov") {
            if (!value(v)) return false;
            o.fov_deg = std::stod(v);
        } else if (a == "--lane-width") {
            if (!value(v)) return false;
            o.lane_width_m = std::stod(v);
        } else if (a == "--help" || a == "-h") {
            o.mode = "help";
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return false;
        }
    }
    return true;
}

void print_model(const CameraModel& m) {
    std::printf("  image %dx%d  fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n", m.image_size.width,
                m.image_size.height, m.K(0, 0), m.K(1, 1), m.K(0, 2), m.K(1, 2));
    std::printf("  height=%.3f m  pitch=%.3f deg  yaw=%.3f deg\n", m.height_m,
                m.pitch_rad * 180.0 / CV_PI, m.yaw_rad * 180.0 / CV_PI);
}

int run_calibrate(const Options& o) {
    if (o.out_path.empty()) {
        std::cerr << "--calibrate needs --out FILE.yml\n";
        return 1;
    }
    CameraModel model;
    ChessboardResult r;
    std::string error;
    if (!calibrate_from_chessboards(o.calib_dir, 9, 6, model, r, error)) {
        std::cerr << "calibration failed: " << error << "\n";
        return 1;
    }
    std::printf("used %d/%d images, RMS reprojection error %.3f px\n", r.images_used,
                r.images_total, r.rms_reprojection_px);
    print_model(model);
    if (!save_camera_model(o.out_path, model, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << "wrote " << o.out_path << "\n";
    return 0;
}

int run_estimate_mount(const Options& o) {
    if (o.out_path.empty()) {
        std::cerr << "--estimate-mount needs --out OUT.yml\n";
        return 1;
    }
    CameraModel model;
    std::string error;
    if (o.camera_model_path.empty()) {
        cv::VideoCapture cap;
        cv::Mat first;
        if (!open_video_file(cap, o.source.path) || !cap.read(first)) {
            std::cerr << "cannot read " << o.source.path << "\n";
            return 1;
        }
        model = default_camera_model(first.size(), o.fov_deg);
        std::cout << "no --camera-model: assuming " << o.fov_deg
                  << " deg horizontal FOV and no lens distortion (distances will be approximate)\n";
    } else if (!load_camera_model(o.camera_model_path, model, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    MountEstimate e;
    int frames = o.max_frames > 0 ? static_cast<int>(o.max_frames) : 300;
    if (!estimate_mount_from_video(o.source.path, o.lane_width_m, frames, model, e, error)) {
        std::cerr << "mount estimation failed: " << error << "\n";
        return 1;
    }
    std::printf("frames with both lane lines: %d\n", e.frames_used);
    std::printf("vanishing point (%.1f, %.1f) px, median spread %.1f px\n", e.vp_u_px,
                e.vp_v_px, e.vp_spread_px);
    std::printf("lane width at h=1: %.3f -> height = %.2f / %.3f\n",
                e.lane_width_at_unit_height, o.lane_width_m, e.lane_width_at_unit_height);
    print_model(model);
    if (!save_camera_model(o.out_path, model, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << "wrote " << o.out_path << "\n";
    return 0;
}

// Numbers that say whether perception was stable, not just whether it ran.
struct RunStats {
    long long frames = 0;
    long long per_status[4] = {0, 0, 0, 0};
    long long resets = 0;
    long long lane_changes = 0;
    long long measurements = 0;
    long long rejected = 0;
    long long speed_frames = 0;
    double conf_sum = 0.0;
    double d_offset_sq = 0.0;
    double d_curv_sq = 0.0;
    long long d_count = 0;
    std::vector<double> proc_ms;
    bool have_prev = false;
    double prev_offset = 0.0;
    double prev_curv = 0.0;

    void add(const PerceptionFrame& f) {
        ++frames;
        ++per_status[static_cast<int>(f.status)];
        conf_sum += f.confidence;
        proc_ms.push_back(f.timings.total_ms);
        speed_frames += f.speed_measured ? 1 : 0;
        for (const LaneMeasurement& m : f.detection.lanes) {
            ++measurements;
            rejected += m.accepted ? 0 : 1;
        }
        bool changed_lane = false;
        for (const TrackEvent& e : f.events) {
            resets += e.what.rfind("reset", 0) == 0 ? 1 : 0;
            bool lc = e.what.rfind("lane change", 0) == 0;
            lane_changes += lc ? 1 : 0;
            changed_lane = changed_lane || lc;
        }
        if (f.geometry.valid) {
            if (have_prev && !changed_lane) {
                double a = f.geometry.lateral_offset_m - prev_offset;
                double b = f.geometry.curvature_1pm - prev_curv;
                d_offset_sq += a * a;
                d_curv_sq += b * b;
                ++d_count;
            }
            prev_offset = f.geometry.lateral_offset_m;
            prev_curv = f.geometry.curvature_1pm;
            have_prev = true;
        } else {
            have_prev = false;
        }
    }

    void print(double wall_s) {
        if (frames == 0) return;
        std::sort(proc_ms.begin(), proc_ms.end());
        auto pct = [&](double p) { return proc_ms[static_cast<std::size_t>(p * (proc_ms.size() - 1))]; };
        auto share = [&](TrackStatus s) { return 100.0 * per_status[static_cast<int>(s)] / frames; };
        std::printf("---- run summary ----\n");
        std::printf("frames %lld in %.1f s wall (%.1f fps)\n", frames, wall_s, frames / wall_s);
        std::printf("status: LOCKED %.1f%%  PARTIAL %.1f%%  COASTING %.1f%%  SEARCHING %.1f%%\n",
                    share(TrackStatus::Locked), share(TrackStatus::Partial), share(TrackStatus::Coasting),
                    share(TrackStatus::Searching));
        std::printf("mean confidence %.3f   resets %lld   lane changes %lld\n", conf_sum / frames, resets,
                    lane_changes);
        std::printf("measurements %lld, rejected by gate %lld (%.1f%%)\n", measurements, rejected,
                    measurements ? 100.0 * rejected / measurements : 0.0);
        if (d_count > 0) {
            std::printf("jitter (RMS frame-to-frame change): offset %.4f m, curvature %.6f 1/m\n",
                        std::sqrt(d_offset_sq / d_count), std::sqrt(d_curv_sq / d_count));
        }
        std::printf("speed measured on %.1f%% of frames\n", 100.0 * speed_frames / frames);
        std::printf("processing ms: median %.2f  p95 %.2f  max %.2f\n", pct(0.5), pct(0.95), pct(1.0));
    }
};

int run_vision(const Options& o) {
    FrameSource source;
    std::string error;
    if (!source.open(o.source, error)) {
        std::cerr << "Atlas Vision: " << error << "\n";
        return 1;
    }
    std::cout << "source: " << source.describe() << "\n";

    Frame frame;
    if (!source.next(frame)) {
        std::cerr << "Atlas Vision: source produced no frames\n";
        return 1;
    }
    CameraModel camera;
    if (o.camera_model_path.empty()) {
        camera = default_camera_model(frame.image.size(), o.fov_deg);
    } else if (!load_camera_model(o.camera_model_path, camera, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    if (camera.image_size != frame.image.size()) {
        std::cerr << "camera model is " << camera.image_size.width << "x" << camera.image_size.height
                  << " but frames are " << frame.image.cols << "x" << frame.image.rows << "\n";
        return 1;
    }
    std::cout << "camera model: " << camera.source << "\n";
    print_model(camera);

    VisionPipeline pipeline;
    pipeline.configure(camera, GroundGrid());
    if (o.objects) {
        if (!pipeline.enable_objects(o.object_params, error)) {
            std::cerr << "objects: " << error << "\n"
                      << "Download https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
                         "yolox_tiny.onnx to data/models/, pass --object-model FILE, or run with --no-objects.\n";
            return 1;
        }
        std::cout << "objects: " << o.object_params.model_path << "\n";
    } else {
        std::cout << "objects: off (--no-objects)\n";
    }
    Dashboard dashboard;
    DataLogger logger;
    if (o.log) {
        std::string name = o.source.live ? "webcam" + std::to_string(o.source.camera_index) : o.source.path;
        if (!logger.start(o.logger, name, camera, pipeline.ground().grid(), error)) {
            std::cerr << "logger: " << error << "\n";
            return 1;
        }
        std::cout << "logging to " << logger.session_dir() << "\n";
    }

    using clock = std::chrono::steady_clock;
    auto t_start = clock::now();
    auto t_report = t_start;
    auto t_fps = t_start;
    long long since_report = 0;
    double loop_fps = 0.0;
    RunStats stats;
    DashboardStatus ds;
    ds.source = source.describe();
    ds.camera = camera.source;
    ds.show_flow = o.show_flow;
    PerceptionFrame pf;
    bool have_frame = true;
    bool paused = false;
    bool step_once = false;
    bool window_seen = false;

    while (have_frame) {
        if (!paused || step_once) {
            pipeline.process(frame, pf);
            logger.consider(pf);
            dashboard.observe(pf);
            stats.add(pf);
            for (const TrackEvent& e : pf.events) {
                std::printf("  [frame %lld t=%.2fs] %s\n", pf.frame_id, e.time_s, e.what.c_str());
            }
            ++since_report;
            step_once = false;
        }

        double fps_dt = std::chrono::duration<double>(clock::now() - t_fps).count();
        t_fps = clock::now();
        if (fps_dt > 0 && !paused) {
            loop_fps = loop_fps == 0.0 ? 1.0 / fps_dt : 0.9 * loop_fps + 0.1 / fps_dt;
        }

        bool want_snapshot = !o.snapshot_dir.empty() && pf.frame_id % o.snapshot_every == 0 && !paused;
        if (!o.headless || want_snapshot) {
            ds.source_stats = source.stats();
            ds.loop_fps = loop_fps;
            ds.memory_mib = process_memory_mib();
            ds.logger = logger.summary();
            ds.paused = paused;
            const cv::Mat& view = dashboard.render(pf, pipeline, ds);
            if (want_snapshot) {
                cv::imwrite(o.snapshot_dir + "/dash_" + std::to_string(pf.frame_id) + ".jpg", view);
            }
            if (!o.headless) {
                cv::imshow("Atlas Vision", view);
                int key = cv::waitKey(paused ? 30 : 1);
                if (key == 'q' || key == 27) break;
                if (key == ' ') paused = !paused;
                if (key == 'n') step_once = true;
                if (key == 'f') ds.show_flow = !ds.show_flow;
                if (key == 'r') ds.raw_only = !ds.raw_only;
                if (key == 'g') logger.label_next("good");
                if (key == 'b') logger.label_next("bad");
                // Backends that do not implement WND_PROP_VISIBLE return -1, so
                // only treat "not visible" as closed once it has been visible.
                double visible = cv::getWindowProperty("Atlas Vision", cv::WND_PROP_VISIBLE);
                if (visible >= 1) {
                    window_seen = true;
                } else if (window_seen) {
                    break;
                }
            }
        }

        double since = std::chrono::duration<double>(clock::now() - t_report).count();
        if (since >= 10.0) {
            FrameSourceStats s = source.stats();
            double elapsed = std::chrono::duration<double>(clock::now() - t_start).count();
            std::printf("[t=%6.0fs] frames=%lld  %.1f fps  proc %.1f ms  %s conf=%.2f  dropped=%lld  "
                        "read_fail=%lld  reconnects=%lld  loops=%lld  mem=%.1f MiB\n",
                        elapsed, stats.frames, since_report / since, pf.timings.total_ms,
                        status_name(pf.status), pf.confidence, s.dropped, s.read_failures, s.reconnects,
                        s.loops, process_memory_mib());
            std::fflush(stdout);
            t_report = clock::now();
            since_report = 0;
        }
        if (o.max_frames > 0 && stats.frames >= o.max_frames) break;
        if (!paused || step_once) {
            have_frame = source.next(frame);
        }
    }

    double wall = std::chrono::duration<double>(clock::now() - t_start).count();
    source.close();
    logger.stop();
    if (o.log) std::cout << logger.session_dir() << " written\n";
    FrameSourceStats s = source.stats();
    std::printf("source: dropped=%lld read_fail=%lld reconnects=%lld\n", s.dropped, s.read_failures,
                s.reconnects);
    stats.print(wall);
    cv::destroyAllWindows();
    return 0;
}

}  // namespace

int atlas_vision_main(int argc, char** argv) {
    Options o;
    if (!parse(argc, argv, o)) {
        usage();
        return 1;
    }
    if (o.mode.empty()) {
        return -1;
    }
    if (o.mode == "help") {
        usage();
        return 0;
    }
    if (o.mode == "calibrate") {
        return run_calibrate(o);
    }
    if (o.mode == "estimate-mount") {
        return run_estimate_mount(o);
    }
    if (o.mode == "review") {
        return review_session(o.review_dir, o.snapshot_dir);
    }
    return run_vision(o);
}
