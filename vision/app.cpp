#include "app.hpp"
#include "calibration.hpp"
#include "camera_model.hpp"
#include "dashboard.hpp"
#include "data_logger.hpp"
#include "frame_source.hpp"
#include "pipeline.hpp"
#include "process_stats.hpp"
#include "review.hpp"
#include "video_prep.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
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
    std::string calib_video;
    std::string batch_dir;
    std::string review_dir;
    ChessboardOptions chessboard;
    int process_width = 0;
    bool headless = false;
    long long max_frames = -1;
    double lane_width_m = 3.7;
    double fov_deg = 70.0;
    std::string snapshot_dir;
    long long snapshot_every = 100;
    bool show_flow = false;
    bool objects = true;
    bool object_model_explicit = false;
    ObjectDetectorParams object_params;
    bool log = false;
    DataLoggerConfig logger;
};

void usage() {
    std::cout <<
        "Atlas Vision\n"
        "  atlas --webcam [--camera-index N] [--no-objects]   live camera\n"
        "  atlas --video FILE [--loop]                  recorded video\n"
        "     --camera-model FILE.yml   calibration (default: --fov 70 deg, level, 1.4 m high)\n"
        "     --headless                no window; prints telemetry and a run summary\n"
        "     --max-frames N            stop after N frames (default: run until the source ends)\n"
        "     --snapshots DIR [--snapshot-every N]   save dashboard images\n"
        "     --object-model FILE.onnx | --no-objects   (default data/models/yolox_tiny.onnx)\n"
        "     --flow                    draw optical-flow vectors\n"
        "     --log [--log-dir DIR] [--log-budget-mb MB]   record training data\n"
        "     --process-width W         resize frames to width W right after capture (default: native;\n"
        "                               1280 recommended for 1080p/4K phone video)\n"
        "  atlas --batch-log DIR --camera-model FILE.yml   headless + --log on every .mov/.mp4 in DIR\n"
        "     [--process-width W] [--log-dir DIR] [--log-budget-mb MB] [--no-objects] [--max-frames N]\n"
        "  atlas --review SESSION_DIR [--snapshots DIR] browse / label logged training data\n"
        "  atlas --calibrate DIR --out FILE.yml         chessboard intrinsics from .jpg/.jpeg/.png photos\n"
        "  atlas --calibrate-video FILE --out FILE.yml  chessboard intrinsics from a video\n"
        "     [--board 9x6] (inner corners) [--square-mm 25] [--process-width W]\n"
        "  atlas --estimate-mount FILE [--camera-model IN.yml | --fov DEG] --out OUT.yml [--lane-width 3.7]\n"
        "     [--process-width W]\n";
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
            o.object_model_explicit = true;
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
        } else if (a == "--calibrate-video") {
            o.mode = "calibrate-video";
            if (!value(o.calib_video)) return false;
        } else if (a == "--board") {
            if (!value(v)) return false;
            int c = 0, r = 0;
            if (std::sscanf(v.c_str(), "%dx%d", &c, &r) != 2 || c < 2 || r < 2) {
                std::cerr << "--board wants inner corners as COLSxROWS, e.g. 9x6\n";
                return false;
            }
            o.chessboard.inner_cols = c;
            o.chessboard.inner_rows = r;
        } else if (a == "--square-mm") {
            if (!value(v)) return false;
            o.chessboard.square_mm = std::stod(v);
        } else if (a == "--process-width") {
            if (!value(v)) return false;
            o.process_width = std::stoi(v);
            if (o.process_width < 0) {
                std::cerr << "--process-width must be >= 0\n";
                return false;
            }
        } else if (a == "--batch-log") {
            o.mode = "batch-log";
            if (!value(o.batch_dir)) return false;
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

const char* kPortraitHint = "record in landscape (phone on its side) and keep the same orientation for "
                             "calibration and drives";

// Makes the camera model match the processed frame size: rescales it when
// only the resolution differs, fails when the aspect ratio differs.
bool fit_camera_to_frames(CameraModel& camera, cv::Size frames, std::string& error) {
    if (camera.image_size == frames) {
        return true;
    }
    auto dims = [](cv::Size s) { return std::to_string(s.width) + "x" + std::to_string(s.height); };
    if (!same_aspect_ratio(camera.image_size, frames)) {
        error = "camera model is " + dims(camera.image_size) + " but frames are " + dims(frames) +
                " (different aspect ratio: recalibrate at this resolution/orientation)";
        return false;
    }
    std::cout << "notice: camera model is " << dims(camera.image_size) << ", frames are " << dims(frames)
              << "; scaling the intrinsics to match\n";
    camera = scaled_camera_model(camera, frames);
    return true;
}

int run_calibrate(const Options& o) {
    if (o.out_path.empty()) {
        std::cerr << "--calibrate / --calibrate-video need --out FILE.yml\n";
        return 1;
    }
    ChessboardOptions opts = o.chessboard;
    opts.process_width = o.process_width;
    std::printf("board %dx%d inner corners, %.1f mm squares\n", opts.inner_cols, opts.inner_rows, opts.square_mm);
    CameraModel model;
    ChessboardResult r;
    std::string error;
    bool ok = o.mode == "calibrate-video"
                  ? calibrate_from_chessboard_video(o.calib_video, opts, model, r, error)
                  : calibrate_from_chessboards(o.calib_dir, opts, model, r, error);
    if (!ok) {
        std::cerr << "calibration failed: " << error << "\n";
        return 1;
    }
    std::printf("per-view reprojection error (px):\n");
    for (std::size_t i = 0; i < r.per_view_error_px.size(); ++i) {
        std::printf("  %-24s %.3f\n", r.view_names[i].c_str(), r.per_view_error_px[i]);
    }
    std::printf("used %d views (%d with a board, %d total), %d outliers rejected\n", r.images_used,
                r.boards_found, r.images_total, r.outliers_rejected);
    std::printf("RMS reprojection error %.3f px (before outlier rejection %.3f px)\n", r.rms_reprojection_px,
                r.rms_initial_px);
    if (r.rms_reprojection_px > 1.0) {
        std::printf("warning: RMS > 1 px; check the board is flat, sharp and the --board size is right\n");
    }
    if (r.images_used < 15) {
        std::printf("warning: only %d views; 15-40 views covering the frame edges give stable intrinsics\n",
                    r.images_used);
    }
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
    cv::Size frames;
    {
        cv::VideoCapture cap;
        cv::Mat first;
        if (!open_video_file(cap, o.source.path, error) || !cap.read(first)) {
            std::cerr << (error.empty() ? "cannot read " + o.source.path : error) << "\n";
            return 1;
        }
        frames = processing_size(first.size(), o.process_width);
        std::string orientation = describe_orientation(cap);
        std::printf("video %dx%d%s%s, processing at %dx%d\n", first.cols, first.rows,
                    orientation.empty() ? "" : ", ", orientation.c_str(), frames.width, frames.height);
        if (first.rows > first.cols) {
            std::cerr << "video is portrait; " << kPortraitHint << "\n";
            return 1;
        }
    }
    if (o.camera_model_path.empty()) {
        model = default_camera_model(frames, o.fov_deg);
        std::cout << "no --camera-model: assuming " << o.fov_deg
                  << " deg horizontal FOV and no lens distortion (distances will be approximate)\n";
    } else if (!load_camera_model(o.camera_model_path, model, error) ||
               !fit_camera_to_frames(model, frames, error)) {
        std::cerr << error << "\n";
        return 1;
    }
    MountEstimate e;
    int max_frames = o.max_frames > 0 ? static_cast<int>(o.max_frames) : 300;
    if (!estimate_mount_from_video(o.source.path, o.lane_width_m, max_frames, model, e, error)) {
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

struct RunResult {
    int code = 1;
    std::string error;
    RunStats stats;
    double wall_s = 0.0;
    std::string session_dir;
};

// One video / camera run with the given options: used by --video, --webcam
// and once per file by --batch-log.
RunResult run_vision(const Options& o) {
    RunResult result;
    auto fail = [&](const std::string& message) {
        std::cerr << "Atlas Vision: " << message << "\n";
        result.error = message;
        result.code = 1;
        return result;
    };
    FrameSource source;
    std::string error;
    if (!source.open(o.source, error)) {
        return fail(error);
    }
    std::cout << "source: " << source.describe() << "\n";

    Frame frame;
    if (!source.next(frame)) {
        return fail("source produced no frames");
    }
    {
        FrameSourceStats s = source.stats();
        std::printf("frames: native %dx%d, processing %dx%d\n", s.native_size.width, s.native_size.height,
                    frame.image.cols, frame.image.rows);
    }
    if (frame.image.rows > frame.image.cols) {
        return fail("source is portrait (" + std::to_string(frame.image.cols) + "x" +
                    std::to_string(frame.image.rows) + "); " + kPortraitHint);
    }
    CameraModel camera;
    if (o.camera_model_path.empty()) {
        camera = default_camera_model(frame.image.size(), o.fov_deg);
    } else if (!load_camera_model(o.camera_model_path, camera, error) ||
               !fit_camera_to_frames(camera, frame.image.size(), error)) {
        return fail(error);
    }
    std::cout << "camera model: " << camera.source << "\n";
    print_model(camera);

    VisionPipeline pipeline;
    pipeline.configure(camera, GroundGrid());
    if (o.objects) {
        if (!pipeline.enable_objects(o.object_params, error)) {
            bool missing_default = !o.object_model_explicit &&
                                   error.find("object model not found") != std::string::npos;
            if (missing_default) {
                std::cerr << "warning: " << error
                          << " — continuing with objects off. Download "
                             "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
                             "yolox_tiny.onnx to data/models/, or pass --object-model FILE.\n";
                std::cout << "objects: off (default model missing)\n";
            } else {
                std::cerr << "Download https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
                             "yolox_tiny.onnx to data/models/, pass --object-model FILE, or run with --no-objects.\n";
                return fail("objects: " + error);
            }
        } else {
            std::cout << "objects: " << o.object_params.model_path << "\n";
        }
    } else {
        std::cout << "objects: off (--no-objects)\n";
    }
    Dashboard dashboard;
    DataLogger logger;
    if (o.log) {
        std::string name = o.source.live ? "webcam" + std::to_string(o.source.camera_index) : o.source.path;
        if (!logger.start(o.logger, name, camera, pipeline.ground().grid(), error)) {
            return fail("logger: " + error);
        }
        result.session_dir = logger.session_dir();
        std::cout << "logging to " << logger.session_dir() << "\n";
    }

    using clock = std::chrono::steady_clock;
    auto t_start = clock::now();
    auto t_report = t_start;
    auto t_fps = t_start;
    long long since_report = 0;
    double loop_fps = 0.0;
    RunStats& stats = result.stats;
    DashboardStatus ds;
    ds.source = source.describe();
    ds.camera = camera.source;
    ds.show_flow = o.show_flow;
    PerceptionFrame pf;
    bool have_frame = true;
    bool paused = false;
    bool step_once = false;
    bool window_created = false;
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
                if (!window_created) {
                    cv::namedWindow("Atlas Vision", cv::WINDOW_NORMAL);
                    window_created = true;
                }
                cv::imshow("Atlas Vision", view);
                int key = cv::waitKey(paused ? 30 : 1);
                if (key == 'q' || key == 27) break;
                if (key == ' ') paused = !paused;
                if (key == 'n') step_once = true;
                if (key == 'f') ds.show_flow = !ds.show_flow;
                if (key == 'r') ds.raw_only = !ds.raw_only;
                if (key == 'g') logger.label_next("good");
                if (key == 'b') logger.label_next("bad");
                // Backends that do not implement WND_PROP_VISIBLE return -1.
                // Only quit after the window has been visible (>= 1) and later
                // reports 0 (user closed it). Do not treat -1 as closed.
                double visible = cv::getWindowProperty("Atlas Vision", cv::WND_PROP_VISIBLE);
                if (visible >= 1.0) {
                    window_seen = true;
                } else if (window_seen && visible == 0.0) {
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
    if (!o.headless) {
        cv::destroyAllWindows();
    }
    result.wall_s = wall;
    result.code = 0;
    return result;
}

int run_batch(const Options& o) {
    if (o.camera_model_path.empty()) {
        std::cerr << "--batch-log needs --camera-model FILE.yml (calibrate first: see docs/IPHONE_RECORDING.md)\n";
        return 1;
    }
    std::vector<std::string> videos = list_files(o.batch_dir, is_video_file_name);
    if (videos.empty()) {
        std::cerr << "no .mov/.mp4/.m4v files in " << o.batch_dir << "\n";
        return 1;
    }
    std::printf("batch: %zu videos, logging to %s, storage budget %.0f MB total / %.0f MB per session\n",
                videos.size(), o.logger.root.c_str(), o.logger.max_total_mb, o.logger.max_session_mb);
    std::printf("note: when the total budget is exceeded the OLDEST sessions in %s are deleted, including "
                "earlier ones from this batch; raise it with --log-budget-mb if needed\n",
                o.logger.root.c_str());
    if (!o.snapshot_dir.empty()) {
        std::printf("note: --snapshots is ignored in batch mode\n");
    }

    std::vector<RunResult> results;
    for (std::size_t i = 0; i < videos.size(); ++i) {
        std::printf("\n==== [%zu/%zu] %s ====\n", i + 1, videos.size(), videos[i].c_str());
        std::fflush(stdout);
        Options v = o;
        v.mode = "run";
        v.source.live = false;
        v.source.loop = false;
        v.source.path = videos[i];
        v.headless = true;
        v.log = true;
        v.snapshot_dir.clear();
        results.push_back(run_vision(v));
    }

    int failed = 0;
    std::printf("\n==== batch summary ====\n");
    for (std::size_t i = 0; i < videos.size(); ++i) {
        const RunResult& r = results[i];
        std::string name = std::filesystem::path(videos[i]).filename().string();
        if (r.code != 0) {
            ++failed;
            std::printf("%-28s FAILED: %s\n", name.c_str(), r.error.c_str());
            continue;
        }
        const RunStats& s = r.stats;
        double n = s.frames > 0 ? static_cast<double>(s.frames) : 1.0;
        std::printf("%-28s frames %6lld  %6.1f s  LOCKED %5.1f%%  conf %.2f  resets %lld  -> %s\n", name.c_str(),
                    s.frames, r.wall_s, 100.0 * s.per_status[static_cast<int>(TrackStatus::Locked)] / n,
                    s.conf_sum / n, s.resets, r.session_dir.c_str());
    }
    std::printf("%zu ok, %d failed\n", videos.size() - failed, failed);
    return failed == 0 ? 0 : 1;
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
    o.source.process_width = o.process_width;
    if (o.mode == "calibrate" || o.mode == "calibrate-video") {
        return run_calibrate(o);
    }
    if (o.mode == "estimate-mount") {
        return run_estimate_mount(o);
    }
    if (o.mode == "review") {
        return review_session(o.review_dir, o.snapshot_dir);
    }
    if (o.mode == "batch-log") {
        return run_batch(o);
    }
    return run_vision(o).code;
}
