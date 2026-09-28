#include "data_logger.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

const int kSchemaVersion = 1;

// Minimal JSON builder: enough for numbers, strings, arrays and objects.
class Json {
public:
    Json& begin_obj() { sep(); out_ << '{'; first_ = true; return *this; }
    Json& end_obj() { out_ << '}'; first_ = false; return *this; }
    Json& begin_arr() { sep(); out_ << '['; first_ = true; return *this; }
    Json& end_arr() { out_ << ']'; first_ = false; return *this; }
    Json& key(const char* k) { sep(); out_ << '"' << k << "\":"; first_ = true; return *this; }
    Json& num(double v) {
        sep();
        if (std::isfinite(v)) {
            char b[32];
            std::snprintf(b, sizeof(b), "%.6g", v);
            out_ << b;
        } else {
            out_ << "null";
        }
        return *this;
    }
    Json& integer(long long v) { sep(); out_ << v; return *this; }
    Json& boolean(bool v) { sep(); out_ << (v ? "true" : "false"); return *this; }
    Json& null() { sep(); out_ << "null"; return *this; }
    Json& str(const std::string& s) {
        sep();
        out_ << '"';
        for (char c : s) {
            if (c == '"' || c == '\\') out_ << '\\' << c;
            else if (c == '\n') out_ << "\\n";
            else out_ << c;
        }
        out_ << '"';
        return *this;
    }
    Json& kv(const char* k, double v) { return key(k).num(v); }
    Json& kv(const char* k, const std::string& v) { return key(k).str(v); }
    Json& kvb(const char* k, bool v) { return key(k).boolean(v); }
    Json& kvi(const char* k, long long v) { return key(k).integer(v); }
    template <typename M>
    Json& matrix(const char* k, const M& m, int rows, int cols) {
        key(k).begin_arr();
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) num(m(r, c));
        }
        return end_arr();
    }
    std::string str_value() const { return out_.str(); }

private:
    void sep() {
        if (!first_) out_ << ',';
        first_ = false;
    }
    std::ostringstream out_;
    bool first_ = true;
};

std::uint64_t dir_size(const fs::path& p) {
    std::uint64_t total = 0;
    for (const auto& e : fs::recursive_directory_iterator(p)) {
        if (e.is_regular_file()) total += e.file_size();
    }
    return total;
}

std::string timestamp_name() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char b[32];
    std::strftime(b, sizeof(b), "%Y%m%d_%H%M%S", &tm);
    return b;
}

}  // namespace

DataLogger::~DataLogger() {
    stop();
}

void DataLogger::enforce_total_budget() {
    fs::path root(cfg_.root);
    if (!fs::exists(root)) return;
    std::vector<fs::path> sessions;
    for (const auto& e : fs::directory_iterator(root)) {
        if (e.is_directory()) sessions.push_back(e.path());
    }
    // Session folder names start with a timestamp, so name order is age order.
    std::sort(sessions.begin(), sessions.end());
    std::uint64_t total = 0;
    std::vector<std::uint64_t> sizes;
    for (const fs::path& s : sessions) {
        sizes.push_back(dir_size(s));
        total += sizes.back();
    }
    // Leave room for one full new session.
    double limit = (cfg_.max_total_mb - cfg_.max_session_mb) * 1024.0 * 1024.0;
    for (std::size_t i = 0; i < sessions.size() && total > limit; ++i) {
        std::cout << "[logger] storage budget: deleting oldest session " << sessions[i].string() << " ("
                  << sizes[i] / (1024 * 1024) << " MiB)\n";
        fs::remove_all(sessions[i]);
        total -= sizes[i];
    }
}

bool DataLogger::start(const DataLoggerConfig& cfg, const std::string& source_name,
                       const CameraModel& camera, const GroundGrid& grid, std::string& error) {
    cfg_ = cfg;
    camera_ = camera;
    grid_ = grid;
    std::error_code ec;
    fs::create_directories(cfg.root, ec);
    if (ec) {
        error = "cannot create " + cfg.root + ": " + ec.message();
        return false;
    }
    enforce_total_budget();
    std::string stem = fs::path(source_name).stem().string();
    dir_ = (fs::path(cfg.root) / (timestamp_name() + "_" + stem)).string();
    fs::create_directories(fs::path(dir_) / "images", ec);
    if (ec) {
        error = "cannot create " + dir_ + ": " + ec.message();
        return false;
    }

    Json j;
    j.begin_obj();
    j.kvi("schema_version", kSchemaVersion);
    j.kv("source", source_name);
    j.kv("image_space", std::string("undistorted"));
    j.key("camera").begin_obj();
    j.kvi("width", camera.image_size.width).kvi("height", camera.image_size.height);
    j.matrix("K", camera.K, 3, 3);
    j.key("dist").begin_arr();
    for (int i = 0; i < static_cast<int>(camera.dist.total()); ++i) j.num(camera.dist.at<double>(i));
    j.end_arr();
    j.kv("height_m", camera.height_m).kv("pitch_rad", camera.pitch_rad).kv("yaw_rad", camera.yaw_rad);
    j.kvb("intrinsics_calibrated", camera.intrinsics_calibrated);
    j.kvb("mount_calibrated", camera.mount_calibrated);
    j.matrix("H_ground_to_image", camera.H_ground_to_image, 3, 3);
    j.end_obj();
    j.key("ground_grid").begin_obj();
    j.kv("x_min", grid.x_min).kv("x_max", grid.x_max).kv("y_half", grid.y_half).kv("resolution_m", grid.resolution_m);
    j.end_obj();
    j.key("conventions").begin_obj();
    j.kv("vehicle_frame", std::string("X forward, Y left, metres, origin on ground below camera"));
    j.kv("lane_polynomial", std::string("Y = c0 + c1*X + c2*X^2"));
    j.kv("road_state", std::string("[y_centre, c1, c2, width]; boundary k at y_centre + k*width"));
    j.kv("lanes_image", std::string("TuSimple style: x pixel at each h_sample row, -2 = no lane"));
    j.end_obj();
    j.end_obj();
    std::ofstream(fs::path(dir_) / "session.json") << j.str_value() << "\n";

    stop_ = false;
    running_ = true;
    thread_ = std::thread(&DataLogger::worker, this);
    return true;
}

void DataLogger::worker() {
    std::ofstream out(fs::path(dir_) / "frames.jsonl", std::ios::app);
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, cfg_.jpeg_quality};
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        std::vector<uchar> buf;
        cv::imencode(".jpg", job.image, buf, params);
        std::ofstream(fs::path(dir_) / job.image_path, std::ios::binary)
            .write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        out << job.json << "\n";
        out.flush();
        std::lock_guard<std::mutex> lock(mutex_);
        bytes_ += buf.size() + job.json.size() + 1;
        ++records_;
        if (bytes_ > cfg_.max_session_mb * 1024.0 * 1024.0) {
            budget_hit_ = true;
        }
    }
}

void DataLogger::label_next(const std::string& label) {
    pending_label_ = label;
}

void DataLogger::consider(const PerceptionFrame& f) {
    if (!running_) return;
    std::vector<std::string> reasons;
    double t = f.media_time_s;
    if (!pending_label_.empty()) reasons.push_back("human_label");
    if (t - last_keyframe_s_ >= cfg_.keyframe_interval_s) reasons.push_back("keyframe");
    if (f.status != last_status_) reasons.push_back("status_change");
    bool rate_ok = t - last_event_s_ >= cfg_.event_min_interval_s;
    if (rate_ok && !f.events.empty()) reasons.push_back("tracker_event");
    if (rate_ok && f.road.valid && f.confidence < cfg_.low_confidence) reasons.push_back("low_confidence");
    bool rejected = std::any_of(f.detection.lanes.begin(), f.detection.lanes.end(),
                                [](const LaneMeasurement& m) { return !m.accepted; });
    if (rate_ok && rejected) reasons.push_back("rejected_measurement");
    last_status_ = f.status;
    if (reasons.empty()) return;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (budget_hit_) return;
        if (queue_.size() >= 8) {
            ++dropped_;
            return;
        }
    }
    if (std::find(reasons.begin(), reasons.end(), "keyframe") != reasons.end()) last_keyframe_s_ = t;
    if (reasons.size() > 1 || reasons[0] != "keyframe") last_event_s_ = t;

    char name[48];
    std::snprintf(name, sizeof(name), "images/%08lld.jpg", f.frame_id);
    Job job{name, f.undistorted.clone(), record_json(f, name, reasons)};
    pending_label_.clear();
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(job));
    cv_.notify_one();
}

std::string DataLogger::record_json(const PerceptionFrame& f, const std::string& image_rel,
                                    const std::vector<std::string>& reasons) const {
    Json j;
    j.begin_obj();
    j.kvi("schema_version", kSchemaVersion);
    j.kvi("frame_id", f.frame_id);
    j.kv("media_time_s", f.media_time_s).kv("capture_time_s", f.capture_time_s);
    j.kv("image", image_rel);
    j.key("reasons").begin_arr();
    for (const std::string& r : reasons) j.str(r);
    j.end_arr();
    j.key("labels").begin_obj();
    if (pending_label_.empty()) j.key("human").null();
    else j.kv("human", pending_label_);
    j.end_obj();

    j.kv("status", std::string(status_name(f.status))).kv("confidence", f.confidence);
    j.key("road_state").begin_obj().kvb("valid", f.road.valid);
    if (f.road.valid) {
        j.key("x").begin_arr();
        for (int i = 0; i < 4; ++i) j.num(f.road.x[i]);
        j.end_arr();
        j.matrix("P", f.road.P, 4, 4);
    }
    j.end_obj();
    const RoadGeometry& g = f.geometry;
    j.key("geometry").begin_obj().kvb("valid", g.valid);
    if (g.valid) {
        j.kv("lateral_offset_m", g.lateral_offset_m).kv("lateral_offset_sigma_m", g.lateral_offset_sigma_m);
        j.kv("heading_error_rad", g.heading_error_rad);
        j.kv("curvature_1pm", g.curvature_1pm).kv("curvature_sigma_1pm", g.curvature_sigma_1pm);
        j.kv("lane_width_m", g.lane_width_m).kv("lane_width_sigma_m", g.lane_width_sigma_m);
        j.key("lane_vanishing_px").begin_arr().num(g.lane_vanishing_px.x).num(g.lane_vanishing_px.y).end_arr();
    }
    j.end_obj();
    j.key("ego_motion").begin_obj();
    j.kvb("valid", f.ego.valid).kv("forward_m", f.ego.forward_m).kv("lateral_m", f.ego.lateral_m);
    j.kv("yaw_rad", f.ego.yaw_rad).kvi("tracked", f.ego.tracked).kvi("inliers", f.ego.inliers);
    j.kv("speed_mps", f.speed_mps).kvb("speed_measured", f.speed_measured);
    j.end_obj();

    j.key("measurements").begin_arr();
    for (const LaneMeasurement& m : f.detection.lanes) {
        j.begin_obj();
        j.kv("slot", std::string(slot_name(m.slot)));
        j.key("coeffs").begin_arr().num(m.coeffs[0]).num(m.coeffs[1]).num(m.coeffs[2]).end_arr();
        j.matrix("cov", m.cov, 3, 3);
        j.kv("x_near", m.x_near).kv("x_far", m.x_far).kv("rms_m", m.rms_m).kv("coverage", m.coverage);
        j.kvb("dashed", m.dashed).kvb("yellow", m.yellow).kvb("guided", m.guided);
        j.kvb("accepted", m.accepted).kv("mahalanobis2", m.mahalanobis2);
        // Evidence points every ~0.5 m, in image pixels.
        j.key("points_px").begin_arr();
        for (std::size_t i = 0; i < m.points.size(); i += 10) {
            cv::Point2d px = camera_.ground_to_image(m.points[i].x, m.points[i].y);
            j.begin_arr().num(std::round(px.x * 10) / 10).num(std::round(px.y * 10) / 10).end_arr();
        }
        j.end_arr();
        j.end_obj();
    }
    j.end_arr();

    j.kvb("objects_enabled", f.objects_enabled);
    j.key("objects").begin_arr();
    for (const TrackedObject& t : f.objects) {
        j.begin_obj();
        j.kvi("track_id", t.id).kv("class", std::string(coco_name(t.class_id))).kvi("class_id", t.class_id);
        j.kv("score", t.score).kvb("confirmed", t.confirmed).kvb("detected_this_frame", t.matched);
        j.key("box_xywh").begin_arr().num(t.box.x).num(t.box.y).num(t.box.width).num(t.box.height).end_arr();
        j.kvb("ground_valid", t.ground_valid);
        if (t.ground_valid) {
            j.key("ground_xy_m").begin_arr().num(t.g[0]).num(t.g[1]).end_arr();
            j.key("rel_velocity_mps").begin_arr().num(t.g[2]).num(t.g[3]).end_arr();
            j.key("sigma_xy_m").begin_arr().num(std::sqrt(t.P(0, 0))).num(std::sqrt(t.P(1, 1))).end_arr();
        }
        j.kvb("in_ego_lane", t.in_ego_lane).kv("ttc_s", t.ttc_s);
        j.end_obj();
    }
    j.end_arr();

    // Tracked lanes as TuSimple-style samples: pseudo-labels for training.
    double horizon = camera_.horizon_row_at(camera_.image_size.width / 2.0);
    int row0 = static_cast<int>(std::ceil((horizon + 20.0) / 10.0) * 10.0);
    j.key("h_samples").begin_arr();
    for (int v = row0; v < camera_.image_size.height; v += 10) j.integer(v);
    j.end_arr();
    j.key("lanes_image").begin_arr();
    if (f.road.valid) {
        for (int s = 0; s < kLaneSlots; ++s) {
            double k = slot_width_multiple(static_cast<LaneSlot>(s));
            j.begin_obj().kv("slot", std::string(slot_name(static_cast<LaneSlot>(s))));
            j.key("x").begin_arr();
            for (int v = row0; v < camera_.image_size.height; v += 10) {
                // Intersect the image row with the lane: find ground X on the
                // row centre, then iterate once using the lane's lateral offset.
                double X, Y;
                double u = camera_.image_size.width / 2.0;
                bool ok = camera_.image_to_ground({u, static_cast<double>(v)}, X, Y);
                for (int it = 0; ok && it < 3; ++it) {
                    cv::Point2d px = camera_.ground_to_image(X, f.road.lateral(X, k));
                    ok = camera_.image_to_ground({px.x, static_cast<double>(v)}, X, Y);
                }
                bool drawable = ok && X >= grid_.x_min && X <= grid_.x_max &&
                                f.road.lateral_sigma(X, k) < 0.4;
                if (!drawable) {
                    j.integer(-2);
                    continue;
                }
                cv::Point2d px = camera_.ground_to_image(X, f.road.lateral(X, k));
                bool inside = px.x >= 0 && px.x < camera_.image_size.width;
                j.integer(inside ? static_cast<long long>(std::lround(px.x)) : -2);
            }
            j.end_arr();
            j.key("sigma_m_at_15m").num(f.road.lateral_sigma(15.0, k));
            j.end_obj();
        }
    }
    j.end_arr();

    j.key("path").begin_arr();
    for (const cv::Point2d& p : f.path.points) j.begin_arr().num(p.x).num(p.y).end_arr();
    j.end_arr();
    j.kv("steering_rad", f.path.steering_rad);
    j.key("events").begin_arr();
    for (const TrackEvent& e : f.events) j.str(e.what);
    j.end_arr();
    j.kv("processing_ms", f.timings.total_ms);
    j.end_obj();
    return j.str_value();
}

void DataLogger::stop() {
    if (!running_) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        cv_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
    running_ = false;
}

std::string DataLogger::summary() const {
    if (!running_) return "logging off (--log to record training data)";
    std::lock_guard<std::mutex> lock(mutex_);
    char b[256];
    std::snprintf(b, sizeof(b), "log %s: %lld records, %.1f / %.0f MiB, dropped %lld%s", dir_.c_str(), records_,
                  bytes_ / (1024.0 * 1024.0), cfg_.max_session_mb, dropped_,
                  budget_hit_ ? "  SESSION BUDGET REACHED" : "");
    return b;
}
