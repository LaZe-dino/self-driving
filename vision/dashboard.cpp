#include "dashboard.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

const int kWidth = 1600;
const int kHeight = 900;
const cv::Rect kCameraRect(0, 0, 1120, 630);
const cv::Rect kBevRect(1120, 0, 240, 630);
const cv::Rect kFeatureRect(1360, 0, 240, 630);
const cv::Rect kTextRect(0, 630, 560, 270);
const cv::Rect kChartRect(560, 630, 1040, 270);
const std::size_t kHistory = 375;

const cv::Scalar kWhite(235, 235, 235);
const cv::Scalar kGrey(140, 140, 140);
const cv::Scalar kDark(28, 28, 28);
const cv::Scalar kEgoLine(240, 210, 60);
const cv::Scalar kOuterLine(190, 130, 70);
const cv::Scalar kPath(40, 150, 255);
const cv::Scalar kAccepted(80, 230, 80);
const cv::Scalar kRejected(60, 60, 240);

cv::Scalar status_color(TrackStatus s) {
    switch (s) {
        case TrackStatus::Locked: return cv::Scalar(90, 220, 110);
        case TrackStatus::Partial: return cv::Scalar(40, 210, 240);
        case TrackStatus::Coasting: return cv::Scalar(30, 140, 255);
        case TrackStatus::Searching: return cv::Scalar(90, 90, 255);
    }
    return kWhite;
}

void text(cv::Mat& img, const std::string& s, cv::Point p, double scale, const cv::Scalar& c,
          int thick = 1) {
    cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, c, thick, cv::LINE_AA);
}

// Distance ahead to which a boundary is worth drawing: stop where the 1-sigma
// lateral uncertainty passes 0.4 m instead of drawing a confident-looking line.
double drawable_extent(const RoadState& road, double k, double x_min, double x_max) {
    double X = x_min;
    for (; X <= x_max; X += 1.0) {
        if (road.lateral_sigma(X, k) > 0.4) {
            break;
        }
    }
    return X;
}

void dashed(cv::Mat& img, const std::vector<cv::Point>& pts, const cv::Scalar& c, int thick,
            double dash, double gap) {
    bool on = true;
    double remaining = dash;
    for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
        cv::Point2d a = pts[i], b = pts[i + 1];
        double len = std::hypot(b.x - a.x, b.y - a.y);
        double pos = 0.0;
        while (pos < len) {
            double step = std::min(remaining, len - pos);
            if (on) {
                cv::line(img, a + (b - a) * (pos / len), a + (b - a) * ((pos + step) / len), c, thick,
                         cv::LINE_AA);
            }
            pos += step;
            remaining -= step;
            if (remaining <= 1e-9) {
                on = !on;
                remaining = on ? dash : gap;
            }
        }
    }
}

bool measured_slot(const PerceptionFrame& f, LaneSlot s, const LaneMeasurement** out) {
    for (const LaneMeasurement& m : f.detection.lanes) {
        if (m.accepted && m.slot == s) {
            *out = &m;
            return true;
        }
    }
    return false;
}

template <typename Proj>
std::vector<cv::Point> boundary(const RoadState& r, double k, double x0, double x1, double offset,
                                Proj proj) {
    std::vector<cv::Point> pts;
    for (double X = x0; X <= x1 + 1e-9; X += 0.5) {
        double Y = r.lateral(X, k) + offset * r.lateral_sigma(X, k);
        pts.push_back(proj(X, Y));
    }
    return pts;
}

}  // namespace

Dashboard::Dashboard() : canvas_(kHeight, kWidth, CV_8UC3) {}

void Dashboard::draw_camera(const PerceptionFrame& f, const VisionPipeline& p,
                            const DashboardStatus& s, cv::Mat& panel) {
    const CameraModel& cam = p.camera();
    const GroundGrid& grid = p.ground().grid();
    cv::Mat img;
    if (s.raw_only) {
        img = f.raw.clone();
        text(img, "RAW CAMERA (no processing)", {20, 40}, 0.9, kWhite, 2);
        cv::resize(img, panel, panel.size(), 0, 0, cv::INTER_AREA);
        return;
    }
    f.undistorted.convertTo(img, -1, 0.75, 0);
    auto proj = [&](double X, double Y) {
        cv::Point2d q = cam.ground_to_image(X, Y);
        return cv::Point(cvRound(q.x), cvRound(q.y));
    };

    // Horizon predicted by the calibrated mount.
    if (f.geometry.valid || cam.mount_calibrated) {
        cv::Point a(0, cvRound(cam.horizon_row_at(0)));
        cv::Point b(img.cols, cvRound(cam.horizon_row_at(img.cols)));
        dashed(img, {a, b}, kGrey, 1, 12, 10);
        char hb[96];
        std::snprintf(hb, sizeof(hb), "horizon: pitch %+.2f deg %s", cam.pitch_rad * 180.0 / CV_PI,
                      f.pitch_measured ? "(live from lane parallelism)" : "(held)");
        text(img, hb, {a.x + 10, a.y - 8}, 0.5, kGrey);
    }

    if (s.show_flow) {
        for (std::size_t i = 0; i < f.ego.from_px.size(); ++i) {
            bool in = i < f.ego.inlier_mask.size() && f.ego.inlier_mask[i];
            cv::line(img, f.ego.from_px[i], f.ego.to_px[i], in ? kAccepted : kRejected, 1, cv::LINE_AA);
            cv::circle(img, f.ego.to_px[i], 2, in ? kAccepted : kRejected, cv::FILLED);
        }
    }

    const RoadState& road = f.road;
    if (road.valid) {
        cv::Scalar sc = status_color(f.status);
        double x0 = grid.x_min;
        double xl = drawable_extent(road, 0.5, x0, grid.x_max);
        double xr = drawable_extent(road, -0.5, x0, grid.x_max);
        double xe = std::min(xl, xr);

        // Ego-lane area: opacity follows confidence.
        if (xe > x0 + 1.0) {
            std::vector<cv::Point> poly = boundary(road, 0.5, x0, xe, 0.0, proj);
            std::vector<cv::Point> right = boundary(road, -0.5, x0, xe, 0.0, proj);
            poly.insert(poly.end(), right.rbegin(), right.rend());
            cv::Mat overlay = img.clone();
            cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{poly}, sc);
            cv::addWeighted(overlay, 0.12 + 0.2 * f.confidence, img, 0.88 - 0.2 * f.confidence, 0, img);
        }

        for (int si = 0; si < kLaneSlots; ++si) {
            LaneSlot slot = static_cast<LaneSlot>(si);
            double k = slot_width_multiple(slot);
            double x1 = drawable_extent(road, k, x0, grid.x_max);
            if (x1 < x0 + 2.0) {
                continue;
            }
            const LaneMeasurement* m = nullptr;
            bool measured = measured_slot(f, slot, &m);
            bool inner = slot == LaneSlot::Left || slot == LaneSlot::Right;
            if (!inner && !measured) {
                continue;
            }
            cv::Scalar c = inner ? kEgoLine : kOuterLine;
            // +/- 2 sigma band.
            std::vector<cv::Point> band = boundary(road, k, x0, x1, 2.0, proj);
            std::vector<cv::Point> lo = boundary(road, k, x0, x1, -2.0, proj);
            band.insert(band.end(), lo.rbegin(), lo.rend());
            cv::Mat overlay = img.clone();
            cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{band}, c);
            cv::addWeighted(overlay, 0.25, img, 0.75, 0, img);
            std::vector<cv::Point> line = boundary(road, k, x0, x1, 0.0, proj);
            if (measured) {
                cv::polylines(img, line, false, c, inner ? 4 : 2, cv::LINE_AA);
            } else {
                // Not observed this frame: the line comes from the tracker's memory.
                dashed(img, line, c, 3, 18, 14);
            }
        }

        // Predicted path with chevrons that move at the measured speed.
        if (f.path.valid) {
            std::vector<cv::Point> left_edge, right_edge;
            const std::vector<cv::Point2d>& P = f.path.points;
            std::vector<double> arc(P.size(), 0.0);
            for (std::size_t i = 1; i < P.size(); ++i) {
                arc[i] = arc[i - 1] + std::hypot(P[i].x - P[i - 1].x, P[i].y - P[i - 1].y);
            }
            for (std::size_t i = 0; i < P.size(); ++i) {
                if (P[i].x < x0) continue;
                left_edge.push_back(proj(P[i].x, P[i].y + 0.9));
                right_edge.push_back(proj(P[i].x, P[i].y - 0.9));
            }
            if (left_edge.size() > 1) {
                std::vector<cv::Point> ribbon = left_edge;
                ribbon.insert(ribbon.end(), right_edge.rbegin(), right_edge.rend());
                cv::Mat overlay = img.clone();
                cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{ribbon}, kPath);
                cv::addWeighted(overlay, 0.22, img, 0.78, 0, img);
            }
            const double spacing = 4.0;
            double phase = std::fmod(odometer_m_, spacing);
            for (double sarc = spacing - phase; sarc < arc.back(); sarc += spacing) {
                std::size_t i = std::upper_bound(arc.begin(), arc.end(), sarc) - arc.begin();
                if (i == 0 || i >= P.size()) continue;
                double a = (sarc - arc[i - 1]) / std::max(1e-9, arc[i] - arc[i - 1]);
                cv::Point2d c = P[i - 1] + (P[i] - P[i - 1]) * a;
                if (c.x < x0 + 0.8) continue;
                cv::Point2d dir = P[i] - P[i - 1];
                dir *= 1.0 / std::max(1e-9, std::hypot(dir.x, dir.y));
                cv::Point2d nrm(-dir.y, dir.x);
                cv::Point2d tip = c + dir * 0.6;
                cv::Point2d l = c + nrm * 0.6, r = c - nrm * 0.6;
                cv::line(img, proj(l.x, l.y), proj(tip.x, tip.y), kPath, 3, cv::LINE_AA);
                cv::line(img, proj(r.x, r.y), proj(tip.x, tip.y), kPath, 3, cv::LINE_AA);
            }
        }

        // The lane's own vanishing point: where it would meet the horizon if straight.
        cv::circle(img, f.geometry.lane_vanishing_px, 7, sc, 2, cv::LINE_AA);
    }

    // Raw evidence: detected paint points, green = used by the tracker,
    // red = rejected by the gate.
    for (const LaneMeasurement& m : f.detection.lanes) {
        cv::Scalar c = m.accepted ? kAccepted : kRejected;
        for (std::size_t i = 0; i < m.points.size(); i += 4) {
            cv::circle(img, proj(m.points[i].x, m.points[i].y), 2, c, cv::FILLED);
        }
    }

    for (const TrackedObject& t : f.objects) {
        if (!t.confirmed) continue;
        cv::Scalar c = t.in_ego_lane ? cv::Scalar(60, 60, 255)
                                     : (t.class_id == 0 ? cv::Scalar(80, 200, 255) : cv::Scalar(255, 170, 90));
        cv::Rect r(t.box);
        if (t.matched) {
            cv::rectangle(img, r, c, t.in_ego_lane ? 3 : 2, cv::LINE_AA);
        } else {
            // Not detected this frame: position predicted by the tracker.
            std::vector<cv::Point> q = {r.tl(), {r.br().x, r.y}, r.br(), {r.x, r.br().y}, r.tl()};
            dashed(img, q, c, 2, 10, 8);
        }
        char b[128];
        if (t.ground_valid) {
            double sX = std::sqrt(t.P(0, 0));
            int n = std::snprintf(b, sizeof(b), "#%d %s %.1fm +/-%.1f %+.1fm/s", t.id, coco_name(t.class_id), t.g[0],
                                  sX, t.g[2]);
            if (t.ttc_s > 0 && t.ttc_s < 10) std::snprintf(b + n, sizeof(b) - n, " TTC %.1fs", t.ttc_s);
        } else {
            std::snprintf(b, sizeof(b), "#%d %s %.2f", t.id, coco_name(t.class_id), t.score);
        }
        int base = 0;
        cv::Size ts = cv::getTextSize(b, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::Point org(r.x, std::max(ts.height + 60, r.y - 6));
        cv::rectangle(img, {org.x - 2, org.y - ts.height - 4, ts.width + 4, ts.height + 8}, kDark, cv::FILLED);
        text(img, b, org, 0.5, c);
    }

    // Banner.
    cv::rectangle(img, {0, 0, img.cols, 56}, kDark, cv::FILLED);
    std::string banner = std::string("ATLAS VISION   ") + status_name(f.status);
    text(img, banner, {20, 38}, 1.0, status_color(f.status), 2);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "confidence %.2f", f.confidence);
    text(img, buf, {470, 38}, 0.8, kWhite, 1);
    int bar = static_cast<int>(200 * f.confidence);
    cv::rectangle(img, {700, 22, 200, 20}, kGrey, 1);
    cv::rectangle(img, {700, 22, bar, 20}, status_color(f.status), cv::FILLED);
    text(img, s.paused ? "PAUSED" : "", {1150, 38}, 0.8, kWhite, 2);

    cv::resize(img, panel, panel.size(), 0, 0, cv::INTER_AREA);
}

void Dashboard::draw_bev(const PerceptionFrame& f, const VisionPipeline& p, cv::Mat& panel,
                         bool features) {
    const GroundGrid& g = p.ground().grid();
    cv::Mat img;
    if (features) {
        cv::Mat bin;
        cv::cvtColor(f.detection.binary, bin, cv::COLOR_GRAY2BGR);
        img = bin * 0.8;
    } else {
        img = f.bev.clone();
    }
    auto proj = [&](double X, double Y) {
        return cv::Point(cvRound(g.col_of_y(Y)), cvRound(g.row_of_x(X)));
    };
    // Metric grid: every 5 m forward, every lane-width-ish 1 m sideways.
    for (double X = 10; X < g.x_max; X += 5) {
        int r = cvRound(g.row_of_x(X));
        cv::line(img, {0, r}, {img.cols, r}, cv::Scalar(70, 70, 70), 1);
        text(img, std::to_string(static_cast<int>(X)) + "m", {3, r - 3}, 0.4, kGrey);
    }
    int c0 = cvRound(g.col_of_y(0.0));
    cv::line(img, {c0, 0}, {c0, img.rows}, cv::Scalar(60, 60, 60), 1);

    if (features) {
        for (const SearchWindow& w : f.detection.windows) {
            cv::Scalar c = w.pixels > 0 ? cv::Scalar(200, 160, 60) : cv::Scalar(90, 70, 50);
            cv::rectangle(img, w.rect, c, 1);
        }
    }

    const RoadState& road = f.road;
    if (road.valid) {
        for (int si = 0; si < kLaneSlots; ++si) {
            LaneSlot slot = static_cast<LaneSlot>(si);
            double k = slot_width_multiple(slot);
            bool inner = slot == LaneSlot::Left || slot == LaneSlot::Right;
            const LaneMeasurement* m = nullptr;
            bool measured = measured_slot(f, slot, &m);
            if (!inner && !measured) continue;
            cv::Scalar c = inner ? kEgoLine : kOuterLine;
            std::vector<cv::Point> hi = boundary(road, k, g.x_min, g.x_max, 2.0, proj);
            std::vector<cv::Point> lo = boundary(road, k, g.x_min, g.x_max, -2.0, proj);
            cv::polylines(img, hi, false, c * 0.6, 1, cv::LINE_AA);
            cv::polylines(img, lo, false, c * 0.6, 1, cv::LINE_AA);
            std::vector<cv::Point> line = boundary(road, k, g.x_min, g.x_max, 0.0, proj);
            if (measured) {
                cv::polylines(img, line, false, c, 2, cv::LINE_AA);
            } else {
                dashed(img, line, c, 2, 10, 8);
            }
        }
        if (f.path.valid) {
            std::vector<cv::Point> pts;
            for (const cv::Point2d& q : f.path.points) {
                if (q.x >= g.x_min) pts.push_back(proj(q.x, q.y));
            }
            cv::polylines(img, pts, false, kPath, 2, cv::LINE_AA);
        }
    }
    for (const LaneMeasurement& m : f.detection.lanes) {
        cv::Scalar c = m.accepted ? kAccepted : kRejected;
        for (std::size_t i = 0; i < m.points.size(); i += 3) {
            cv::circle(img, proj(m.points[i].x, m.points[i].y), 1, c, cv::FILLED);
        }
    }
    if (!features) {
        // Objects on the road plane: a car-sized footprint behind the ground
        // contact, plus a bar showing +/- 1 sigma of range.
        for (const TrackedObject& t : f.objects) {
            if (!t.confirmed || !t.ground_valid) continue;
            double X = t.g[0], Y = t.g[1];
            cv::Scalar c = t.in_ego_lane ? cv::Scalar(60, 60, 255) : cv::Scalar(255, 170, 90);
            cv::Point a = proj(X + 4.5, Y + 0.9), b = proj(X, Y - 0.9);
            cv::rectangle(img, cv::Rect(a, b), c, 2);
            double sX = std::sqrt(t.P(0, 0));
            cv::line(img, proj(X - sX, Y), proj(X + sX, Y), c, 1);
            text(img, "#" + std::to_string(t.id), {b.x + 2, b.y}, 0.35, c);
        }
    }

    cv::Mat scaled;
    double scale = std::min(static_cast<double>(panel.cols) / img.cols,
                            static_cast<double>(panel.rows - 26) / img.rows);
    cv::resize(img, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
    panel.setTo(kDark);
    int ox = (panel.cols - scaled.cols) / 2;
    scaled.copyTo(panel(cv::Rect(ox, 26, scaled.cols, scaled.rows)));
    text(panel, features ? "PAINT FEATURES + SEARCH" : "BIRD'S-EYE VIEW (metric)", {6, 18}, 0.45, kWhite);
}

void Dashboard::draw_text(const PerceptionFrame& f, const DashboardStatus& s, cv::Mat& panel) {
    panel.setTo(kDark);
    char b[256];
    int y = 24;
    auto line = [&](const std::string& str, const cv::Scalar& c = kWhite, double sc = 0.5) {
        text(panel, str, {12, y}, sc, c);
        y += 21;
    };
    const RoadGeometry& g = f.geometry;
    if (g.valid) {
        std::snprintf(b, sizeof(b), "offset from lane centre  %+.2f m  (+/- %.2f)", g.lateral_offset_m,
                      g.lateral_offset_sigma_m);
        line(b);
        std::snprintf(b, sizeof(b), "lane heading vs car      %+.2f deg", g.heading_error_rad * 180.0 / CV_PI);
        line(b);
        double R = std::fabs(g.curvature_1pm) > 1e-5 ? 1.0 / g.curvature_1pm : 0.0;
        if (R != 0.0 && std::fabs(R) < 5000.0) {
            std::snprintf(b, sizeof(b), "curvature %+.5f 1/m (+/- %.5f)  R = %.0f m %s", g.curvature_1pm,
                          g.curvature_sigma_1pm, std::fabs(R), R > 0 ? "left" : "right");
        } else {
            std::snprintf(b, sizeof(b), "curvature %+.5f 1/m (+/- %.5f)  ~straight", g.curvature_1pm,
                          g.curvature_sigma_1pm);
        }
        line(b);
        std::snprintf(b, sizeof(b), "lane width               %.2f m  (+/- %.2f)", g.lane_width_m,
                      g.lane_width_sigma_m);
        line(b);
    } else {
        line("no road model: searching for two lane lines", status_color(TrackStatus::Searching));
        y += 63;
    }
    if (f.speed_measured) {
        std::snprintf(b, sizeof(b), "speed %.1f m/s (%.0f km/h) from ground optical flow, %d/%d inliers",
                      f.speed_mps, f.speed_mps * 3.6, f.ego.inliers, f.ego.tracked);
        line(b);
    } else {
        std::snprintf(b, sizeof(b), "speed unknown (optical flow: %d tracks, %d inliers)", f.ego.tracked,
                      f.ego.inliers);
        line(b, kGrey);
    }
    if (f.path.valid) {
        std::snprintf(b, sizeof(b), "pure pursuit steering %+.2f deg, lookahead %.0f m",
                      f.path.steering_rad * 180.0 / CV_PI, f.path.lookahead_m);
        line(b);
    } else {
        y += 21;
    }

    std::string lanes = "lines:";
    for (const LaneMeasurement& m : f.detection.lanes) {
        std::snprintf(b, sizeof(b), " %s[%s %s %s d2=%.1f]", slot_name(m.slot), m.dashed ? "dash" : "solid",
                      m.yellow ? "yel" : "wht", m.accepted ? "ok" : "REJ", m.mahalanobis2);
        lanes += b;
    }
    if (f.detection.lanes.empty()) lanes += " none detected";
    text(panel, lanes, {12, y}, 0.38, kWhite);
    y += 18;

    int confirmed = 0;
    for (const TrackedObject& o : f.objects) confirmed += o.confirmed ? 1 : 0;
    if (f.objects_enabled) {
        std::snprintf(b, sizeof(b), "objects: %zu detections, %d confirmed tracks (YOLOX, COCO classes)",
                      f.detections.size(), confirmed);
    } else {
        std::snprintf(b, sizeof(b), "objects: off");
    }
    text(panel, b, {12, y}, 0.38, kWhite);
    y += 18;

    const StageTimings& t = f.timings;
    std::snprintf(b, sizeof(b), "frame %lld t=%.1fs %.0f fps proc %.1f ms (undist %.1f obj %.1f bev %.1f flow %.1f lanes %.1f trk %.1f)",
                  f.frame_id, f.media_time_s, s.loop_fps, t.total_ms, t.undistort_ms, t.objects_ms, t.bev_ms,
                  t.ego_ms, t.lanes_ms, t.track_ms);
    text(panel, b, {12, y}, 0.38, kGrey);
    y += 18;
    std::snprintf(b, sizeof(b), "dropped %lld  read failures %lld  reconnects %lld  mem %.0f MiB",
                  s.source_stats.dropped, s.source_stats.read_failures, s.source_stats.reconnects, s.memory_mib);
    text(panel, b, {12, y}, 0.38, kGrey);
    y += 18;
    text(panel, s.logger, {12, y}, 0.38, kGrey);
    y += 18;
    for (const std::string& e : event_log_) {
        text(panel, e, {12, y}, 0.38, cv::Scalar(120, 200, 255));
        y += 16;
    }
    text(panel, "q quit  space pause  n step  f flow  r raw  g/b label good/bad", {12, panel.rows - 8}, 0.38, kGrey);
}

void Dashboard::draw_charts(cv::Mat& panel) {
    panel.setTo(cv::Scalar(22, 22, 22));
    struct Series {
        const char* name;
        double lo, hi;
        double (*value)(const HistorySample&);
        double (*sigma)(const HistorySample&);
    };
    const Series series[] = {
        {"offset m", -1.5, 1.5, [](const HistorySample& h) { return h.offset; },
         [](const HistorySample& h) { return h.offset_sigma; }},
        {"curvature 1/m", -0.01, 0.01, [](const HistorySample& h) { return h.curvature; },
         [](const HistorySample& h) { return h.curvature_sigma; }},
        {"width m", 2.5, 5.0, [](const HistorySample& h) { return h.width; }, nullptr},
        {"confidence", 0.0, 1.0, [](const HistorySample& h) { return h.confidence; }, nullptr},
        {"speed m/s", 0.0, 40.0, [](const HistorySample& h) { return h.speed; }, nullptr},
    };
    int n = 5;
    int h = (panel.rows - 14) / n;
    int x0 = 110, x1 = panel.cols - 10;
    for (int si = 0; si < n; ++si) {
        const Series& S = series[si];
        int top = 4 + si * h, bot = top + h - 8;
        cv::rectangle(panel, {x0, top, x1 - x0, bot - top}, cv::Scalar(55, 55, 55), 1);
        text(panel, S.name, {8, top + 18}, 0.42, kWhite);
        char b[32];
        std::snprintf(b, sizeof(b), "%g", S.hi);
        text(panel, b, {8, top + 34}, 0.35, kGrey);
        std::snprintf(b, sizeof(b), "%g", S.lo);
        text(panel, b, {8, bot}, 0.35, kGrey);
        auto ymap = [&](double v) {
            double a = std::clamp((v - S.lo) / (S.hi - S.lo), 0.0, 1.0);
            return cvRound(bot - a * (bot - top));
        };
        if (S.lo < 0 && S.hi > 0) {
            cv::line(panel, {x0, ymap(0)}, {x1, ymap(0)}, cv::Scalar(60, 60, 60), 1);
        }
        std::size_t N = history_.size();
        cv::Point prev(-1, -1);
        for (std::size_t i = 0; i < N; ++i) {
            const HistorySample& hs = history_[i];
            int x = x0 + static_cast<int>((x1 - x0) * static_cast<double>(i + kHistory - N) / (kHistory - 1));
            bool ok = si == 3 || (si == 4 ? hs.speed_measured : hs.valid);
            if (si == 3) {
                cv::line(panel, {x, bot + 1}, {x, bot + 5}, status_color(hs.status), 1);
            }
            if (!ok) {
                prev = {-1, -1};
                continue;
            }
            double v = S.value(hs);
            if (S.sigma) {
                double sg = S.sigma(hs);
                cv::line(panel, {x, ymap(v - 2 * sg)}, {x, ymap(v + 2 * sg)}, cv::Scalar(90, 70, 50), 1);
            }
            cv::Point p(x, ymap(v));
            if (prev.x >= 0) {
                cv::line(panel, prev, p, si == 3 ? status_color(hs.status) : kEgoLine, 1, cv::LINE_AA);
            }
            prev = p;
        }
    }
    text(panel, "last 15 s  (bands = +/- 2 sigma; strip = tracker status)", {x0, panel.rows - 2}, 0.35, kGrey);
}

void Dashboard::observe(const PerceptionFrame& f) {
    if (f.speed_measured) {
        odometer_m_ += f.speed_mps * f.dt_s;
    }
    HistorySample hs;
    hs.t = f.media_time_s;
    hs.valid = f.geometry.valid;
    hs.status = f.status;
    hs.offset = f.geometry.lateral_offset_m;
    hs.offset_sigma = f.geometry.lateral_offset_sigma_m;
    hs.curvature = f.geometry.curvature_1pm;
    hs.curvature_sigma = f.geometry.curvature_sigma_1pm;
    hs.width = f.geometry.lane_width_m;
    hs.confidence = f.confidence;
    hs.speed = f.speed_mps;
    hs.speed_measured = f.speed_measured;
    history_.push_back(hs);
    while (history_.size() > kHistory) history_.pop_front();
    for (const TrackEvent& e : f.events) {
        char b[160];
        std::snprintf(b, sizeof(b), "t=%.1fs %s", e.time_s, e.what.c_str());
        event_log_.push_back(b);
        while (event_log_.size() > 3) event_log_.pop_front();
    }
}

const cv::Mat& Dashboard::render(const PerceptionFrame& f, const VisionPipeline& pipeline,
                                 const DashboardStatus& status) {
    canvas_.setTo(kDark);
    cv::Mat cam = canvas_(kCameraRect);
    draw_camera(f, pipeline, status, cam);
    cv::Mat bev = canvas_(kBevRect);
    draw_bev(f, pipeline, bev, false);
    cv::Mat feat = canvas_(kFeatureRect);
    draw_bev(f, pipeline, feat, true);
    cv::Mat txt = canvas_(kTextRect);
    draw_text(f, status, txt);
    cv::Mat charts = canvas_(kChartRect);
    draw_charts(charts);
    return canvas_;
}
