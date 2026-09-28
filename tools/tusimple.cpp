#include "tusimple.hpp"
#include "lane_rasterize.hpp"
#include "../vision/cv_compat.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>

namespace {

std::vector<int> read_ints(const cv::FileNode& n) {
    std::vector<int> v;
    for (const cv::FileNode& e : n) v.push_back(static_cast<int>(std::lround(static_cast<double>(e))));
    return v;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    return v[m];
}

struct ImageLine {
    double a, b;  // x = a y + b, undistorted pixels
    std::vector<cv::Point2d> pts;
};

// Straight lines through the near part of each labelled lane (the far part
// bends on curves), in undistorted pixels.
std::vector<ImageLine> near_lines(const TuSimpleLabel& l, const CameraModel& camera) {
    std::vector<ImageLine> out;
    for (const std::vector<int>& lane : l.lanes) {
        if (lane.size() != l.h_samples.size()) continue;
        std::vector<cv::Point2d> pts;
        for (std::size_t i = 0; i < lane.size(); ++i) {
            if (lane[i] >= 0) pts.push_back({static_cast<double>(lane[i]), static_cast<double>(l.h_samples[i])});
        }
        if (pts.size() < 6) continue;
        std::sort(pts.begin(), pts.end(), [](const cv::Point2d& p, const cv::Point2d& q) { return p.y < q.y; });
        pts.erase(pts.begin(), pts.begin() + static_cast<std::ptrdiff_t>(pts.size() / 3));
        if (has_distortion(camera)) {
            std::vector<cv::Point2d> und;
            cv::undistortPoints(pts, und, camera.K, camera.dist, cv::noArray(), camera.K);
            pts = und;
        }
        double sy = 0, sx = 0, syy = 0, sxy = 0, n = static_cast<double>(pts.size());
        for (const cv::Point2d& p : pts) {
            sy += p.y;
            sx += p.x;
            syy += p.y * p.y;
            sxy += p.x * p.y;
        }
        double var = syy - sy * sy / n;
        if (var < 1e-6) continue;
        double a = (sxy - sx * sy / n) / var;
        double b = (sx - a * sy) / n;
        double ss = 0;
        for (const cv::Point2d& p : pts) ss += std::pow(p.x - (a * p.y + b), 2);
        if (std::sqrt(ss / n) > 3.0) continue;  // curved: not a straight line
        out.push_back({a, b, pts});
    }
    return out;
}

}  // namespace

bool parse_tusimple_label(const std::string& json_line, TuSimpleLabel& out, std::string& error) {
    try {
        cv::FileStorage fs(json_line, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        if (!fs.isOpened()) {
            error = "not JSON";
            return false;
        }
        out.raw_file = static_cast<std::string>(fs["raw_file"]);
        out.h_samples = read_ints(fs["h_samples"]);
        out.lanes.clear();
        for (const cv::FileNode& lane : fs["lanes"]) out.lanes.push_back(read_ints(lane));
    } catch (const cv::Exception& e) {
        error = e.what();
        return false;
    }
    if (out.raw_file.empty() || out.h_samples.empty()) {
        error = "missing raw_file or h_samples";
        return false;
    }
    for (const std::vector<int>& lane : out.lanes) {
        if (lane.size() != out.h_samples.size()) {
            error = out.raw_file + ": lane length != h_samples length";
            return false;
        }
    }
    return true;
}

bool load_tusimple_labels(const std::string& path, std::vector<TuSimpleLabel>& out, std::string& error) {
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
        TuSimpleLabel l;
        std::string e;
        if (!parse_tusimple_label(line, l, e)) {
            error = path + ":" + std::to_string(line_no) + ": " + e;
            return false;
        }
        out.push_back(std::move(l));
    }
    return true;
}

double tusimple_lane_angle(const std::vector<int>& xs, const std::vector<int>& ys) {
    // sklearn LinearRegression().fit(ys[:, None], xs).coef_[0]
    double n = 0, sx = 0, sy = 0, syy = 0, sxy = 0;
    for (std::size_t i = 0; i < xs.size() && i < ys.size(); ++i) {
        if (xs[i] < 0) continue;
        n += 1;
        sx += xs[i];
        sy += ys[i];
        syy += static_cast<double>(ys[i]) * ys[i];
        sxy += static_cast<double>(xs[i]) * ys[i];
    }
    if (n <= 1) return 0.0;
    double var = syy - sy * sy / n;
    if (var <= 0) return 0.0;
    return std::atan((sxy - sx * sy / n) / var);
}

double tusimple_line_accuracy(const std::vector<int>& pred, const std::vector<int>& gt, double thresh) {
    if (gt.empty()) return 0.0;
    int hits = 0;
    for (std::size_t i = 0; i < gt.size(); ++i) {
        double p = i < pred.size() && pred[i] >= 0 ? pred[i] : -100.0;
        double g = gt[i] >= 0 ? gt[i] : -100.0;
        hits += std::fabs(p - g) < thresh ? 1 : 0;
    }
    return static_cast<double>(hits) / static_cast<double>(gt.size());
}

bool tusimple_bench(const std::vector<std::vector<int>>& pred, const std::vector<std::vector<int>>& gt,
                    const std::vector<int>& y_samples, double run_time_ms, TuSimpleScore& s,
                    const TuSimpleParams& p) {
    s = TuSimpleScore();
    for (const std::vector<int>& lane : pred) {
        if (lane.size() != y_samples.size()) return false;
    }
    if (run_time_ms > p.max_run_time_ms || gt.size() + 2 < pred.size()) {
        s.accuracy = 0.0;
        s.fp = 0.0;
        s.fn = 1.0;
        s.rejected = true;
        return true;
    }
    double fn = 0.0;
    for (const std::vector<int>& g : gt) {
        double thresh = p.pixel_thresh / std::cos(tusimple_lane_angle(g, y_samples));
        double best = 0.0;
        for (const std::vector<int>& q : pred) best = std::max(best, tusimple_line_accuracy(q, g, thresh));
        if (best < p.pt_thresh) fn += 1.0;
        else ++s.matched;
        s.line_accuracy.push_back(best);
    }
    double fp = static_cast<double>(pred.size()) - s.matched;
    if (gt.size() > 4 && fn > 0) fn -= 1.0;
    double sum = 0.0;
    for (double a : s.line_accuracy) sum += a;
    if (gt.size() > 4) sum -= *std::min_element(s.line_accuracy.begin(), s.line_accuracy.end());
    double denom = std::max(std::min(4.0, static_cast<double>(gt.size())), 1.0);
    s.accuracy = sum / denom;
    s.fp = pred.empty() ? 0.0 : fp / static_cast<double>(pred.size());
    s.fn = fn / denom;
    return true;
}

bool fit_mount_from_labels(const std::vector<TuSimpleLabel>& labels, int max_images, double lane_width_m,
                           CameraModel& camera, MountFit& fit, std::string& error) {
    fit = MountFit();
    const cv::Size size = camera.image_size;
    std::vector<double> us, vs;
    std::vector<std::vector<ImageLine>> kept;
    int n = 0;
    for (const TuSimpleLabel& l : labels) {
        if (n++ >= max_images) break;
        std::vector<ImageLine> lines = near_lines(l, camera);
        if (lines.size() < 2) continue;
        // Least-squares intersection: minimise the summed squared
        // perpendicular distance to every line  n . p = d.
        cv::Matx22d A = cv::Matx22d::zeros();
        cv::Vec2d r(0, 0);
        std::vector<std::pair<cv::Vec2d, double>> nd;
        for (const ImageLine& li : lines) {
            double s = std::sqrt(1.0 + li.a * li.a);
            cv::Vec2d nv(1.0 / s, -li.a / s);
            double d = li.b / s;
            A += nv * nv.t();
            r += nv * d;
            nd.push_back({nv, d});
        }
        // Nearly parallel lines (e.g. two lanes on the same side) pin the
        // point down only along one direction.
        double tr = A(0, 0) + A(1, 1), det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        double min_eig = 0.5 * (tr - std::sqrt(std::max(0.0, tr * tr - 4.0 * det)));
        if (min_eig < 0.02) continue;
        cv::Vec2d vp = A.inv() * r;
        double worst = 0.0;
        for (const auto& e : nd) worst = std::max(worst, std::fabs(e.first.dot(vp) - e.second));
        if (worst > 15.0) continue;
        if (vp[1] < 0 || vp[1] > size.height || vp[0] < -size.width || vp[0] > 2.0 * size.width) continue;
        us.push_back(vp[0]);
        vs.push_back(vp[1]);
        kept.push_back(std::move(lines));
    }
    fit.images_used = static_cast<int>(us.size());
    if (us.size() < 5) {
        error = "only " + std::to_string(us.size()) + " labelled images with a clear vanishing point";
        return false;
    }
    fit.vp_u_px = median(us);
    fit.vp_v_px = median(vs);
    std::vector<double> dev;
    for (double v : vs) dev.push_back(std::fabs(v - fit.vp_v_px));
    fit.vp_spread_px = median(dev);

    // Forward direction (1, 0, 0) projects to
    //   u = cx + fx tan(yaw) / cos(pitch),  v = cy - fy tan(pitch).
    double fx = camera.K(0, 0), fy = camera.K(1, 1), cx = camera.K(0, 2), cy = camera.K(1, 2);
    camera.pitch_rad = std::atan((cy - fit.vp_v_px) / fy);
    camera.yaw_rad = std::atan((fit.vp_u_px - cx) * std::cos(camera.pitch_rad) / fx);
    camera.height_m = 1.0;
    camera.update();

    // Everything on the ground scales with camera height, so measure the ego
    // lane width with h = 1 and scale to the assumed real width.
    std::vector<double> widths;
    for (const std::vector<ImageLine>& lines : kept) {
        double best_left = 1e9, best_right = -1e9, c1_left = 0, c1_right = 0;
        for (const ImageLine& li : lines) {
            std::vector<cv::Point2d> g;
            for (const cv::Point2d& p : li.pts) {
                double X, Y;
                if (camera.image_to_ground(p, X, Y)) g.push_back({X, Y});
            }
            if (g.size() < 3) continue;
            double n2 = static_cast<double>(g.size()), sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (const cv::Point2d& q : g) {
                sx += q.x;
                sy += q.y;
                sxx += q.x * q.x;
                sxy += q.x * q.y;
            }
            double var = sxx - sx * sx / n2;
            if (var < 1e-9) continue;
            double c1 = (sxy - sx * sy / n2) / var;
            double c0 = (sy - c1 * sx) / n2;
            if (c0 > 0 && c0 < best_left) {
                best_left = c0;
                c1_left = c1;
            }
            if (c0 < 0 && c0 > best_right) {
                best_right = c0;
                c1_right = c1;
            }
        }
        if (best_left > 1e8 || best_right < -1e8) continue;
        double heading = std::atan(0.5 * (c1_left + c1_right));
        widths.push_back((best_left - best_right) * std::cos(heading));
    }
    fit.images_for_height = static_cast<int>(widths.size());
    if (widths.size() < 5) {
        error = "only " + std::to_string(widths.size()) + " images show both ego lane lines";
        return false;
    }
    fit.lane_width_at_unit_height = median(widths);
    camera.height_m = lane_width_m / fit.lane_width_at_unit_height;
    camera.mount_calibrated = true;
    camera.update();
    return true;
}
