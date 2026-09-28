#include "lane_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace {

double robust_sigma(std::vector<float>& v, double& median_out) {
    if (v.empty()) {
        median_out = 0.0;
        return 0.0;
    }
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    double med = v[v.size() / 2];
    for (float& x : v) {
        x = std::fabs(x - static_cast<float>(med));
    }
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    median_out = med;
    // For Gaussian noise, sigma = 1.4826 * median absolute deviation.
    return 1.4826 * v[v.size() / 2];
}

// Ridge response along each row: how much brighter a thin stripe is than
// BOTH of its neighbours. A shadow edge is bright on one side only, so the
// min() of the two contrasts stays near zero there, while paint scores high.
void ridge_filter(const cv::Mat& channel, const cv::Mat& valid, int stripe_px, int side_px,
                  cv::Mat& ridge) {
    cv::Mat f, centre, sides;
    channel.convertTo(f, CV_32F);
    cv::blur(f, centre, cv::Size(stripe_px, 1));
    cv::blur(f, sides, cv::Size(5, 1));
    ridge = cv::Mat::zeros(channel.size(), CV_32F);
    for (int r = 0; r < f.rows; ++r) {
        const float* c = centre.ptr<float>(r);
        const float* s = sides.ptr<float>(r);
        const uchar* m = valid.ptr<uchar>(r);
        float* out = ridge.ptr<float>(r);
        for (int x = side_px; x < f.cols - side_px; ++x) {
            if (!m[x] || !m[x - side_px] || !m[x + side_px]) {
                continue;
            }
            out[x] = std::min(c[x] - s[x - side_px], c[x] - s[x + side_px]);
        }
    }
}

double threshold_for(const cv::Mat& ridge, const cv::Mat& valid, double k, double floor_value,
                     double& noise) {
    std::vector<float> samples;
    samples.reserve(ridge.total() / 8);
    for (int r = 0; r < ridge.rows; r += 2) {
        const float* p = ridge.ptr<float>(r);
        const uchar* m = valid.ptr<uchar>(r);
        for (int x = 0; x < ridge.cols; x += 4) {
            if (m[x]) {
                samples.push_back(p[x]);
            }
        }
    }
    double med = 0.0;
    noise = robust_sigma(samples, med);
    return std::max(floor_value, med + k * noise);
}

// Ridge-weighted centroid of feature pixels in [c0, c1] on one row.
bool row_centroid(const LaneDetection& det, int row, int c0, int c1, double& col, int& count) {
    c0 = std::max(c0, 0);
    c1 = std::min(c1, det.binary.cols - 1);
    const uchar* b = det.binary.ptr<uchar>(row);
    const float* rl = det.ridge_L.ptr<float>(row);
    const float* rb = det.ridge_b.ptr<float>(row);
    double sw = 0.0, swx = 0.0;
    count = 0;
    for (int x = c0; x <= c1; ++x) {
        if (!b[x]) {
            continue;
        }
        double w = std::max(rl[x], rb[x]);
        sw += w;
        swx += w * x;
        ++count;
    }
    if (count == 0 || sw <= 0.0) {
        return false;
    }
    col = swx / sw;
    return true;
}

}  // namespace

bool fit_lane_polynomial(const std::vector<cv::Point2d>& pts, double c2_prior_sigma,
                         double correlation_inflation, cv::Vec3d& coeffs, cv::Matx33d& cov,
                         double& rms, std::vector<cv::Point2d>* inliers) {
    std::vector<cv::Point2d> use = pts;
    for (int pass = 0; pass < 2; ++pass) {
        if (use.size() < 6) {
            return false;
        }
        // Normal equations A^T A c = A^T y with rows a_i = [1, X, X^2].
        cv::Matx33d AtA = cv::Matx33d::zeros();
        cv::Vec3d Aty(0, 0, 0);
        for (const cv::Point2d& p : use) {
            cv::Vec3d a(1.0, p.x, p.x * p.x);
            AtA += a * a.t();
            Aty += a * p.y;
        }
        // First solve with a weak prior to measure the residual noise sigma^2.
        double prior_info = 1.0 / (c2_prior_sigma * c2_prior_sigma);
        cv::Matx33d N0 = AtA;
        N0(2, 2) += 0.05 * 0.05 * prior_info;
        cv::Vec3d c = N0.solve(Aty, cv::DECOMP_CHOLESKY);
        double sse = 0.0;
        for (const cv::Point2d& p : use) {
            double r = p.y - (c[0] + c[1] * p.x + c[2] * p.x * p.x);
            sse += r * r;
        }
        rms = std::sqrt(sse / use.size());

        // MAP estimate: minimise |y - A c|^2 / s^2 + c2^2 / sp^2.
        double s2 = std::max(rms * rms, 0.03 * 0.03) * correlation_inflation;
        cv::Matx33d N = AtA * (1.0 / s2);
        N(2, 2) += prior_info;
        coeffs = N.solve(Aty * (1.0 / s2), cv::DECOMP_CHOLESKY);
        cov = N.inv(cv::DECOMP_CHOLESKY);

        if (pass == 0) {
            double gate = std::max(0.2, 3.0 * rms);
            std::vector<cv::Point2d> kept;
            for (const cv::Point2d& p : use) {
                double r = p.y - (coeffs[0] + coeffs[1] * p.x + coeffs[2] * p.x * p.x);
                if (std::fabs(r) <= gate) {
                    kept.push_back(p);
                }
            }
            use.swap(kept);
        }
    }
    if (inliers) {
        *inliers = use;
    }
    return true;
}

void LaneDetector::extract_features(const cv::Mat& bev_bgr, const cv::Mat& valid,
                                    const GroundGrid& grid, LaneDetection& out) const {
    cv::cvtColor(bev_bgr, lab_, cv::COLOR_BGR2Lab);
    cv::Mat ch[3];
    cv::split(lab_, ch);
    int stripe = std::max(1, static_cast<int>(std::lround(p_.marking_width_m / grid.resolution_m)));
    if (stripe % 2 == 0) {
        ++stripe;
    }
    int side = std::max(stripe, static_cast<int>(std::lround(p_.side_offset_m / grid.resolution_m)));
    ridge_filter(ch[0], valid, stripe, side, out.ridge_L);
    // In 8-bit Lab, b = 128 is neutral and b > 128 is yellow. Yellow paint on
    // pale concrete has little lightness contrast but strong b contrast.
    ridge_filter(ch[2], valid, stripe, side, out.ridge_b);

    double tL = threshold_for(out.ridge_L, valid, p_.noise_k, p_.min_contrast_L, out.noise_L);
    double tb = threshold_for(out.ridge_b, valid, p_.noise_k, p_.min_contrast_b, out.noise_b);
    out.binary = cv::Mat::zeros(valid.size(), CV_8UC1);
    for (int r = 0; r < valid.rows; ++r) {
        const float* rl = out.ridge_L.ptr<float>(r);
        const float* rb = out.ridge_b.ptr<float>(r);
        const uchar* bch = ch[2].ptr<uchar>(r);
        uchar* o = out.binary.ptr<uchar>(r);
        for (int x = 0; x < valid.cols; ++x) {
            if (rl[x] > tL || (rb[x] > tb && bch[x] > 138)) {
                o[x] = 255;
            }
        }
    }
}

bool LaneDetector::fit(const GroundGrid& grid, const cv::Mat& bev_lab, LaneMeasurement& m) const {
    if (static_cast<int>(m.points.size()) < p_.min_points) {
        return false;
    }
    std::vector<cv::Point2d> inliers;
    if (!fit_lane_polynomial(m.points, p_.c2_prior_sigma, p_.correlation_inflation, m.coeffs,
                             m.cov, m.rms_m, &inliers)) {
        return false;
    }
    if (static_cast<int>(inliers.size()) < p_.min_points) {
        return false;
    }
    m.points.swap(inliers);
    m.x_near = 1e9;
    m.x_far = -1e9;
    for (const cv::Point2d& p : m.points) {
        m.x_near = std::min(m.x_near, p.x);
        m.x_far = std::max(m.x_far, p.x);
    }
    if (m.x_far - m.x_near < p_.min_span_m) {
        return false;
    }

    // Coverage: fraction of 2 m bins between near and far ends holding paint.
    int bins = std::max(1, static_cast<int>(std::ceil((m.x_far - m.x_near) / 2.0)));
    std::vector<int> hits(bins, 0);
    double b_sum = 0.0;
    for (const cv::Point2d& p : m.points) {
        int i = std::min(bins - 1, static_cast<int>((p.x - m.x_near) / 2.0));
        ++hits[i];
        int row = static_cast<int>(grid.row_of_x(p.x) + 0.5);
        int col = static_cast<int>(grid.col_of_y(p.y) + 0.5);
        if (row >= 0 && row < bev_lab.rows && col >= 0 && col < bev_lab.cols) {
            b_sum += bev_lab.at<cv::Vec3b>(row, col)[2];
        }
    }
    int filled = 0;
    for (int h : hits) {
        filled += h >= 3 ? 1 : 0;
    }
    m.coverage = static_cast<double>(filled) / bins;
    m.dashed = m.coverage < 0.75 && (m.x_far - m.x_near) > 10.0;
    m.yellow = b_sum / m.points.size() > 145.0;
    return true;
}

bool LaneDetector::trace_guided(const LaneDetection& det, const GroundGrid& grid,
                                const RoadState& prior, LaneSlot slot, LaneMeasurement& m,
                                std::vector<SearchWindow>& windows) const {
    double k = slot_width_multiple(slot);
    int win_rows = static_cast<int>(p_.window_height_m / grid.resolution_m);
    m.slot = slot;
    m.guided = true;
    for (int top = 0; top < det.binary.rows; top += win_rows) {
        int bottom = std::min(det.binary.rows, top + win_rows);
        int min_c = det.binary.cols, max_c = 0, pixels = 0;
        for (int r = top; r < bottom; ++r) {
            double X = grid.x_of_row(r);
            // The search band widens with the tracker's uncertainty at this range.
            double hw = std::clamp(2.5 * prior.lateral_sigma(X, k) + 0.25,
                                   p_.min_guided_half_width_m, p_.max_guided_half_width_m) /
                        grid.resolution_m;
            double c = grid.col_of_y(prior.lateral(X, k));
            int c0 = static_cast<int>(c - hw), c1 = static_cast<int>(c + hw);
            min_c = std::min(min_c, c0);
            max_c = std::max(max_c, c1);
            double col;
            int count;
            if (row_centroid(det, r, c0, c1, col, count)) {
                m.points.push_back({X, grid.y_of_col(col)});
                pixels += count;
            }
        }
        if (max_c >= 0 && min_c < det.binary.cols) {
            windows.push_back({cv::Rect(min_c, top, max_c - min_c, bottom - top), slot, pixels});
        }
    }
    return true;
}

bool LaneDetector::trace_fresh(const LaneDetection& det, const GroundGrid& grid, double seed_col,
                               LaneSlot slot, LaneMeasurement& m,
                               std::vector<SearchWindow>& windows) const {
    int win_rows = static_cast<int>(p_.window_height_m / grid.resolution_m);
    int hw = static_cast<int>(p_.fresh_half_width_m / grid.resolution_m);
    m.slot = slot;
    m.guided = false;
    double centre = seed_col;
    double slope = 0.0;  // columns per window, from the last two windows that found paint
    double last_found_centre = seed_col;
    int windows_since_found = 0;
    bool any_found = false;
    for (int bottom = det.binary.rows; bottom > 0; bottom -= win_rows) {
        int top = std::max(0, bottom - win_rows);
        int c0 = static_cast<int>(centre) - hw, c1 = static_cast<int>(centre) + hw;
        double sum_col = 0.0;
        int found_rows = 0, pixels = 0;
        for (int r = top; r < bottom; ++r) {
            double col;
            int count;
            if (row_centroid(det, r, c0, c1, col, count)) {
                m.points.push_back({grid.x_of_row(r), grid.y_of_col(col)});
                sum_col += col;
                ++found_rows;
                pixels += count;
            }
        }
        windows.push_back({cv::Rect(c0, top, c1 - c0, bottom - top), slot, pixels});
        ++windows_since_found;
        if (found_rows >= 5) {
            double new_centre = sum_col / found_rows;
            if (any_found) {
                slope = (new_centre - last_found_centre) / windows_since_found;
            }
            last_found_centre = new_centre;
            centre = new_centre + slope;
            windows_since_found = 0;
            any_found = true;
        } else {
            centre += slope;
        }
        if (centre < 0 || centre >= det.binary.cols) {
            break;
        }
    }
    return true;
}

void LaneDetector::detect(const cv::Mat& bev_bgr, const cv::Mat& valid, const GroundGrid& grid,
                          const RoadState* prior, LaneDetection& out) const {
    out = LaneDetection();
    extract_features(bev_bgr, valid, grid, out);

    // Column histogram of paint in the near field (most reliable, least blur).
    int cols = out.binary.cols;
    out.histogram.assign(cols, 0.0f);
    int r_near = out.binary.rows - 1;
    int r_far = std::max(0, static_cast<int>(grid.row_of_x(grid.x_min + 14.0)));
    for (int r = r_far; r <= r_near; ++r) {
        const uchar* b = out.binary.ptr<uchar>(r);
        for (int x = 0; x < cols; ++x) {
            out.histogram[x] += b[x] ? 1.0f : 0.0f;
        }
    }
    cv::Mat h(1, cols, CV_32F, out.histogram.data());
    cv::GaussianBlur(h, h, cv::Size(0, 0), 3.0, 0.0);

    out.used_prior = prior && prior->valid;
    std::vector<LaneMeasurement> found;
    if (out.used_prior) {
        for (int s = 0; s < kLaneSlots; ++s) {
            LaneMeasurement m;
            trace_guided(out, grid, *prior, static_cast<LaneSlot>(s), m, out.windows);
            if (fit(grid, lab_, m)) {
                found.push_back(m);
            }
        }
    } else {
        struct Peak { double y = 0.0; float v = 0.0f; int col = 0; };
        std::vector<Peak> peaks;
        int sep = static_cast<int>(1.2 / grid.resolution_m);
        for (int x = 1; x + 1 < cols; ++x) {
            float v = out.histogram[x];
            if (v < 25.0f) {
                continue;
            }
            bool is_max = true;
            for (int d = -sep; d <= sep && is_max; ++d) {
                int j = x + d;
                if (j >= 0 && j < cols && j != x && out.histogram[j] > v) {
                    is_max = false;
                }
            }
            if (is_max) {
                peaks.push_back({grid.y_of_col(x), v, x});
            }
        }
        // Pick the peak that best matches a line near +/- half a lane width.
        auto pick = [&](double expected_y, double lo, double hi, Peak& best) {
            double best_score = 0.0;
            for (const Peak& p : peaks) {
                if (p.y < lo || p.y > hi) {
                    continue;
                }
                double d = (p.y - expected_y) / 1.2;
                double score = p.v * std::exp(-d * d);
                if (score > best_score) {
                    best_score = score;
                    best = p;
                }
            }
            return best_score > 0.0;
        };
        Peak left, right, lo, ro;
        bool has_left = pick(1.85, 0.3, 3.3, left);
        bool has_right = pick(-1.85, -3.3, -0.3, right);
        auto trace = [&](const Peak& p, LaneSlot slot) {
            LaneMeasurement m;
            trace_fresh(out, grid, p.col, slot, m, out.windows);
            if (fit(grid, lab_, m)) {
                found.push_back(m);
            }
        };
        if (has_left) {
            trace(left, LaneSlot::Left);
            if (pick(left.y + 3.7, left.y + 2.7, left.y + 4.8, lo)) {
                trace(lo, LaneSlot::LeftOuter);
            }
        }
        if (has_right) {
            trace(right, LaneSlot::Right);
            if (pick(right.y - 3.7, right.y - 4.8, right.y - 2.7, ro)) {
                trace(ro, LaneSlot::RightOuter);
            }
        }
    }
    out.lanes = found;
}
