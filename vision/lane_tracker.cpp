#include "lane_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

void LaneTracker::reset(const std::string& why) {
    if (state_.valid) {
        events_.push_back({time_s_, "reset: " + why});
    }
    state_ = RoadState();
    status_ = TrackStatus::Searching;
    for (double& t : last_update_) {
        t = -1e9;
    }
    last_any_update_ = -1e9;
    consecutive_rejections_ = 0;
    split_frames_ = 0;
}

void LaneTracker::predict(double time_s, double forward_m, double yaw_rad, double forward_var,
                          double yaw_var) {
    if (!has_time_) {
        time_s_ = time_s;
        has_time_ = true;
        return;
    }
    double dt = std::clamp(time_s - time_s_, 0.0, 0.5);
    time_s_ = time_s;
    if (!state_.valid) {
        return;
    }

    // Driving forward by d re-expresses the road polynomial around X' = X - d:
    //   Y(X' + d) = (c0 + c1 d + c2 d^2) + (c1 + 2 c2 d) X' + c2 X'^2
    // Turning left by psi rotates the road the other way: c1 -= psi.
    double d = forward_m;
    cv::Matx44d F(1, d, d * d, 0,
                  0, 1, 2 * d, 0,
                  0, 0, 1, 0,
                  0, 0, 0, 1);
    cv::Vec4d x0 = state_.x;
    cv::Vec4d x = F * x0;
    x[1] -= yaw_rad;
    state_.x = x;
    cv::Matx44d Q = cv::Matx44d::diag(cv::Vec4d(p_.q_offset * p_.q_offset, p_.q_heading * p_.q_heading,
                                                p_.q_curv * p_.q_curv, p_.q_width * p_.q_width)) * dt;
    // Odometry is itself uncertain. Its noise enters through the Jacobian of
    // the motion model with respect to the inputs (d, psi):
    //   dx/dd = [c1 + 2 c2 d, 2 c2, 0, 0],   dx/dpsi = [0, -1, 0, 0]
    cv::Vec4d Gd(x0[1] + 2 * x0[2] * d, 2 * x0[2], 0, 0);
    cv::Vec4d Gpsi(0, -1, 0, 0);
    cv::Matx44d Qu = Gd * Gd.t() * forward_var + Gpsi * Gpsi.t() * yaw_var;
    state_.P = F * state_.P * F.t() + Q + Qu;

    double coast = time_s_ - last_any_update_;
    double sigma = state_.lateral_sigma(15.0, 0.0);
    if (coast > p_.max_coast_s) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "no lane measurement for %.1f s", coast);
        reset(buf);
    } else if (sigma > p_.lost_sigma_m) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "centre uncertainty %.2f m at 15 m", sigma);
        reset(buf);
    }
}

namespace {

cv::Matx<double, 3, 4> measurement_matrix(LaneSlot slot) {
    return cv::Matx<double, 3, 4>(1, 0, 0, slot_width_multiple(slot),
                                  0, 1, 0, 0,
                                  0, 0, 1, 0);
}

}  // namespace

cv::Matx33d LaneTracker::measurement_noise(const LaneMeasurement& m) const {
    // Total measurement noise = fit noise + model error; they are independent,
    // so their covariances add.
    cv::Matx33d R = m.cov + cv::Matx33d::diag(cv::Vec3d(p_.model_sigma_c0 * p_.model_sigma_c0,
                                                        p_.model_sigma_c1 * p_.model_sigma_c1,
                                                        p_.model_sigma_c2 * p_.model_sigma_c2));
    bool outer = m.slot == LaneSlot::LeftOuter || m.slot == LaneSlot::RightOuter;
    return outer ? R * p_.outer_cov_scale : R;
}

double LaneTracker::mahalanobis2(const LaneMeasurement& m) const {
    // Mahalanobis distance: how many "sigmas" the measurement is from the
    // prediction, accounting for both prediction and measurement uncertainty.
    cv::Matx<double, 3, 4> H = measurement_matrix(m.slot);
    cv::Vec3d innovation = m.coeffs - H * state_.x;
    cv::Matx33d S = H * state_.P * H.t() + measurement_noise(m);
    return (innovation.t() * S.inv(cv::DECOMP_CHOLESKY) * innovation)(0);
}

bool LaneTracker::apply(LaneMeasurement& m) {
    cv::Matx<double, 3, 4> H = measurement_matrix(m.slot);
    cv::Matx33d R = measurement_noise(m);
    cv::Vec3d innovation = m.coeffs - H * state_.x;
    cv::Matx33d S = H * state_.P * H.t() + R;
    cv::Matx33d S_inv = S.inv(cv::DECOMP_CHOLESKY);
    m.mahalanobis2 = (innovation.t() * S_inv * innovation)(0);
    if (m.mahalanobis2 > p_.gate_chi2) {
        m.accepted = false;
        return false;
    }
    cv::Matx<double, 4, 3> K = state_.P * H.t() * S_inv;
    state_.x += K * innovation;
    // Joseph form keeps P symmetric positive definite despite rounding.
    cv::Matx44d I_KH = cv::Matx44d::eye() - K * H;
    state_.P = I_KH * state_.P * I_KH.t() + K * R * K.t();
    m.accepted = true;
    last_update_[static_cast<int>(m.slot)] = time_s_;
    last_any_update_ = time_s_;
    return true;
}

bool LaneTracker::try_initialise(const std::vector<LaneMeasurement>& ms) {
    const LaneMeasurement* L = nullptr;
    const LaneMeasurement* R = nullptr;
    for (const LaneMeasurement& m : ms) {
        if (m.slot == LaneSlot::Left) L = &m;
        if (m.slot == LaneSlot::Right) R = &m;
    }
    if (!L || !R) {
        return false;
    }
    double w = L->coeffs[0] - R->coeffs[0];
    if (w < p_.min_width || w > p_.max_width || std::fabs(L->coeffs[1] - R->coeffs[1]) > 0.15) {
        return false;
    }
    // Start from a loose guess, then let the two measurements set the
    // covariance through the normal update equations.
    state_.valid = true;
    state_.x = cv::Vec4d(0.5 * (L->coeffs[0] + R->coeffs[0]), 0.5 * (L->coeffs[1] + R->coeffs[1]),
                         0.5 * (L->coeffs[2] + R->coeffs[2]), w);
    state_.P = cv::Matx44d::diag(cv::Vec4d(1.0, 0.01, 1e-4, 1.0));
    char buf[96];
    std::snprintf(buf, sizeof(buf), "initialised: width %.2f m", w);
    events_.push_back({time_s_, buf});
    return true;
}

void LaneTracker::handle_lane_change() {
    double w = state_.x[3];
    int shift = 0;
    if (state_.x[0] < -0.5 * w) {
        state_.x[0] += w;
        shift = 1;
        events_.push_back({time_s_, "lane change to the left"});
    } else if (state_.x[0] > 0.5 * w) {
        state_.x[0] -= w;
        shift = -1;
        events_.push_back({time_s_, "lane change to the right"});
    }
    if (shift == 1) {
        // Old left line is now the right line of the new lane.
        last_update_[static_cast<int>(LaneSlot::Right)] = last_update_[static_cast<int>(LaneSlot::Left)];
        last_update_[static_cast<int>(LaneSlot::Left)] = last_update_[static_cast<int>(LaneSlot::LeftOuter)];
    } else if (shift == -1) {
        last_update_[static_cast<int>(LaneSlot::Left)] = last_update_[static_cast<int>(LaneSlot::Right)];
        last_update_[static_cast<int>(LaneSlot::Right)] = last_update_[static_cast<int>(LaneSlot::RightOuter)];
    }
}

void LaneTracker::update(std::vector<LaneMeasurement>& ms) {
    if (!state_.valid && !try_initialise(ms)) {
        return;
    }
    // Data association. The detector's label is only a guess (it knows which
    // side of the car a line is on, not which road-model boundary it is).
    // Geometry first: a line can only be the boundary laterally nearest to it
    // in the middle of its observed range. Without this, a loosely-trusted
    // outer slot can "explain" an inner line by shrinking the lane width.
    // Then statistics: gate that one candidate by Mahalanobis distance, and
    // assign greedily, best first, each measurement and slot at most once.
    struct Pair { std::size_t m; LaneSlot slot; double d2; };
    std::vector<Pair> pairs;
    for (std::size_t i = 0; i < ms.size(); ++i) {
        LaneMeasurement& m = ms[i];
        m.accepted = false;
        double X = 0.5 * (m.x_near + m.x_far);
        double y = m.coeffs[0] + m.coeffs[1] * X + m.coeffs[2] * X * X;
        double best = 1e9;
        for (int s = 0; s < kLaneSlots; ++s) {
            double gap = std::fabs(y - state_.lateral(X, slot_width_multiple(static_cast<LaneSlot>(s))));
            if (gap < best) {
                best = gap;
                m.slot = static_cast<LaneSlot>(s);
            }
        }
        m.mahalanobis2 = mahalanobis2(m);
        if (m.mahalanobis2 <= p_.gate_chi2) {
            pairs.push_back({i, m.slot, m.mahalanobis2});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d2 < b.d2; });
    std::vector<bool> m_used(ms.size(), false);
    bool slot_used[kLaneSlots] = {false, false, false, false};
    int accepted = 0;
    for (const Pair& p : pairs) {
        if (m_used[p.m] || slot_used[static_cast<int>(p.slot)]) {
            continue;
        }
        m_used[p.m] = true;
        slot_used[static_cast<int>(p.slot)] = true;
        ms[p.m].slot = p.slot;
        accepted += apply(ms[p.m]) ? 1 : 0;
    }

    if (accepted == 0 && !ms.empty()) {
        ++consecutive_rejections_;
        if (consecutive_rejections_ >= p_.reinit_after_rejections) {
            reset("detections disagreed with the track for " +
                  std::to_string(consecutive_rejections_) + " frames");
            if (try_initialise(ms)) {
                for (LaneMeasurement& m : ms) {
                    apply(m);
                }
            }
            return;
        }
    } else if (accepted > 0) {
        consecutive_rejections_ = 0;
    }

    bool left_ok = false, right_ok = false, inner_rejected = false;
    for (const LaneMeasurement& m : ms) {
        left_ok = left_ok || (m.accepted && m.slot == LaneSlot::Left);
        right_ok = right_ok || (m.accepted && m.slot == LaneSlot::Right);
        inner_rejected = inner_rejected ||
                         (!m.accepted && (m.slot == LaneSlot::Left || m.slot == LaneSlot::Right));
    }
    split_frames_ = (inner_rejected && (left_ok != right_ok)) ? split_frames_ + 1 : 0;
    if (split_frames_ >= p_.reinit_after_split) {
        std::vector<LaneMeasurement> copy = ms;
        reset("one lane line disagreed with the track for " + std::to_string(split_frames_) + " frames");
        if (try_initialise(copy)) {
            for (LaneMeasurement& m : ms) {
                m.accepted = false;
                if (m.slot == LaneSlot::Left || m.slot == LaneSlot::Right) apply(m);
            }
            last_any_update_ = time_s_;
            status_ = TrackStatus::Locked;
        }
        return;
    }

    double w = state_.x[3];
    if (w < p_.min_width || w > p_.max_width) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "implausible lane width %.2f m", w);
        reset(buf);
        return;
    }
    handle_lane_change();

    bool left = last_update_[static_cast<int>(LaneSlot::Left)] == time_s_;
    bool right = last_update_[static_cast<int>(LaneSlot::Right)] == time_s_;
    if (left && right) {
        status_ = TrackStatus::Locked;
    } else if (last_any_update_ == time_s_) {
        status_ = TrackStatus::Partial;
    } else {
        status_ = TrackStatus::Coasting;
    }
}

double LaneTracker::time_since_update(LaneSlot s) const {
    return time_s_ - last_update_[static_cast<int>(s)];
}

double LaneTracker::confidence() const {
    if (!state_.valid) {
        return 0.0;
    }
    double s = state_.lateral_sigma(15.0, 0.0) / 0.25;
    return std::exp(-0.5 * s * s);
}

std::vector<TrackEvent> LaneTracker::take_events() {
    std::vector<TrackEvent> out;
    out.swap(events_);
    return out;
}
