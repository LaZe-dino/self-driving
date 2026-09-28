#include "horizon_estimator.hpp"
#include <algorithm>
#include <cmath>

void HorizonEstimator::configure(double calibrated_pitch_rad) {
    calibrated_ = calibrated_pitch_rad;
    pitch_ = calibrated_pitch_rad;
    measured_ = false;
}

namespace {

// Image line through the projections of two ground points on the lane's
// tangent at X1. Re-projecting through the same homography recovers the
// original pixels, so the result does not depend on the pitch assumed when
// the lane was measured.
bool tangent_in_image(const CameraModel& cam, const LaneMeasurement& m, cv::Point2d& p, cv::Point2d& d) {
    double X1 = std::max(m.x_near, 8.0);
    double X2 = std::min(m.x_far, X1 + 12.0);
    if (X2 - X1 < 8.0) {
        return false;
    }
    double y1 = m.coeffs[0] + m.coeffs[1] * X1 + m.coeffs[2] * X1 * X1;
    double slope = m.coeffs[1] + 2.0 * m.coeffs[2] * X1;
    cv::Point2d a = cam.ground_to_image(X1, y1);
    cv::Point2d b = cam.ground_to_image(X2, y1 + slope * (X2 - X1));
    p = a;
    d = b - a;
    return std::hypot(d.x, d.y) > 5.0;
}

}  // namespace

bool HorizonEstimator::update(const CameraModel& cam, const std::vector<LaneMeasurement>& ms, double dt_s) {
    measured_ = false;
    const LaneMeasurement* L = nullptr;
    const LaneMeasurement* R = nullptr;
    // Only lines the tracker accepted: a car edge mistaken for a lane line
    // would otherwise tilt the horizon and corrupt every later measurement.
    for (const LaneMeasurement& m : ms) {
        if (m.accepted && m.slot == LaneSlot::Left) L = &m;
        if (m.accepted && m.slot == LaneSlot::Right) R = &m;
    }
    cv::Point2d pl, dl, pr, dr;
    if (!L || !R || !tangent_in_image(cam, *L, pl, dl) || !tangent_in_image(cam, *R, pr, dr)) {
        return false;
    }
    double den = dl.x * dr.y - dl.y * dr.x;
    if (std::fabs(den) < 1e-6) {
        return false;
    }
    cv::Point2d w = pr - pl;
    double s = (w.x * dr.y - w.y * dr.x) / den;
    cv::Point2d vp = pl + s * dl;

    double meas = std::atan((cam.K(1, 2) - vp.y) / cam.K(1, 1));
    last_meas_ = meas;
    // A single bad pair of lines must not throw the horizon around: limit the
    // step, then low-pass with time constant ~0.5 s, within a bounded range.
    double step = std::clamp(meas - pitch_, -max_jump_rad_, max_jump_rad_);
    double alpha = std::clamp(dt_s / time_constant_s_, 0.0, 1.0);
    pitch_ = std::clamp(pitch_ + alpha * step, calibrated_ - max_deviation_rad_, calibrated_ + max_deviation_rad_);
    measured_ = true;
    return true;
}
