#pragma once

#include "camera_model.hpp"
#include "lane_types.hpp"
#include <vector>

// Tracks the road-plane pitch relative to the camera. Real roads change grade
// and the car pitches over bumps; a fixed pitch makes parallel lane lines fan
// out in the bird's-eye view. The left and right lines of a lane are parallel
// on the road, so their image-space vanishing point gives the current pitch:
//   pitch = atan((cy - v_vp) / fy)       (independent of yaw)
class HorizonEstimator {
public:
    void configure(double calibrated_pitch_rad);
    // Uses this frame's left/right measurements (ground coordinates under the
    // current camera model). Returns true if the pitch was updated.
    bool update(const CameraModel& camera, const std::vector<LaneMeasurement>& ms, double dt_s);

    double pitch() const { return pitch_; }
    double calibrated_pitch() const { return calibrated_; }
    bool measured() const { return measured_; }
    double last_measurement() const { return last_meas_; }

private:
    double calibrated_ = 0.0;
    double pitch_ = 0.0;
    double last_meas_ = 0.0;
    bool measured_ = false;
    double time_constant_s_ = 0.5;
    double max_deviation_rad_ = 0.06;
    double max_jump_rad_ = 0.02;
};
