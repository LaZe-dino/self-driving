#include "test_main.hpp"
#include "../vision/horizon_estimator.hpp"
#include "../vision/lane_detector.hpp"

// Sees ground line Y = y0 through `truth`, back-projects the pixels through
// `assumed` and fits the polynomial the detector would report.
static LaneMeasurement observe(const CameraModel& truth, const CameraModel& assumed, double y0, LaneSlot slot) {
    std::vector<cv::Point2d> pts;
    for (double X = 7.0; X < 40.0; X += 0.25) {
        cv::Point2d px = truth.ground_to_image(X, y0);
        double Xa, Ya;
        if (assumed.image_to_ground(px, Xa, Ya) && Xa < 60.0) pts.push_back({Xa, Ya});
    }
    LaneMeasurement m;
    m.slot = slot;
    m.accepted = true;
    double rms;
    fit_lane_polynomial(pts, 0.01, 10.0, m.coeffs, m.cov, rms, nullptr);
    m.x_near = pts.front().x;
    m.x_far = pts.back().x;
    return m;
}

TEST(horizon_estimator_recovers_pitch_from_parallel_lines) {
    CameraModel truth = default_camera_model(cv::Size(1280, 720), 60.0);
    truth.height_m = 1.2;
    truth.pitch_rad = 0.03;
    truth.update();
    CameraModel assumed = truth;
    assumed.pitch_rad = 0.0;
    assumed.update();

    // With the wrong pitch the two lines are not parallel in the ground view.
    LaneMeasurement l0 = observe(truth, assumed, 1.8, LaneSlot::Left);
    LaneMeasurement r0 = observe(truth, assumed, -1.8, LaneSlot::Right);
    CHECK(std::fabs(l0.coeffs[1] - r0.coeffs[1]) > 0.05);

    HorizonEstimator h;
    h.configure(0.0);
    for (int i = 0; i < 100; ++i) {
        std::vector<LaneMeasurement> ms = {observe(truth, assumed, 1.8, LaneSlot::Left),
                                           observe(truth, assumed, -1.8, LaneSlot::Right)};
        CHECK(h.update(assumed, ms, 0.04));
        assumed.pitch_rad = h.pitch();
        assumed.update();
    }
    CHECK_NEAR(h.pitch(), 0.03, 0.001);
    LaneMeasurement l1 = observe(truth, assumed, 1.8, LaneSlot::Left);
    LaneMeasurement r1 = observe(truth, assumed, -1.8, LaneSlot::Right);
    CHECK_NEAR(l1.coeffs[1] - r1.coeffs[1], 0.0, 0.005);
}
