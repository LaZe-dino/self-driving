#include "test_main.hpp"
#include "../vision/lane_tracker.hpp"

static LaneMeasurement meas(LaneSlot slot, double c0, double c1 = 0.0, double c2 = 0.0) {
    LaneMeasurement m;
    m.slot = slot;
    m.coeffs = cv::Vec3d(c0, c1, c2);
    m.cov = cv::Matx33d::diag(cv::Vec3d(0.05 * 0.05, 0.005 * 0.005, 0.0003 * 0.0003));
    return m;
}

static void step(LaneTracker& t, double time, std::vector<LaneMeasurement> ms) {
    t.predict(time, 0.0, 0.0);
    t.update(ms);
}

TEST(tracker_initialises_and_locks) {
    LaneTracker t;
    for (int i = 0; i < 10; ++i) {
        step(t, i * 0.04, {meas(LaneSlot::Left, 1.9), meas(LaneSlot::Right, -1.7)});
    }
    CHECK(t.status() == TrackStatus::Locked);
    CHECK_NEAR(t.state().x[3], 3.6, 0.02);
    CHECK_NEAR(t.state().x[0], 0.1, 0.02);
    CHECK(t.confidence() > 0.8);
}

TEST(tracker_holds_lane_when_one_line_disappears) {
    LaneTracker t;
    double time = 0.0;
    for (int i = 0; i < 25; ++i, time += 0.04) {
        step(t, time, {meas(LaneSlot::Left, 1.8), meas(LaneSlot::Right, -1.8)});
    }
    double sigma_before = t.state().lateral_sigma(15.0, -0.5);
    for (int i = 0; i < 25; ++i, time += 0.04) {
        step(t, time, {meas(LaneSlot::Left, 1.8)});
    }
    CHECK(t.status() == TrackStatus::Partial);
    // The right line is still estimated from the left line plus the width.
    CHECK_NEAR(t.state().lateral(0.0, -0.5), -1.8, 0.05);
    CHECK(t.state().lateral_sigma(15.0, -0.5) > sigma_before);
}

TEST(tracker_coasts_then_resets_when_all_lines_vanish) {
    LaneTracker t;
    double time = 0.0;
    for (int i = 0; i < 25; ++i, time += 0.04) {
        step(t, time, {meas(LaneSlot::Left, 1.8), meas(LaneSlot::Right, -1.8)});
    }
    double conf_locked = t.confidence();
    step(t, time + 0.5, {});
    CHECK(t.status() == TrackStatus::Coasting);
    CHECK(t.state().valid);
    CHECK(t.confidence() < conf_locked);
    step(t, time + 1.0, {});
    step(t, time + 1.6, {});
    CHECK(!t.state().valid);
    CHECK(t.status() == TrackStatus::Searching);
}

TEST(tracker_gates_a_sudden_jump) {
    LaneTracker t;
    double time = 0.0;
    for (int i = 0; i < 25; ++i, time += 0.04) {
        step(t, time, {meas(LaneSlot::Left, 1.8), meas(LaneSlot::Right, -1.8)});
    }
    std::vector<LaneMeasurement> ms = {meas(LaneSlot::Left, 3.2), meas(LaneSlot::Right, -1.8)};
    t.predict(time, 0.0, 0.0);
    t.update(ms);
    CHECK(!ms[0].accepted);
    CHECK(ms[1].accepted);
    CHECK_NEAR(t.state().lateral(0.0, 0.5), 1.8, 0.05);
}

TEST(tracker_follows_a_gradual_lane_change) {
    LaneTracker t;
    double time = 0.0;
    double drift = 0.0;
    // Car drifts left at 0.5 m/s; the lines move right in the car frame.
    for (int i = 0; i < 150; ++i, time += 0.04, drift += 0.02) {
        std::vector<LaneMeasurement> ms;
        const double lines[] = {-5.4, -1.8, 1.8, 5.4};
        for (double y : lines) {
            double c0 = y - drift;
            // The detector labels lines by where they are relative to the car.
            if (c0 > 0 && c0 < 3.6) ms.push_back(meas(LaneSlot::Left, c0));
            if (c0 < 0 && c0 > -3.6) ms.push_back(meas(LaneSlot::Right, c0));
        }
        step(t, time, ms);
    }
    CHECK(t.state().valid);
    // After 3 m of drift the car is in the next lane, near its centre.
    CHECK_NEAR(t.state().x[0], 0.6, 0.15);
    bool saw_change = false;
    for (const TrackEvent& e : t.take_events()) {
        saw_change = saw_change || e.what == "lane change to the left";
    }
    CHECK(saw_change);
}
