#include "object_tracker.hpp"
#include <algorithm>
#include <cmath>

namespace {

double iou(const cv::Rect2d& a, const cv::Rect2d& b) {
    double inter = (a & b).area();
    double uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0;
}

const double kBoxBottomSigmaPx = 3.0;
const double kAccelSigma = 3.0;  // m/s^2, relative acceleration process noise

}  // namespace

void ObjectTracker::update_ground(TrackedObject& t, const CameraModel& cam, double dt, bool measured) {
    if (!touches_ground(t.class_id)) {
        t.ground_valid = false;
        return;
    }
    if (t.ground_valid) {
        // Constant velocity: X += vX dt. Unknown acceleration enters as noise
        // through G = [dt^2/2, dt] on each axis.
        cv::Matx44d F(1, 0, dt, 0,
                      0, 1, 0, dt,
                      0, 0, 1, 0,
                      0, 0, 0, 1);
        double q = kAccelSigma * kAccelSigma;
        double a = dt * dt * dt * dt / 4, b = dt * dt * dt / 2, c = dt * dt;
        cv::Matx44d Q(a * q, 0, b * q, 0,
                      0, a * q, 0, b * q,
                      b * q, 0, c * q, 0,
                      0, b * q, 0, c * q);
        t.g = F * t.g;
        t.P = F * t.P * F.t() + Q;
    }
    if (!measured) {
        return;
    }
    // Ground contact = bottom centre of the box.
    cv::Point2d foot(t.box.x + 0.5 * t.box.width, t.box.y + t.box.height);
    double X, Y;
    if (!cam.image_to_ground(foot, X, Y) || X > 120.0) {
        return;
    }
    // Range from the ground contact: v - v_horizon ~ f h / X, so a small
    // error dv in the box bottom gives dX ~ X^2 / (f h) dv.
    double sX = std::max(0.3, X * X / (cam.K(1, 1) * cam.height_m) * kBoxBottomSigmaPx);
    double sY = 0.3 + X / cam.K(0, 0) * kBoxBottomSigmaPx;
    if (!t.ground_valid) {
        t.g = cv::Vec4d(X, Y, 0, 0);
        t.P = cv::Matx44d::diag(cv::Vec4d(sX * sX, sY * sY, 10.0 * 10.0, 3.0 * 3.0));
        t.ground_valid = true;
        return;
    }
    cv::Matx<double, 2, 4> H(1, 0, 0, 0,
                             0, 1, 0, 0);
    cv::Matx22d R(sX * sX, 0, 0, sY * sY);
    cv::Vec2d z(X, Y);
    cv::Vec2d nu = z - H * t.g;
    cv::Matx22d S = H * t.P * H.t() + R;
    cv::Matx<double, 4, 2> K = t.P * H.t() * S.inv();
    t.g += K * nu;
    cv::Matx44d I_KH = cv::Matx44d::eye() - K * H;
    t.P = I_KH * t.P * I_KH.t() + K * R * K.t();
}

void ObjectTracker::update(const std::vector<Detection>& dets, const CameraModel& cam, const RoadState& road,
                           double dt) {
    dt = std::clamp(dt, 1e-3, 0.5);
    // Predict each box forward with its image velocity, then associate by
    // overlap (IoU), best pairs first.
    for (TrackedObject& t : tracks_) {
        t.box.x += t.box_velocity.x * dt;
        t.box.y += t.box_velocity.y * dt;
        t.matched = false;
    }
    struct Pair { std::size_t t, d; double iou; };
    std::vector<Pair> pairs;
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
        for (std::size_t di = 0; di < dets.size(); ++di) {
            if (dets[di].class_id != tracks_[ti].class_id) continue;
            double o = iou(tracks_[ti].box, dets[di].box);
            if (o > 0.25) pairs.push_back({ti, di, o});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.iou > b.iou; });
    std::vector<bool> det_used(dets.size(), false);
    for (const Pair& p : pairs) {
        TrackedObject& t = tracks_[p.t];
        if (t.matched || det_used[p.d]) continue;
        t.matched = det_used[p.d] = true;
        const Detection& d = dets[p.d];
        cv::Point2d c_old(t.box.x + t.box.width / 2, t.box.y + t.box.height / 2);
        cv::Point2d c_new(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
        t.box_velocity = 0.7 * t.box_velocity + 0.3 * ((c_new - c_old) * (1.0 / dt));
        // Blend toward the detection: boxes from a single frame jitter.
        t.box = cv::Rect2d(0.4 * t.box.x + 0.6 * d.box.x, 0.4 * t.box.y + 0.6 * d.box.y,
                           0.4 * t.box.width + 0.6 * d.box.width, 0.4 * t.box.height + 0.6 * d.box.height);
        t.score = d.score;
        ++t.hits;
        t.misses = 0;
        t.confirmed = t.confirmed || t.hits >= 3;
    }
    for (TrackedObject& t : tracks_) {
        if (!t.matched) ++t.misses;
        update_ground(t, cam, dt, t.matched);
    }
    // Forget tracks unseen for ~0.4 s (or 2 frames if never confirmed).
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const TrackedObject& t) { return t.misses > (t.confirmed ? 10 : 2); }),
                  tracks_.end());
    for (std::size_t di = 0; di < dets.size(); ++di) {
        if (det_used[di]) continue;
        TrackedObject t;
        t.id = next_id_++;
        t.class_id = dets[di].class_id;
        t.score = dets[di].score;
        t.box = dets[di].box;
        t.box_velocity = {0, 0};
        t.hits = 1;
        t.matched = true;
        update_ground(t, cam, dt, true);
        tracks_.push_back(t);
    }
    for (TrackedObject& t : tracks_) {
        t.in_ego_lane = false;
        t.ttc_s = -1.0;
        if (!t.ground_valid || !road.valid) continue;
        double X = t.g[0], Y = t.g[1];
        t.in_ego_lane = Y < road.lateral(X, 0.5) && Y > road.lateral(X, -0.5);
        // Time to collision: distance / closing speed, if closing.
        if (t.in_ego_lane && t.g[2] < -0.5) t.ttc_s = X / -t.g[2];
    }
}
