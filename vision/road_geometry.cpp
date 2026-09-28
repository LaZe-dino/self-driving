#include "road_geometry.hpp"
#include "control.hpp"
#include "vehicle.hpp"
#include <algorithm>
#include <cmath>

RoadGeometry compute_road_geometry(const RoadState& road, const CameraModel& camera) {
    RoadGeometry g;
    if (!road.valid) {
        return g;
    }
    double yc = road.x[0], c1 = road.x[1], c2 = road.x[2];
    g.valid = true;
    g.lateral_offset_m = -yc;
    g.lateral_offset_sigma_m = std::sqrt(road.P(0, 0));
    g.heading_error_rad = std::atan(c1);
    // Curvature of y(x): k = y'' / (1 + y'^2)^(3/2), with y' = c1, y'' = 2 c2 at x = 0.
    double denom = std::pow(1.0 + c1 * c1, 1.5);
    g.curvature_1pm = 2.0 * c2 / denom;
    g.curvature_sigma_1pm = 2.0 * std::sqrt(road.P(2, 2)) / denom;
    g.lane_width_m = road.x[3];
    g.lane_width_sigma_m = std::sqrt(road.P(3, 3));
    g.lane_vanishing_px = camera.vanishing_point(1.0, c1);
    double w = camera.image_size.width;
    g.horizon_left_px = {0.0, camera.horizon_row_at(0.0)};
    g.horizon_right_px = {w, camera.horizon_row_at(w)};
    return g;
}

PathPrediction predict_path(const RoadState& road, double speed_mps, bool speed_measured,
                            double max_range_m) {
    PathPrediction path;
    if (!road.valid) {
        return path;
    }
    Config cfg;
    path.speed_mps = speed_mps;
    path.speed_measured = speed_measured;
    // Look further ahead at speed: roughly 0.8 s of travel, 8-20 m.
    path.lookahead_m = std::clamp(0.8 * speed_mps, 8.0, 20.0);

    Vehicle car(0.0, 0.0, std::max(speed_mps, 1.0), 0.0, 0.0);
    const double ds = 0.5;
    path.points.push_back({0.0, 0.0});
    for (int i = 0; i < 200; ++i) {
        // Find the lane-centre point one lookahead distance from the car.
        double tx = car.get_x() + path.lookahead_m;
        double ty = road.lateral(tx, 0.0);
        for (int it = 0; it < 3; ++it) {
            double dy = ty - car.get_y();
            double dx_needed = std::sqrt(std::max(0.0, path.lookahead_m * path.lookahead_m - dy * dy));
            tx = car.get_x() + dx_needed;
            ty = road.lateral(tx, 0.0);
        }
        if (tx > max_range_m) {
            break;
        }
        double steer = pure_pursuit_steering(car.get_x(), car.get_y(), car.get_heading(), tx, ty,
                                             car.get_wheelbase(), cfg);
        if (i == 0) {
            path.steering_rad = steer;
        }
        car.set_steering(steer);
        car.step(ds / car.get_velocity());
        path.points.push_back({car.get_x(), car.get_y()});
    }
    path.valid = path.points.size() > 1;
    return path;
}
