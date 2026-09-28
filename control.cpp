#include "control.hpp"
#include <cmath>

double wrap_angle(double angle) {
    const double pi = 3.14159265358979323846;
    while (angle > pi) {
        angle = angle - 2 * pi;
    }
    while (angle < -pi) {
        angle = angle + 2.0 * pi;
    }
    return angle;
}

double pure_pursuit_steering(double x, double y, double heading,
                             double target_x, double target_y,
                             double wheelbase, const Config& cfg) {
    double dx = target_x - x;
    double dy = target_y - y;
    double lookahead_dist = std::sqrt(dx * dx + dy * dy);
    if (lookahead_dist < 1e-6) {
        return 0.0;
    }

    double alpha = wrap_angle(std::atan2(dy, dx) - heading);
    double steering = std::atan(2.0 * wheelbase * std::sin(alpha) / lookahead_dist);

    if (steering > cfg.max_steering) {
        steering = cfg.max_steering;
    }
    if (steering < -cfg.max_steering) {
        steering = -cfg.max_steering;
    }
    return steering;
}

Waypoint lookahead_waypoint(const std::vector<Waypoint>& path,
                            std::size_t target_index,
                            double x, double y,
                            double lookahead) {
    for (std::size_t i = target_index; i < path.size(); ++i) {
        double dx = path[i].x - x;
        double dy = path[i].y - y;
        double distance = std::sqrt(dx * dx + dy * dy);
        if (distance >= lookahead || i + 1 == path.size()) {
            return path[i];
        }
    }
    return path[target_index];
}

double longitudinal_acceleration(double velocity, double distance,
                                 const Config& cfg) {
    double target_speed = cfg.cruise_speed;
    if (distance < cfg.slow_zone) {
        target_speed = cfg.slow_speed;
    }
    return cfg.accel_gain * (target_speed - velocity);
}
