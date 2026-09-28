#pragma once

#include <vector>

struct Waypoint {
    double x;
    double y;
};

struct Config {
    double dt = 0.1;
    int max_steps = 400;
    double reach_radius = 5.0;
    double slow_zone = 15.0;
    double cruise_speed = 12.0;
    double slow_speed = 4.0;
    double accel_gain = 0.8;
    double lookahead = 8.0;
    double max_steering = 0.4;
    bool animate = true;
    int frame_ms = 50;
};

double wrap_angle(double angle);

double pure_pursuit_steering(double x, double y, double heading,
                             double target_x, double target_y,
                             double wheelbase, const Config& cfg);

double longitudinal_acceleration(double velocity, double distance,
                                 const Config& cfg);

Waypoint lookahead_waypoint(const std::vector<Waypoint>& path,
                            std::size_t target_index,
                            double x, double y,
                            double lookahead);
