#include "vehicle.hpp"
#include <cmath>

Vehicle::Vehicle(double start_x, double start_y, double start_velocity, double start_acceleration, double start_heading)
    : x(start_x),
      y(start_y),
      velocity(start_velocity),
      acceleration(start_acceleration),
      heading(start_heading) {
}

void Vehicle::set_acceleration(double new_acceleration) {
    acceleration = new_acceleration;
}

void Vehicle::set_steering(double new_steering) {
    steering_angle = new_steering;
}

void Vehicle::step(double dt) {
    velocity = velocity + acceleration * dt;
    if (velocity < 0.0) {
        velocity = 0.0;
    }
    heading = heading + (velocity / wheelbase) * std::tan(steering_angle) * dt;
    x = x + velocity * std::cos(heading) * dt;
    y = y + velocity * std::sin(heading) * dt;
}

double Vehicle::get_x() const {
    return x;
}

double Vehicle::get_y() const {
    return y;
}

double Vehicle::get_velocity() const {
    return velocity;
}

double Vehicle::get_heading() const {
    return heading;
}

double Vehicle::get_wheelbase() const {
    return wheelbase;
}