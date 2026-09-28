#pragma once

class Vehicle {
    private:
        double x;
        double y;
        double velocity;
        double acceleration;
        double heading;
        double steering_angle = 0.0;
        double wheelbase = 2.5;
    
    public:
        Vehicle(double start_x, double start_y, double start_velocity,
                double start_acceleration, double start_heading);

        void step(double dt);

        double get_x() const;
        double get_y() const;
        double get_velocity() const;
        double get_heading() const;
        double get_wheelbase() const;
        void set_steering(double new_steering);
        void set_acceleration(double new_acceleration);
    };