#include <iostream>
#include <string>
#include "vehicle.hpp"
#include "control.hpp"
#include "world.hpp"
#include "road.hpp"
#include "planning.hpp"
#include "perception.hpp"
#include "viz.hpp"
#include <vector>
#include "camera.hpp"
#include "webcam.hpp"

int main(int argc, char** argv) {
    bool want_webcam = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--webcam") {
            want_webcam = true;
        }
    }

    if (want_webcam) {
        if (!atlas_webcam_available()) {
            atlas_webcam_start();
            return 1;
        }
        if (!atlas_webcam_start()) {
            return 1;
        }
    }

    Config cfg;
    if (want_webcam) {
        cfg.animate = false;
    }
    Vehicle car(0.0, 0.0, 0.0, 0.0, 0.0);

    WorldState world;
    world.path = {
        {20.0, 10.0},
        {40.0, 10.0},
        {40.0, 30.0},
    };

    Road road;
    road.centerline = world.path;
    road.half_width = 2.0;

    std::vector<double> reached_at;

    CameraImage cam;
    const double view_meters = 24.0;

    for (int step = 0; step < cfg.max_steps; ++step) {
        world.x = car.get_x();
        world.y = car.get_y();
        world.heading = car.get_heading();
        world.velocity = car.get_velocity();
        world.time = step * cfg.dt;
        cam = render_topdown(world, road, 9, 19, view_meters);
        world.perception = perceive_from_camera(cam, world, view_meters);

        if (advance_if_reached(world, cfg)) {
            reached_at.push_back(world.time);
            if (world.path_complete) {
                break;
            }
            if (want_webcam && !atlas_webcam_pump()) {
                break;
            }
            continue;
        }

        double distance = distance_to_current_waypoint(world);
        car.set_acceleration(longitudinal_acceleration(world.velocity, distance, cfg));

        Waypoint aim = lookahead_waypoint(
            world.path, world.target_index, world.x, world.y, cfg.lookahead);
        double steering = pure_pursuit_steering(
            world.x, world.y, world.heading, aim.x, aim.y,
            car.get_wheelbase(), cfg);
        car.set_steering(steering);
        car.step(cfg.dt);

        world.x = car.get_x();
        world.y = car.get_y();
        world.heading = car.get_heading();
        world.velocity = car.get_velocity();
        world.time = (step + 1) * cfg.dt;
        world.trail.push_back({world.x, world.y});

        if (cfg.animate) {
            clear_screen();
            draw_world(world);
            print_camera(cam);
            wait_frame(cfg.frame_ms);
        } else {
            std::cout << "t=" << world.time
                      << " s, x=" << world.x
                      << " m, y=" << world.y
                      << " m, heading=" << world.heading
                      << " rad, v=" << world.velocity
                      << " m/s, points=" << world.target_index
                      << ", distance=" << distance << " m\n";
        }

        if (want_webcam && !atlas_webcam_pump()) {
            break;
        }
    }

    if (!cfg.animate) {
        draw_world(world);
        print_camera(cam);
    } else {
        clear_screen();
        draw_world(world);
        print_camera(cam);
    }

    for (std::size_t i = 0; i < reached_at.size(); ++i) {
        std::cout << "Reached waypoint " << (i + 1)
                  << " at t=" << reached_at[i] << " s\n";
    }
    if (world.path_complete) {
        std::cout << "Path complete\n";
    }

    if (want_webcam) {
        atlas_webcam_stop();
    }

    return 0;
}
