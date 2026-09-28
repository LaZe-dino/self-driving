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
#ifdef ATLAS_HAS_OPENCV
#include "vision/app.hpp"
#endif

static int run_waypoint_sim() {
    Config cfg;
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
        }
    }

    draw_world(world);
    print_camera(cam);
    for (std::size_t i = 0; i < reached_at.size(); ++i) {
        std::cout << "Reached waypoint " << (i + 1)
                  << " at t=" << reached_at[i] << " s\n";
    }
    if (world.path_complete) {
        std::cout << "Path complete\n";
    }
    return 0;
}

int main(int argc, char** argv) {
#ifdef ATLAS_HAS_OPENCV
    int vision_result = atlas_vision_main(argc, argv);
    if (vision_result >= 0) {
        return vision_result;
    }
#else
    if (argc > 1) {
        std::cerr << "This build has no OpenCV, so Atlas Vision is unavailable.\n"
                  << "Windows MSYS2 UCRT64: pacman -S mingw-w64-ucrt-x86_64-opencv mingw-w64-ucrt-x86_64-qt6-base\n"
                  << "Then delete build/, reconfigure CMake and rebuild.\n";
        return 1;
    }
#endif
    return run_waypoint_sim();
}
