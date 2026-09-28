#include "planning.hpp"
#include <cmath>

double distance_to_current_waypoint(const WorldState& world) {
    if (world.target_index >= world.path.size()) {
        return 0.0;
    }
    double dx = world.path[world.target_index].x - world.x;
    double dy = world.path[world.target_index].y - world.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool advance_if_reached(WorldState& world, const Config& cfg) {
    if (world.path_complete || world.target_index >= world.path.size()) {
        world.path_complete = true;
        return true;
    }
    if (distance_to_current_waypoint(world) < cfg.reach_radius) {
        world.target_index = world.target_index + 1;
        if (world.target_index >= world.path.size()) {
            world.path_complete = true;
        }
        return true;
    }
    return false;
}
