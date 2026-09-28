#pragma once

#include "control.hpp"
#include "perception.hpp"
#include <vector>

struct WorldState {
    double x = 0.0;
    double y = 0.0;
    double heading = 0.0;
    double velocity = 0.0;
    double time = 0.0;
    std::vector<Waypoint> path;
    std::size_t target_index = 0;
    std::vector<Waypoint> trail;
    bool path_complete = false;
    PerceptionOutput perception;
};
