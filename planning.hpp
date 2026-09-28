#pragma once

#include "world.hpp"
#include "control.hpp"

double distance_to_current_waypoint(const WorldState& world);
bool advance_if_reached(WorldState& world, const Config& cfg);
