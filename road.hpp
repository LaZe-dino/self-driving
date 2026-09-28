#pragma once

#include "control.hpp"
#include <vector>

struct Road {
    std::vector<Waypoint> centerline;
    double half_width = 2.0;
};
