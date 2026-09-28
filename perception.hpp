#pragma once

#include "control.hpp"
#include "road.hpp"
#include <vector>

struct LaneLine {
    std::vector<Waypoint> points;
};

struct DetectedObject {
    double x;
    double y;
};

struct PerceptionOutput {
    std::vector<LaneLine> lanes;
    std::vector<DetectedObject> objects;
};

PerceptionOutput fake_perceive(const Road& road);
