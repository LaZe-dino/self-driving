#include "perception.hpp"
#include <cmath>

static void offset_centerline(const std::vector<Waypoint>& center, double offset,
                              std::vector<Waypoint>& out) {
    for (std::size_t i = 0; i + 1 < center.size(); ++i) {
        double dx = center[i + 1].x - center[i].x;
        double dy = center[i + 1].y - center[i].y;
        double length = std::sqrt(dx * dx + dy * dy);
        if (length < 1e-9) {
            continue;
        }
        dx = dx / length;
        dy = dy / length;
        double px = -dy * offset;
        double py = dx * offset;
        if (out.empty()) {
            out.push_back({center[i].x + px, center[i].y + py});
        }
        out.push_back({center[i + 1].x + px, center[i + 1].y + py});
    }
}

PerceptionOutput fake_perceive(const Road& road) {
    PerceptionOutput out;
    LaneLine left;
    LaneLine right;
    offset_centerline(road.centerline, road.half_width, left.points);
    offset_centerline(road.centerline, -road.half_width, right.points);
    out.lanes.push_back(left);
    out.lanes.push_back(right);
    return out;
}
