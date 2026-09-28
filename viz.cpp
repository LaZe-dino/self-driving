#include "viz.hpp"
#include "control.hpp"
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>

void clear_screen() {
    std::cout << "\033[2J\033[H" << std::flush;
}

void wait_frame(int frame_ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(frame_ms));
}

static char car_glyph(double heading) {
    const double pi = 3.14159265358979323846;
    double a = wrap_angle(heading);
    if (a < -3.0 * pi / 4.0 || a >= 3.0 * pi / 4.0) {
        return '<';
    }
    if (a < -pi / 4.0) {
        return 'v';
    }
    if (a < pi / 4.0) {
        return '>';
    }
    return '^';
}

static bool near(double ax, double ay, double bx, double by, double radius_sq) {
    double dx = ax - bx;
    double dy = ay - by;
    return dx * dx + dy * dy < radius_sq;
}

static void stamp_segment(char& mark, char symbol, double cell_x, double cell_y,
                          double x0, double y0, double x1, double y1,
                          double radius_sq) {
    double dx = x1 - x0;
    double dy = y1 - y0;
    double length = std::sqrt(dx * dx + dy * dy);
    int samples = 1;
    if (length > 0.5) {
        samples = static_cast<int>(length / 0.5);
    }
    for (int s = 0; s <= samples; ++s) {
        double t = 0.0;
        if (samples > 0) {
            t = static_cast<double>(s) / samples;
        }
        double px = x0 + t * dx;
        double py = y0 + t * dy;
        if (near(px, py, cell_x, cell_y, radius_sq)) {
            mark = symbol;
        }
    }
}

void draw_world(const WorldState& world) {
    const int columns = 23;
    const int rows = 18;
    const double x_max = 45.0;
    const double y_max = 35.0;

    std::cout << "t=" << world.time << " s  v=" << world.velocity
              << " m/s  wp=" << world.target_index << "/" << world.path.size()
              << "\n";
    for (int row = rows - 1; row >= 0; --row) {
        for (int column = 0; column < columns; ++column) {
            double cell_x = (column + 0.5) * x_max / columns;
            double cell_y = (row + 0.5) * y_max / rows;
            char mark = '.';

            for (std::size_t lane = 0; lane < world.perception.lanes.size(); ++lane) {
                const std::vector<Waypoint>& pts = world.perception.lanes[lane].points;
                for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
                    stamp_segment(mark, '#', cell_x, cell_y,
                                  pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y,
                                  2.25);
                }
            }
            for (std::size_t i = 0; i < world.path.size(); ++i) {
                if (near(world.path[i].x, world.path[i].y, cell_x, cell_y, 4.0)) {
                    mark = 'W';
                }
            }
            for (std::size_t j = 0; j < world.trail.size(); ++j) {
                if (near(world.trail[j].x, world.trail[j].y, cell_x, cell_y, 4.0)) {
                    mark = '*';
                }
            }
            if (near(world.x, world.y, cell_x, cell_y, 4.0)) {
                mark = car_glyph(world.heading);
            }
            std::cout << mark;
        }
        std::cout << "\n";
    }
    std::cout << "W=waypoint  #=lane  *=trail  >^v<=car\n";
}
