#include "camera.hpp"
#include <cmath>
#include <iostream>
#include "perception.hpp"

char camera_at(const CameraImage& image, int row, int col) {
    if (row < 0 || col < 0 || row >= image.rows || col >= image.cols) {
        return ' ';
    }
    return image.pixels[row * image.cols + col];
}

static bool near_xy(double ax, double ay, double bx, double by, double radius_sq) {
    double dx = ax - bx;
    double dy = ay - by;
    return dx * dx + dy * dy <= radius_sq;
}

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

static void world_of_cell(const WorldState& world, int row, int col,
                          int rows, int cols, double view_meters,
                          double& out_x, double& out_y) {
    double m_per = view_meters / cols;
    out_x = world.x + (col + 0.5 - cols / 2.0) * m_per;
    out_y = world.y + (rows / 2.0 - (row + 0.5)) * m_per; 
}

CameraImage render_topdown(const WorldState& world, const Road& road,
                           int rows, int cols, double view_meters) {
    CameraImage image;
    image.rows = rows;
    image.cols = cols;
    image.pixels.assign(static_cast<std::size_t>(rows * cols), '.');

    std::vector<Waypoint> left;
    std::vector<Waypoint> right;
    offset_centerline(road.centerline, road.half_width, left);
    offset_centerline(road.centerline, -road.half_width, right);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double wx, wy;
            world_of_cell(world, row, col, rows, cols, view_meters, wx, wy);
            char mark = '.';
            for (std::size_t i = 0; i < left.size(); ++i) {
                if (near_xy(left[i].x, left[i].y, wx, wy, 4.0)) {
                    mark = '#';
                }
            }
            for (std::size_t i = 0; i < right.size(); ++i) {
                if (near_xy(right[i].x, right[i].y, wx, wy, 4.0)) {
                    mark = '+';
                }
            }
            for (std::size_t i = 0; i < world.path.size(); ++i) {
                if (near_xy(world.path[i].x, world.path[i].y, wx, wy, 4.0)) {
                    mark = 'W';
                }
            }
            if (near_xy(world.x, world.y, wx, wy, 4.0)) {
                mark = '@';
            }
            image.pixels[row * cols + col] = mark;
        }
    }
    return image;
}

void print_camera(const CameraImage& image) {
    std::cout << "camera (top-down, @ = car, #/+ = lanes)\n";
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            std::cout << camera_at(image, row, col);
        }
        std::cout << "\n";
    }
}
PerceptionOutput perceive_from_camera(const CameraImage& image,
                                      const WorldState& world,
                                      double view_meters) {
    PerceptionOutput out;
    LaneLine left;
    LaneLine right;
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            char p = camera_at(image, row, col);
            double wx, wy;
            world_of_cell(world, row, col, image.rows, image.cols,
                          view_meters, wx, wy);
            if (p == '#') {
                left.points.push_back({wx, wy});
            }
            if (p == '+') {
                right.points.push_back({wx, wy});
            }
        }
    }
    out.lanes.push_back(left);
    out.lanes.push_back(right);
    return out;
}