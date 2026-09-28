#pragma once

#include <vector>
#include "world.hpp"
#include "road.hpp"
#include "perception.hpp"

struct CameraImage {
    int rows;
    int cols;
    std::vector<char> pixels;
};

char camera_at(const CameraImage& image, int row, int col);

CameraImage render_topdown(const WorldState& world, const Road& road,
                           int rows, int cols, double view_meters);

void print_camera(const CameraImage& image);

PerceptionOutput perceive_from_camera(const CameraImage& image,
                                      const WorldState& world,
                                      double view_meters);