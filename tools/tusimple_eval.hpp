#pragma once

#include "lane_rasterize.hpp"
#include "tusimple.hpp"
#include "../vision/lane_types.hpp"
#include "../vision/object_detector.hpp"
#include <string>
#include <vector>

struct TuSimpleEvalOptions {
    std::string root;                          // folder that contains clips/
    std::vector<std::string> label_files;
    std::vector<std::string> fit_label_files;  // default: label_files
    std::string camera_model_path;
    double fov_deg = 60.0;
    bool fit_mount = false;                    // force, even with --camera-model
    bool no_fit_mount = false;                 // keep the level 1.4 m default mount
    int fit_images = 300;
    double lane_width_m = 3.7;
    std::string save_camera_path;
    long long limit = -1;
    std::string out_json;
    std::string predictions_path;              // official submission format
    bool objects = false;
    ObjectDetectorParams object_params;
    RasterizeOptions raster{0.0, 40.0, 0.4};
    bool all_slots = false;                    // outer lanes even if never seen
    int min_points = 2;
    double fps = 20.0;
};

int run_tusimple_eval(const TuSimpleEvalOptions& o);

// TuSimple lanes predicted from the tracked road: one list per enabled slot
// that has at least min_points drawn points, x in original image pixels.
std::vector<std::vector<int>> predict_tusimple_lanes(const RoadState& road, const CameraModel& camera,
                                                     const std::vector<int>& rows, const RasterizeOptions& opt,
                                                     const bool use_slot[kLaneSlots], int min_points);

// Lateral position Y (m, left positive) of a labelled lane X metres ahead by
// flat-ground back-projection. False if the lane does not span X.
bool label_lateral_at(const std::vector<int>& xs, const std::vector<int>& rows, const CameraModel& camera,
                      double X, double& Y);
