#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

struct Detection {
    cv::Rect2d box;  // undistorted image pixels
    int class_id = -1;
    float score = 0.0f;
};

const char* coco_name(int class_id);
// Classes that matter for driving: people, bikes, vehicles, lights, signs.
bool is_road_relevant(int class_id);
// Objects that stand on the road, so the bottom of the box is a ground contact.
bool touches_ground(int class_id);
bool is_vehicle_or_person(int class_id);

struct ObjectDetectorParams {
    std::string model_path = "data/models/yolox_tiny.onnx";
    int input_size = 416;
    float score_threshold = 0.35f;
    float nms_threshold = 0.45f;
#if CV_VERSION_MAJOR >= 5
    // With OpenCV 5.0.0 on Windows the new DNN graph engine crashed (heap
    // corruption in worker threads) when running YOLOX. The classic 4.x-style
    // engine is used instead; whether it avoids the crash is unverified.
    int engine = cv::dnn::ENGINE_CLASSIC;
#endif
};

// YOLOX (Megvii, Apache-2.0) single-stage detector run through OpenCV DNN.
// The network predicts, for every cell of 3 grids (strides 8, 16, 32), a box
// offset, an objectness probability and 80 COCO class probabilities.
class ObjectDetector {
public:
    bool load(const ObjectDetectorParams& p, std::string& error);
    void detect(const cv::Mat& bgr, std::vector<Detection>& out);
    bool loaded() const { return loaded_; }
    const ObjectDetectorParams& params() const { return p_; }

private:
    ObjectDetectorParams p_;
    cv::dnn::Net net_;
    bool loaded_ = false;
    cv::Mat letterbox_;
};
