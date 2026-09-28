#include "object_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace {

const char* kCoco[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

}  // namespace

const char* coco_name(int id) {
    return id >= 0 && id < 80 ? kCoco[id] : "?";
}

bool is_vehicle_or_person(int id) {
    return id == 0 || id == 1 || id == 2 || id == 3 || id == 5 || id == 7;
}

bool touches_ground(int id) {
    return is_vehicle_or_person(id);
}

bool is_road_relevant(int id) {
    return is_vehicle_or_person(id) || id == 9 || id == 11;
}

bool ObjectDetector::load(const ObjectDetectorParams& p, std::string& error) {
    p_ = p;
    if (!std::filesystem::exists(p.model_path)) {
        error = "object model not found: " + p.model_path;
        return false;
    }
#if CV_VERSION_MAJOR >= 5
    net_ = cv::dnn::readNetFromONNX(p.model_path, p.engine);
#else
    net_ = cv::dnn::readNetFromONNX(p.model_path);
#endif
    if (net_.empty()) {
        error = "OpenCV DNN could not load " + p.model_path;
        return false;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    loaded_ = true;
    return true;
}

void ObjectDetector::detect(const cv::Mat& bgr, std::vector<Detection>& out) {
    out.clear();
    const int S = p_.input_size;
    // Letterbox: scale to fit S x S keeping aspect ratio, pad with grey (114)
    // at the bottom/right, exactly as YOLOX was trained.
    double r = std::min(static_cast<double>(S) / bgr.cols, static_cast<double>(S) / bgr.rows);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(cvRound(bgr.cols * r), cvRound(bgr.rows * r)), 0, 0, cv::INTER_LINEAR);
    letterbox_.create(S, S, CV_8UC3);
    letterbox_.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox_(cv::Rect(0, 0, resized.cols, resized.rows)));

    // YOLOX >= 0.1.1 takes raw 0-255 BGR pixels, no mean/std normalisation.
    cv::Mat blob = cv::dnn::blobFromImage(letterbox_, 1.0, cv::Size(), cv::Scalar(), false, false);
    net_.setInput(blob);
    cv::Mat pred = net_.forward();  // 1 x N x 85
    const int expected = (S / 8) * (S / 8) + (S / 16) * (S / 16) + (S / 32) * (S / 32);
    CV_Assert(pred.dims == 3 && pred.size[1] == expected && pred.size[2] == 85);
    const int N = pred.size[1];
    const int C = pred.size[2];
    cv::Mat rows(N, C, CV_32F, pred.ptr<float>());

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> classes;
    const int strides[] = {8, 16, 32};
    int i = 0;
    for (int stride : strides) {
        int g = S / stride;
        for (int gy = 0; gy < g; ++gy) {
            for (int gx = 0; gx < g; ++gx, ++i) {
                const float* p = rows.ptr<float>(i);
                float obj = p[4];
                if (obj < p_.score_threshold) continue;
                int best = 0;
                for (int c = 1; c < C - 5; ++c) {
                    if (p[5 + c] > p[5 + best]) best = c;
                }
                float score = obj * p[5 + best];
                if (score < p_.score_threshold || !is_road_relevant(best)) continue;
                // Decode: centre = (offset + cell) * stride, size = exp(log-size) * stride.
                double cx = (p[0] + gx) * stride, cy = (p[1] + gy) * stride;
                double w = std::exp(p[2]) * stride, h = std::exp(p[3]) * stride;
                boxes.push_back(cv::Rect2d((cx - 0.5 * w) / r, (cy - 0.5 * h) / r, w / r, h / r));
                scores.push_back(score);
                classes.push_back(best);
            }
        }
    }
    std::vector<cv::Rect> int_boxes;
    for (const cv::Rect2d& b : boxes) int_boxes.push_back(cv::Rect(b));
    std::vector<int> keep;
    // Non-maximum suppression per class: of overlapping boxes, keep the best.
    cv::dnn::NMSBoxesBatched(int_boxes, scores, classes, p_.score_threshold, p_.nms_threshold, keep);
    cv::Rect2d image(0, 0, bgr.cols, bgr.rows);
    for (int k : keep) {
        out.push_back({boxes[k] & image, classes[k], scores[k]});
    }
}
