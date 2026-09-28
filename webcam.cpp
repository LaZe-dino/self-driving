#include "webcam.hpp"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef ATLAS_HAS_OPENCV
#include <opencv2/opencv.hpp>

static cv::VideoCapture g_cap;

static double line_slope(const cv::Vec4i& line) {
    double dx = static_cast<double>(line[2] - line[0]);
    double dy = static_cast<double>(line[3] - line[1]);
    if (std::fabs(dx) < 1e-6) {
        return 1e6;
    }
    return dy / dx;
}

static bool x_at_y(const cv::Vec4i& line, double y, double& x) {
    double x1 = line[0];
    double y1 = line[1];
    double x2 = line[2];
    double y2 = line[3];
    double dy = y2 - y1;
    if (std::fabs(dy) < 1e-6) {
        return false;
    }
    double t = (y - y1) / dy;
    x = x1 + t * (x2 - x1);
    return true;
}

static void detect_lanes(const cv::Mat& bgr, cv::Vec4i& left, cv::Vec4i& right,
                         bool& has_left, bool& has_right) {
    has_left = false;
    has_right = false;
    cv::Mat gray, edges;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Canny(gray, edges, 60, 160);

    int w = bgr.cols;
    int h = bgr.rows;
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1);
    std::vector<cv::Point> roi = {
        {static_cast<int>(w * 0.05), h - 1},
        {static_cast<int>(w * 0.42), static_cast<int>(h * 0.55)},
        {static_cast<int>(w * 0.58), static_cast<int>(h * 0.55)},
        {static_cast<int>(w * 0.95), h - 1},
    };
    cv::fillConvexPoly(mask, roi, 255);
    cv::bitwise_and(edges, mask, edges);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180.0, 40, 40, 80);

    std::vector<cv::Vec4i> lefts;
    std::vector<cv::Vec4i> rights;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        double m = line_slope(lines[i]);
        if (std::fabs(m) < 0.35 || std::fabs(m) > 5.0) {
            continue;
        }
        double x_bottom = 0.0;
        if (!x_at_y(lines[i], h - 1.0, x_bottom)) {
            continue;
        }
        if (m < 0.0 && x_bottom < w * 0.6) {
            lefts.push_back(lines[i]);
        }
        if (m > 0.0 && x_bottom > w * 0.4) {
            rights.push_back(lines[i]);
        }
    }

    auto average_line = [](const std::vector<cv::Vec4i>& in, cv::Vec4i& out) -> bool {
        if (in.empty()) {
            return false;
        }
        double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        for (std::size_t i = 0; i < in.size(); ++i) {
            x1 += in[i][0];
            y1 += in[i][1];
            x2 += in[i][2];
            y2 += in[i][3];
        }
        double n = static_cast<double>(in.size());
        out = cv::Vec4i(static_cast<int>(x1 / n), static_cast<int>(y1 / n),
                        static_cast<int>(x2 / n), static_cast<int>(y2 / n));
        return true;
    };

    has_left = average_line(lefts, left);
    has_right = average_line(rights, right);
}

static void draw_vision_ui(cv::Mat& frame, const VisionCommand& command,
                           bool has_left, bool has_right,
                           const cv::Vec4i& left, const cv::Vec4i& right) {
    int w = frame.cols;
    int h = frame.rows;
    frame.convertTo(frame, -1, 0.45, 0);

    double y_top = h * 0.58;
    double y_bot = h - 8.0;
    double x_lb = w * 0.2;
    double x_lt = w * 0.42;
    double x_rb = w * 0.8;
    double x_rt = w * 0.58;
    if (has_left) {
        x_at_y(left, y_bot, x_lb);
        x_at_y(left, y_top, x_lt);
    }
    if (has_right) {
        x_at_y(right, y_bot, x_rb);
        x_at_y(right, y_top, x_rt);
    }

    std::vector<cv::Point> lane = {
        {static_cast<int>(x_lb), static_cast<int>(y_bot)},
        {static_cast<int>(x_lt), static_cast<int>(y_top)},
        {static_cast<int>(x_rt), static_cast<int>(y_top)},
        {static_cast<int>(x_rb), static_cast<int>(y_bot)},
    };
    cv::Mat overlay = frame.clone();
    cv::fillConvexPoly(overlay, lane, cv::Scalar(180, 90, 20));
    cv::addWeighted(overlay, 0.28, frame, 0.72, 0, frame);

    cv::line(frame, lane[0], lane[1], cv::Scalar(220, 200, 40), 5, cv::LINE_AA);
    cv::line(frame, lane[3], lane[2], cv::Scalar(220, 200, 40), 5, cv::LINE_AA);

    std::vector<cv::Point> path;
    for (int i = 0; i <= 12; ++i) {
        double t = static_cast<double>(i) / 12.0;
        double x = (x_lb + x_rb) * 0.5 * (1.0 - t) + (x_lt + x_rt) * 0.5 * t;
        double y = y_bot * (1.0 - t) + y_top * t;
        path.push_back({static_cast<int>(x), static_cast<int>(y)});
    }
    cv::polylines(frame, path, false, cv::Scalar(40, 90, 255), 6, cv::LINE_AA);

    cv::rectangle(frame, cv::Point(0, 0), cv::Point(w, 64), cv::Scalar(10, 10, 10), cv::FILLED);
    std::string status = command.engaged ? "ATLAS VISION  ·  ENGAGED" : "ATLAS VISION  ·  SEARCHING";
    cv::putText(frame, status, cv::Point(18, 40), cv::FONT_HERSHEY_SIMPLEX, 0.85,
                command.engaged ? cv::Scalar(80, 220, 120) : cv::Scalar(80, 180, 255), 2, cv::LINE_AA);

    char hud[96];
    std::snprintf(hud, sizeof(hud), "v=%.1f m/s   steer=%.2f rad   q=quit",
                  command.target_speed, command.steering);
    cv::putText(frame, hud, cv::Point(18, h - 18), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

    int bar_w = static_cast<int>((command.steering / 0.4) * (w * 0.18));
    cv::rectangle(frame, cv::Point(w / 2 - 80, 78), cv::Point(w / 2 + 80, 94),
                  cv::Scalar(40, 40, 40), cv::FILLED);
    cv::rectangle(frame, cv::Point(w / 2, 78), cv::Point(w / 2 + bar_w, 94),
                  cv::Scalar(40, 90, 255), cv::FILLED);
}

bool atlas_webcam_available() {
    return true;
}

bool atlas_webcam_start() {
    g_cap.open(0);
    if (!g_cap.isOpened()) {
        std::cerr << "Could not open camera 0. Close other apps using the camera.\n"
                  << "On macOS: System Settings → Privacy & Security → Camera → allow Terminal.\n";
        return false;
    }
    g_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    g_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    std::cout << "Atlas Vision engaged. Point the camera at a hallway or two parallel edges.\n"
              << "Steering and speed come from detected lanes. Press q to quit.\n";
    return true;
}

bool atlas_webcam_pump(VisionCommand& command) {
    cv::Mat frame;
    if (!g_cap.read(frame) || frame.empty()) {
        std::cerr << "Lost camera frame.\n";
        return false;
    }

    cv::Vec4i left;
    cv::Vec4i right;
    bool has_left = false;
    bool has_right = false;
    detect_lanes(frame, left, right, has_left, has_right);

    double y_bot = frame.rows - 8.0;
    double x_left = frame.cols * 0.28;
    double x_right = frame.cols * 0.72;
    if (has_left) {
        x_at_y(left, y_bot, x_left);
    }
    if (has_right) {
        x_at_y(right, y_bot, x_right);
    }

    double mid = 0.0;
    if (has_left && has_right) {
        mid = 0.5 * (x_left + x_right);
        command.target_speed = 6.0;
        command.engaged = true;
    } else if (has_left) {
        mid = x_left + frame.cols * 0.18;
        command.target_speed = 3.0;
        command.engaged = true;
    } else if (has_right) {
        mid = x_right - frame.cols * 0.18;
        command.target_speed = 3.0;
        command.engaged = true;
    } else {
        mid = frame.cols / 2.0;
        command.target_speed = 1.2;
        command.engaged = false;
    }

    double offset = (mid - frame.cols / 2.0) / (frame.cols / 2.0);
    double steering = -1.2 * offset;
    if (steering > 0.4) {
        steering = 0.4;
    }
    if (steering < -0.4) {
        steering = -0.4;
    }
    if (!command.engaged) {
        steering = 0.0;
    }
    command.steering = steering;

    draw_vision_ui(frame, command, has_left, has_right, left, right);
    cv::imshow("Atlas Vision", frame);
    int key = cv::waitKey(1);
    if (key == 'q' || key == 'Q' || key == 27) {
        return false;
    }
    return true;
}

void atlas_webcam_stop() {
    g_cap.release();
    cv::destroyAllWindows();
}

#else

bool atlas_webcam_available() {
    return false;
}

bool atlas_webcam_start() {
    std::cerr << "This build has no OpenCV. Install OpenCV, reconfigure CMake, rebuild.\n"
              << "macOS: brew install opencv cmake\n"
              << "Windows MSYS2 UCRT64: pacman -S mingw-w64-ucrt-x86_64-opencv\n";
    return false;
}

bool atlas_webcam_pump(VisionCommand&) {
    return false;
}

void atlas_webcam_stop() {
}

#endif
