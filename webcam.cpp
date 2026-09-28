#include "webcam.hpp"
#include <iostream>

#ifdef ATLAS_HAS_OPENCV
#include <opencv2/opencv.hpp>

static cv::VideoCapture g_cap;

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
    std::cout << "Webcam open. Preview window is live. Press q in that window to quit.\n"
              << "This feed does not steer the car yet; the sim still follows the planned path.\n";
    return true;
}

bool atlas_webcam_pump() {
    cv::Mat frame;
    if (!g_cap.read(frame) || frame.empty()) {
        std::cerr << "Lost camera frame.\n";
        return false;
    }
    cv::putText(frame, "Atlas preview (not controlling vehicle)",
                cv::Point(16, 32), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 220, 0), 2);
    cv::imshow("Atlas webcam", frame);
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

bool atlas_webcam_pump() {
    return false;
}

void atlas_webcam_stop() {
}

#endif
