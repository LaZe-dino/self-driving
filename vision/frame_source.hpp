#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

struct Frame {
    long long id = -1;
    double capture_time_s = 0.0;
    double media_time_s = 0.0;
    cv::Mat image;
};

struct FrameSourceStats {
    long long captured = 0;
    long long dropped = 0;
    long long read_failures = 0;
    long long reconnects = 0;
    long long loops = 0;
    double source_fps = 0.0;
};

struct SourceSpec {
    bool live = false;
    int camera_index = 0;
    std::string path;
    bool loop = false;
    int width = 1280;
    int height = 720;
};

// Opens a video file with FFmpeg, falling back to any other backend
// (e.g. AVFoundation on macOS) if this OpenCV build lacks FFmpeg.
bool open_video_file(cv::VideoCapture& cap, const std::string& path);

// Live cameras: a capture thread keeps only the newest frames (queue of 2,
// oldest dropped) so processing never falls behind real time.
// Files: the capture thread blocks when the queue is full, so every frame is
// processed and runs are reproducible.
class FrameSource {
public:
    ~FrameSource();

    bool open(const SourceSpec& spec, std::string& error);
    bool next(Frame& out);
    void close();

    FrameSourceStats stats() const;
    bool live() const { return spec_.live; }
    std::string describe() const;

private:
    bool open_capture(std::string& error);
    void capture_loop();
    void push(Frame&& frame);

    SourceSpec spec_;
    cv::VideoCapture cap_;
    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Frame> queue_;
    std::size_t capacity_ = 2;
    bool stop_ = false;
    bool ended_ = false;
    FrameSourceStats stats_;
    std::string backend_;
    std::chrono::steady_clock::time_point t0_;
};
