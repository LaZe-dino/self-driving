#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    cv::Size native_size;
    cv::Size frame_size;
};

struct SourceSpec {
    bool live = false;
    int camera_index = 0;
    std::string path;
    bool loop = false;
    int width = 1280;
    int height = 720;
    // Frames are resized to this width right after capture (0 = native).
    int process_width = 0;
};

// Sentinel for VideoCapture::open(path) with no apiPreference. Must not collide
// with OpenCV backend ids (CAP_ANY is 0).
inline constexpr int kVideoBackendDefault = -1;

// Backends to try, in order: CAP_ANY, CAP_FFMPEG, CAP_AVFOUNDATION (macOS),
// then open(path) with no apiPreference. Homebrew OpenCV often cannot open a
// file via CAP_FFMPEG by name; CAP_ANY first avoids that warning when another
// backend can decode the file.
std::vector<int> video_file_backends();
bool video_path_is_file(const std::string& path);
std::string describe_video_open_failure(const std::string& path, bool file_exists);

// Opens a video file, trying video_file_backends() in order. Checks that the
// path exists before calling OpenCV so a missing gitignored file is not
// reported as an FFmpeg capture-by-name failure. Asks the backend to apply
// the file's rotation metadata (phone videos).
bool open_video_file(cv::VideoCapture& cap, const std::string& path);
bool open_video_file(cv::VideoCapture& cap, const std::string& path, std::string& error);
// "rotation metadata 90 deg, auto-rotate on" for backends that report it,
// otherwise empty.
std::string describe_orientation(cv::VideoCapture& cap);

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
    std::string orientation_;
    std::chrono::steady_clock::time_point t0_;
};
