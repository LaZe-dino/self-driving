#include "frame_source.hpp"
#include "video_prep.hpp"
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

#if defined(_WIN32)
// Media Foundation is the native Windows camera API; DirectShow is the older
// one. CAP_ANY tries GStreamer first on the MSYS2 OpenCV build.
const int kCameraBackends[] = {cv::CAP_MSMF, cv::CAP_DSHOW, cv::CAP_ANY};
const char* kCameraHint = "no Media Foundation or DirectShow device, or it is in use";
#elif defined(__APPLE__)
const int kCameraBackends[] = {cv::CAP_AVFOUNDATION, cv::CAP_ANY};
const char* kCameraHint =
    "no AVFoundation device, it is in use, or camera access is denied: allow your terminal/IDE in "
    "System Settings > Privacy & Security > Camera, then restart it";
#else
const int kCameraBackends[] = {cv::CAP_V4L2, cv::CAP_ANY};
const char* kCameraHint = "no V4L2 device, it is in use, or no permission on /dev/video*";
#endif

}  // namespace

std::vector<int> video_file_backends() {
    std::vector<int> backends = {cv::CAP_ANY, cv::CAP_FFMPEG};
#if defined(__APPLE__)
    backends.push_back(cv::CAP_AVFOUNDATION);
#endif
    backends.push_back(kVideoBackendDefault);
    return backends;
}

bool video_path_is_file(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
}

std::string describe_video_open_failure(const std::string& path, bool file_exists) {
    if (!file_exists) {
        return "file not found: " + path +
               "\ndata/videos is gitignored and is not cloned with the repo. "
               "Copy or download the clips listed in README.md (Data section).";
    }
    return "could not decode video file " + path +
           " (the file exists, but no OpenCV backend could open it)";
}

bool open_video_file(cv::VideoCapture& cap, const std::string& path, std::string& error) {
    if (!video_path_is_file(path)) {
        error = describe_video_open_failure(path, false);
        return false;
    }
    for (int backend : video_file_backends()) {
        cap.release();
        bool ok = backend == kVideoBackendDefault ? cap.open(path) : cap.open(path, backend);
        if (ok && cap.isOpened()) {
            // FFmpeg defaults to on; set it anyway. Backends without the
            // property just return false.
            cap.set(cv::CAP_PROP_ORIENTATION_AUTO, 1);
            error.clear();
            return true;
        }
    }
    error = describe_video_open_failure(path, true);
    return false;
}

bool open_video_file(cv::VideoCapture& cap, const std::string& path) {
    std::string error;
    return open_video_file(cap, path, error);
}

std::string describe_orientation(cv::VideoCapture& cap) {
    if (cap.getBackendName() != "FFMPEG") {
        return "";
    }
    int meta = static_cast<int>(cap.get(cv::CAP_PROP_ORIENTATION_META));
    bool automatic = cap.get(cv::CAP_PROP_ORIENTATION_AUTO) != 0;
    return "rotation metadata " + std::to_string(meta) + " deg, auto-rotate " + (automatic ? "on" : "off");
}

FrameSource::~FrameSource() {
    close();
}

bool FrameSource::open_capture(std::string& error) {
    if (spec_.live) {
        for (int backend : kCameraBackends) {
            if (cap_.open(spec_.camera_index, backend)) {
                break;
            }
        }
        if (!cap_.isOpened()) {
            error = "could not open camera " + std::to_string(spec_.camera_index) + " (" + kCameraHint + ")";
            return false;
        }
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, spec_.width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, spec_.height);
    } else {
        if (!open_video_file(cap_, spec_.path, error)) {
            return false;
        }
    }
    backend_ = cap_.getBackendName();
    orientation_ = spec_.live ? "" : describe_orientation(cap_);
    stats_.source_fps = cap_.get(cv::CAP_PROP_FPS);
    return true;
}

bool FrameSource::open(const SourceSpec& spec, std::string& error) {
    close();
    spec_ = spec;
    stop_ = false;
    ended_ = false;
    queue_.clear();
    stats_ = FrameSourceStats();
    if (!open_capture(error)) {
        return false;
    }
    t0_ = std::chrono::steady_clock::now();
    thread_ = std::thread(&FrameSource::capture_loop, this);
    return true;
}

void FrameSource::push(Frame&& frame) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (spec_.live) {
        while (queue_.size() >= capacity_) {
            queue_.pop_front();
            ++stats_.dropped;
        }
    } else {
        cv_.wait(lock, [this] { return stop_ || queue_.size() < capacity_; });
        if (stop_) {
            return;
        }
    }
    queue_.push_back(std::move(frame));
    ++stats_.captured;
    cv_.notify_all();
}

void FrameSource::capture_loop() {
    long long next_id = 0;
    int consecutive_failures = 0;
    double loop_offset_s = 0.0;
    double last_media_s = 0.0;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                return;
            }
        }
        Frame frame;
        bool ok = cap_.read(frame.image) && !frame.image.empty();
        double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();

        if (!ok && !spec_.live) {
            if (!spec_.loop) {
                std::lock_guard<std::mutex> lock(mutex_);
                ended_ = true;
                cv_.notify_all();
                return;
            }
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            loop_offset_s = last_media_s + (stats_.source_fps > 0 ? 1.0 / stats_.source_fps : 0.0);
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.loops;
            continue;
        }

        if (!ok) {
            // A live camera can deliver an empty buffer during renegotiation.
            // One bad read is not the end of the stream; a run of them means
            // the device went away and we reopen it.
            ++consecutive_failures;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++stats_.read_failures;
            }
            if (consecutive_failures >= 30) {
                std::cerr << "[frame_source] " << consecutive_failures
                          << " consecutive read failures, reopening camera\n";
                cap_.release();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                std::string error;
                if (!open_capture(error)) {
                    std::cerr << "[frame_source] reopen failed: " << error << "\n";
                }
                std::lock_guard<std::mutex> lock(mutex_);
                ++stats_.reconnects;
                consecutive_failures = 0;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        consecutive_failures = 0;

        cv::Size native = frame.image.size();
        frame.image = resize_for_processing(frame.image, processing_size(native, spec_.process_width));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.native_size = native;
            stats_.frame_size = frame.image.size();
        }
        frame.id = next_id++;
        frame.capture_time_s = now;
        if (spec_.live) {
            frame.media_time_s = now;
        } else {
            last_media_s = loop_offset_s + cap_.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
            frame.media_time_s = last_media_s;
        }
        push(std::move(frame));
    }
}

bool FrameSource::next(Frame& out) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return stop_ || ended_ || !queue_.empty(); });
    if (queue_.empty()) {
        return false;
    }
    out = std::move(queue_.front());
    queue_.pop_front();
    cv_.notify_all();
    return true;
}

void FrameSource::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        cv_.notify_all();
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    cap_.release();
}

FrameSourceStats FrameSource::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

std::string FrameSource::describe() const {
    if (spec_.live) {
        return "camera " + std::to_string(spec_.camera_index) + " via " + backend_;
    }
    return spec_.path + " via " + backend_ + (orientation_.empty() ? "" : " (" + orientation_ + ")") +
           (spec_.loop ? " (looping)" : "");
}
