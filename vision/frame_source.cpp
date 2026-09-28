#include "frame_source.hpp"
#include <iostream>
#include <string>
#include <utility>

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

bool open_video_file(cv::VideoCapture& cap, const std::string& path) {
    return cap.open(path, cv::CAP_FFMPEG) || cap.open(path, cv::CAP_ANY);
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
        if (!open_video_file(cap_, spec_.path)) {
            error = "could not open video file " + spec_.path;
            return false;
        }
    }
    backend_ = cap_.getBackendName();
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
    return spec_.path + " via " + backend_ + (spec_.loop ? " (looping)" : "");
}
