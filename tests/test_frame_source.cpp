#include "test_main.hpp"
#include "../vision/frame_source.hpp"
#include <string>
#include <vector>

TEST(video_file_backends_order) {
    std::vector<int> backends = video_file_backends();
    CHECK(backends.size() >= 3);
    CHECK(backends[0] == cv::CAP_ANY);
    CHECK(backends[1] == cv::CAP_FFMPEG);
    CHECK(backends.back() == kVideoBackendDefault);
#if defined(__APPLE__)
    CHECK(backends.size() == 4u);
    CHECK(backends[2] == cv::CAP_AVFOUNDATION);
#else
    CHECK(backends.size() == 3u);
#endif
}

TEST(video_path_is_file_rejects_missing) {
    CHECK(!video_path_is_file(""));
    CHECK(!video_path_is_file("data/videos/this_file_should_not_exist_atlas.mp4"));
}

TEST(describe_video_open_failure_missing_vs_codec) {
    std::string missing = describe_video_open_failure("data/videos/project_video.mp4", false);
    CHECK(missing.find("file not found") != std::string::npos);
    CHECK(missing.find("gitignored") != std::string::npos);
    CHECK(missing.find("README") != std::string::npos);
    std::string codec = describe_video_open_failure("data/videos/project_video.mp4", true);
    CHECK(codec.find("file not found") == std::string::npos);
    CHECK(codec.find("decode") != std::string::npos);
}

TEST(open_video_file_missing_path_sets_error) {
    cv::VideoCapture cap;
    std::string error;
    CHECK(!open_video_file(cap, "data/videos/this_file_should_not_exist_atlas.mp4", error));
    CHECK(error.find("file not found") != std::string::npos);
    CHECK(error.find("gitignored") != std::string::npos);
    CHECK(!cap.isOpened());
}
