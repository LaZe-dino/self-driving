#include "test_main.hpp"
#include "../vision/camera_model.hpp"
#include "../vision/video_prep.hpp"
#include <opencv2/imgproc.hpp>

static CameraModel iphone_like_camera() {
    CameraModel m = default_camera_model(cv::Size(1920, 1080), 68.0);
    m.K(0, 2) = 955.3;
    m.K(1, 2) = 547.8;
    m.K(1, 1) = m.K(0, 0) * 1.002;
    m.dist = cv::Mat(cv::Matx<double, 1, 5>(0.12, -0.3, 0.001, -0.0005, 0.2)).clone();
    m.height_m = 1.32;
    m.pitch_rad = 0.05;
    m.yaw_rad = -0.01;
    m.update();
    return m;
}

TEST(scaled_camera_projects_ground_to_scaled_pixel) {
    CameraModel m = iphone_like_camera();
    for (cv::Size size : {cv::Size(1280, 720), cv::Size(3840, 2160), cv::Size(960, 540)}) {
        CameraModel s = scaled_camera_model(m, size);
        CHECK(s.image_size == size);
        double sx = static_cast<double>(size.width) / m.image_size.width;
        double sy = static_cast<double>(size.height) / m.image_size.height;
        const double pts[][2] = {{6, 0}, {12, 1.8}, {25, -3.5}, {50, 4}};
        for (const auto& p : pts) {
            cv::Point2d a = m.ground_to_image(p[0], p[1]);
            cv::Point2d b = s.ground_to_image(p[0], p[1]);
            // cv::resize pixel-centre mapping.
            CHECK_NEAR(b.x, (a.x + 0.5) * sx - 0.5, 1e-6);
            CHECK_NEAR(b.y, (a.y + 0.5) * sy - 0.5, 1e-6);
            double X = 0, Y = 0;
            CHECK(s.image_to_ground(b, X, Y));
            CHECK_NEAR(X, p[0], 1e-6);
            CHECK_NEAR(Y, p[1], 1e-6);
        }
        CHECK_NEAR(s.height_m, m.height_m, 1e-12);
        CHECK_NEAR(s.pitch_rad, m.pitch_rad, 1e-12);
        CHECK_NEAR(cv::norm(s.dist, m.dist), 0.0, 1e-12);
    }
}

TEST(scaled_camera_round_trips_and_keeps_same_size) {
    CameraModel m = iphone_like_camera();
    CameraModel same = scaled_camera_model(m, m.image_size);
    CHECK_NEAR(cv::norm(same.K - m.K), 0.0, 1e-12);
    CameraModel back = scaled_camera_model(scaled_camera_model(m, cv::Size(1280, 720)), m.image_size);
    CHECK_NEAR(cv::norm(back.K - m.K), 0.0, 1e-9);
}

TEST(aspect_ratio_and_processing_size) {
    CHECK(same_aspect_ratio(cv::Size(1920, 1080), cv::Size(1280, 720)));
    CHECK(same_aspect_ratio(cv::Size(3840, 2160), cv::Size(1280, 720)));
    CHECK(!same_aspect_ratio(cv::Size(1920, 1080), cv::Size(1080, 1920)));
    CHECK(!same_aspect_ratio(cv::Size(1280, 960), cv::Size(1280, 720)));
    CHECK(!same_aspect_ratio(cv::Size(1920, 1080), cv::Size(1920, 1088)));

    CHECK(processing_size(cv::Size(1920, 1080), 1280) == cv::Size(1280, 720));
    CHECK(processing_size(cv::Size(3840, 2160), 1280) == cv::Size(1280, 720));
    CHECK(processing_size(cv::Size(1920, 1080), 0) == cv::Size(1920, 1080));
    CHECK(processing_size(cv::Size(1280, 720), 1920) == cv::Size(1280, 720));  // never upscale
    CHECK(processing_size(cv::Size(1920, 1080), 1000) == cv::Size(1000, 563));
}

TEST(file_name_filters) {
    CHECK(is_video_file_name("IMG_1234.MOV"));
    CHECK(is_video_file_name("drive.mp4"));
    CHECK(is_video_file_name("clip.M4V"));
    CHECK(!is_video_file_name("IMG_1234.HEIC"));
    CHECK(!is_video_file_name("notes.txt"));
    CHECK(!is_video_file_name("mov"));
    CHECK(is_image_file_name("a.JPG"));
    CHECK(is_image_file_name("a.jpeg"));
    CHECK(is_image_file_name("a.png"));
    CHECK(!is_image_file_name("a.heic"));
}

TEST(frame_sampler_every_half_second) {
    FrameSampler s;
    s.interval_s = 0.5;
    int picked = 0;
    for (int i = 0; i < 300; ++i) {  // 10 s at 30 fps
        picked += s.due(i / 30.0) ? 1 : 0;
    }
    CHECK(picked == 20);
    FrameSampler t;
    CHECK(t.due(0.0));
    CHECK(!t.due(0.49));
    CHECK(t.due(0.5));
}

TEST(laplacian_variance_ranks_blur) {
    cv::Mat board(200, 200, CV_8UC1);
    for (int r = 0; r < 200; ++r) {
        for (int c = 0; c < 200; ++c) board.at<uchar>(r, c) = ((r / 20 + c / 20) % 2) ? 255 : 0;
    }
    cv::Mat blurred;
    cv::GaussianBlur(board, blurred, cv::Size(0, 0), 4.0);
    cv::Rect all(0, 0, 200, 200);
    CHECK(laplacian_variance(board, all) > 4.0 * laplacian_variance(blurred, all));
    CHECK_NEAR(laplacian_variance(cv::Mat(50, 50, CV_8UC1, cv::Scalar(128)), all), 0.0, 1e-9);
}

static std::vector<cv::Point2f> grid_corners(float x0, float y0, float step) {
    std::vector<cv::Point2f> c;
    for (int r = 0; r < 6; ++r) {
        for (int k = 0; k < 9; ++k) c.push_back(cv::Point2f(x0 + k * step, y0 + r * step));
    }
    return c;
}

TEST(board_pose_distance_handles_reversed_order) {
    cv::Size img(1280, 720);
    std::vector<cv::Point2f> a = grid_corners(100, 100, 30);
    std::vector<cv::Point2f> rev(a.rbegin(), a.rend());
    CHECK_NEAR(board_pose_distance(a, rev, img), 0.0, 1e-9);
    std::vector<cv::Point2f> shifted = grid_corners(100 + 146.9f, 100, 30);  // 146.9 px ~ 0.1 diagonal
    CHECK_NEAR(board_pose_distance(a, shifted, img), 146.9 / std::hypot(1280.0, 720.0), 1e-5);
}

TEST(view_selection_drops_blur_and_duplicates) {
    cv::Size img(1280, 720);
    std::vector<ChessboardView> views;
    // Five distinct poses, each seen three times (the board held still), plus
    // two blurry frames at new poses.
    for (int p = 0; p < 5; ++p) {
        for (int k = 0; k < 3; ++k) {
            ChessboardView v;
            v.corners = grid_corners(50.0f + p * 200.0f + k * 1.0f, 100.0f + p * 50.0f, 30.0f);
            v.sharpness = 1000.0 + 10.0 * k + p;
            views.push_back(v);
        }
    }
    for (int b = 0; b < 2; ++b) {
        ChessboardView v;
        v.corners = grid_corners(300.0f + b * 300.0f, 400.0f, 25.0f);
        v.sharpness = 100.0;
        views.push_back(v);
    }
    ViewSelectParams params;
    ViewSelection sel = select_calibration_views(views, img, params);
    CHECK(sel.blurry == 2);
    CHECK(sel.selected.size() == 5u);
    CHECK(sel.duplicates == 10);
    CHECK(sel.surplus == 0);
    // The sharpest view is picked first.
    CHECK(sel.selected[0] == 14);

    params.max_views = 3;
    ViewSelection capped = select_calibration_views(views, img, params);
    CHECK(capped.selected.size() == 3u);
    // 2 unpicked poses x 3 views are surplus; the other copies of the 3 picked poses are duplicates.
    CHECK(capped.surplus == 6);
    CHECK(capped.duplicates == 6);
}

TEST(outlier_views_above_twice_median) {
    std::vector<double> e = {0.3, 0.35, 0.28, 0.4, 1.2, 0.31};
    std::vector<int> bad = outlier_views(e, 2.0);
    CHECK(bad.size() == 1u);
    CHECK(!bad.empty() && bad[0] == 4);
    CHECK(outlier_views({0.3, 0.3, 0.3}, 2.0).empty());
}
