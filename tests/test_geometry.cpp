#include "test_main.hpp"
#include "../vision/camera_model.hpp"
#include "../vision/ground_view.hpp"

static CameraModel test_camera() {
    CameraModel m = default_camera_model(cv::Size(1280, 720), 60.0);
    m.height_m = 1.5;
    m.pitch_rad = 0.04;
    m.yaw_rad = -0.02;
    m.update();
    return m;
}

TEST(ground_image_round_trip) {
    CameraModel m = test_camera();
    const double pts[][2] = {{8, 0}, {15, 1.8}, {30, -3.5}, {60, 5}};
    for (const auto& p : pts) {
        cv::Point2d px = m.ground_to_image(p[0], p[1]);
        double X = 0, Y = 0;
        CHECK(m.image_to_ground(px, X, Y));
        CHECK_NEAR(X, p[0], 1e-6);
        CHECK_NEAR(Y, p[1], 1e-6);
    }
}

TEST(point_below_camera_projects_under_centre_when_level) {
    CameraModel m = default_camera_model(cv::Size(1280, 720), 60.0);
    m.height_m = 1.5;
    m.update();
    // Level camera: a ground point X ahead is h/X * f below the centre row.
    cv::Point2d px = m.ground_to_image(10.0, 0.0);
    CHECK_NEAR(px.x, 640.0, 1e-9);
    CHECK_NEAR(px.y, 360.0 + m.K(1, 1) * 1.5 / 10.0, 1e-9);
    // A point to the left (+Y) must appear left of centre.
    CHECK(m.ground_to_image(10.0, 2.0).x < 640.0);
}

TEST(horizon_and_forward_vanishing_point) {
    CameraModel m = test_camera();
    cv::Point2d vp = m.vanishing_point(1.0, 0.0);
    CHECK_NEAR(vp.y, m.K(1, 2) - m.K(1, 1) * std::tan(m.pitch_rad), 1e-6);
    CHECK_NEAR(vp.x, m.K(0, 2) + m.K(0, 0) * std::tan(m.yaw_rad) / std::cos(m.pitch_rad), 1e-6);
    CHECK_NEAR(m.horizon_row_at(vp.x), vp.y, 1e-6);
    double X = 0, Y = 0;
    CHECK(!m.image_to_ground(cv::Point2d(640, vp.y - 20), X, Y));
}

TEST(ground_grid_mapping_is_consistent) {
    GroundGrid g;
    cv::Matx33d G = g.grid_to_ground();
    for (double row : {0.0, 100.0, 679.0}) {
        for (double col : {0.0, 140.0, 279.0}) {
            cv::Vec3d p = G * cv::Vec3d(col, row, 1);
            CHECK_NEAR(p[0], g.x_of_row(row), 1e-9);
            CHECK_NEAR(p[1], g.y_of_col(col), 1e-9);
            CHECK_NEAR(g.row_of_x(p[0]), row, 1e-9);
            CHECK_NEAR(g.col_of_y(p[1]), col, 1e-9);
        }
    }
}
