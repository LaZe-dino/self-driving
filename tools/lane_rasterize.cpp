#include "lane_rasterize.hpp"
#include "../vision/cv_compat.hpp"
#include <cmath>

std::vector<int> logger_h_samples(const CameraModel& camera) {
    double horizon = camera.horizon_row_at(camera.image_size.width / 2.0);
    int row0 = static_cast<int>(std::ceil((horizon + 20.0) / 10.0) * 10.0);
    std::vector<int> rows;
    for (int v = std::max(row0, 0); v < camera.image_size.height; v += 10) rows.push_back(v);
    return rows;
}

std::vector<int> rasterize_boundary(const RoadState& road, double k, const CameraModel& camera,
                                    const std::vector<int>& rows, const RasterizeOptions& opt) {
    std::vector<int> xs(rows.size(), kNoLanePoint);
    if (!road.valid) return xs;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        double v = rows[i];
        // Intersect the image row with the lane: find ground X on the row
        // centre, then iterate using the lane's lateral offset (rows are not
        // lines of constant X once the camera is yawed).
        double X, Y;
        bool ok = camera.image_to_ground({camera.image_size.width / 2.0, v}, X, Y);
        for (int it = 0; ok && it < 3; ++it) {
            cv::Point2d px = camera.ground_to_image(X, road.lateral(X, k));
            ok = camera.image_to_ground({px.x, v}, X, Y);
        }
        bool drawable = ok && X >= opt.x_min && X <= opt.x_max && road.lateral_sigma(X, k) < opt.max_sigma_m;
        if (!drawable) continue;
        cv::Point2d px = camera.ground_to_image(X, road.lateral(X, k));
        if (px.x >= 0 && px.x < camera.image_size.width) xs[i] = static_cast<int>(std::lround(px.x));
    }
    return xs;
}

bool has_distortion(const CameraModel& camera) {
    return !camera.dist.empty() && cv::countNonZero(camera.dist != 0) > 0;
}

std::vector<int> rasterize_boundary_raw(const RoadState& road, double k, const CameraModel& camera,
                                        const std::vector<int>& rows, const RasterizeOptions& opt) {
    if (!has_distortion(camera)) return rasterize_boundary(road, k, camera, rows, opt);
    std::vector<int> xs(rows.size(), kNoLanePoint);
    if (!road.valid) return xs;

    // Distortion bends image rows, so sample the lane densely along the road,
    // push each sample through the lens model and interpolate per raw row.
    std::vector<cv::Point3d> rays;
    std::vector<bool> drawable;
    const double step = 0.1;
    const double w = camera.image_size.width, h = camera.image_size.height;
    for (double X = std::max(opt.x_min, 0.5); X <= opt.x_max + 1e-9; X += step) {
        cv::Point2d px = camera.ground_to_image(X, road.lateral(X, k));
        // Lens models are only valid over the image; far outside it the
        // polynomial folds back and would create phantom crossings.
        bool near_image = px.x > -0.2 * w && px.x < 1.2 * w && px.y > -0.2 * h && px.y < 1.2 * h;
        rays.push_back({(px.x - camera.K(0, 2)) / camera.K(0, 0), (px.y - camera.K(1, 2)) / camera.K(1, 1), 1.0});
        drawable.push_back(near_image && road.lateral_sigma(X, k) < opt.max_sigma_m);
    }
    if (rays.size() < 2) return xs;
    std::vector<cv::Point2d> raw;
    cv::projectPoints(rays, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera.K, camera.dist, raw);

    for (std::size_t r = 0; r < rows.size(); ++r) {
        double v = rows[r];
        for (std::size_t i = 0; i + 1 < raw.size(); ++i) {
            if (!drawable[i] || !drawable[i + 1]) continue;
            double v0 = raw[i].y, v1 = raw[i + 1].y;
            if ((v - v0) * (v - v1) > 0.0 || std::fabs(v1 - v0) < 1e-9) continue;
            double a = (v - v0) / (v1 - v0);
            double u = raw[i].x + a * (raw[i + 1].x - raw[i].x);
            if (u >= 0 && u < camera.image_size.width) xs[r] = static_cast<int>(std::lround(u));
            break;
        }
    }
    return xs;
}
