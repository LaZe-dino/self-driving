#include "camera_model.hpp"
#include <cmath>

void CameraModel::update() {
    // Axis swap from vehicle (X fwd, Y left, Z up) to an unrotated camera
    // (x right, y down, z fwd): x = -Y, y = -Z, z = X.
    cv::Matx33d axes(0, -1, 0,
                     0, 0, -1,
                     1, 0, 0);
    // Yaw left turns the optical axis toward +Y. In camera coordinates that is
    // a rotation about the camera y axis.
    double cy = std::cos(yaw_rad), sy = std::sin(yaw_rad);
    cv::Matx33d yaw(cy, 0, sy,
                    0, 1, 0,
                    -sy, 0, cy);
    // Pitch down tilts the optical axis toward the road: rotation about x.
    // Check: a level forward ray (0,0,1) lands at y = -sin(pitch) < 0, i.e.
    // above the image centre, which is where the horizon should be.
    double cp = std::cos(pitch_rad), sp = std::sin(pitch_rad);
    cv::Matx33d pitch(1, 0, 0,
                      0, cp, -sp,
                      0, sp, cp);
    R = pitch * yaw * axes;
    cv::Vec3d camera_center(0.0, 0.0, height_m);
    t = -(R * camera_center);

    cv::Matx33d Rt(R(0, 0), R(0, 1), t[0],
                   R(1, 0), R(1, 1), t[1],
                   R(2, 0), R(2, 1), t[2]);
    H_ground_to_image = K * Rt;
    H_image_to_ground = H_ground_to_image.inv();
}

cv::Point2d CameraModel::ground_to_image(double X, double Y) const {
    cv::Vec3d p = H_ground_to_image * cv::Vec3d(X, Y, 1.0);
    return {p[0] / p[2], p[1] / p[2]};
}

bool CameraModel::image_to_ground(const cv::Point2d& px, double& X, double& Y) const {
    cv::Vec3d g = H_image_to_ground * cv::Vec3d(px.x, px.y, 1.0);
    if (std::fabs(g[2]) < 1e-12) {
        return false;
    }
    X = g[0] / g[2];
    Y = g[1] / g[2];
    return X > 0.0;
}

cv::Point2d CameraModel::vanishing_point(double dx, double dy) const {
    // A point infinitely far along direction d has homogeneous coords
    // (dx, dy, 0, 0): translation no longer matters, only rotation.
    cv::Vec3d p = K * (R * cv::Vec3d(dx, dy, 0.0));
    return {p[0] / p[2], p[1] / p[2]};
}

double CameraModel::horizon_row_at(double u) const {
    cv::Point2d a = vanishing_point(1.0, 1.0);
    cv::Point2d b = vanishing_point(1.0, -1.0);
    if (std::fabs(b.x - a.x) < 1e-9) {
        return a.y;
    }
    return a.y + (u - a.x) * (b.y - a.y) / (b.x - a.x);
}

CameraModel default_camera_model(cv::Size image_size, double horizontal_fov_deg) {
    CameraModel m;
    m.image_size = image_size;
    double f = 0.5 * image_size.width / std::tan(0.5 * horizontal_fov_deg * CV_PI / 180.0);
    m.K = cv::Matx33d(f, 0, image_size.width / 2.0,
                      0, f, image_size.height / 2.0,
                      0, 0, 1);
    m.source = "default (assumed " + std::to_string(static_cast<int>(horizontal_fov_deg)) +
               " deg FOV, uncalibrated)";
    m.update();
    return m;
}

CameraModel scaled_camera_model(const CameraModel& model, cv::Size size) {
    CameraModel m = model;
    if (model.image_size.width <= 0 || model.image_size.height <= 0 || size == model.image_size) {
        return m;
    }
    double sx = static_cast<double>(size.width) / model.image_size.width;
    double sy = static_cast<double>(size.height) / model.image_size.height;
    m.K(0, 0) = model.K(0, 0) * sx;
    m.K(0, 1) = model.K(0, 1) * sx;
    m.K(0, 2) = (model.K(0, 2) + 0.5) * sx - 0.5;
    m.K(1, 1) = model.K(1, 1) * sy;
    m.K(1, 2) = (model.K(1, 2) + 0.5) * sy - 0.5;
    m.image_size = size;
    m.dist = model.dist.clone();
    m.source = model.source + " (scaled " + std::to_string(model.image_size.width) + "x" +
               std::to_string(model.image_size.height) + " -> " + std::to_string(size.width) + "x" +
               std::to_string(size.height) + ")";
    m.update();
    return m;
}

bool same_aspect_ratio(cv::Size a, cv::Size b, double tolerance) {
    if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) {
        return false;
    }
    double ra = static_cast<double>(a.width) / a.height;
    double rb = static_cast<double>(b.width) / b.height;
    return std::fabs(ra - rb) <= tolerance * ra;
}

bool load_camera_model(const std::string& path, CameraModel& model, std::string& error) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        error = "cannot open camera model " + path;
        return false;
    }
    cv::Mat K;
    fs["K"] >> K;
    if (K.rows != 3 || K.cols != 3) {
        error = path + " has no 3x3 K matrix";
        return false;
    }
    model.K = cv::Matx33d(K);
    fs["dist"] >> model.dist;
    if (model.dist.empty()) {
        model.dist = cv::Mat::zeros(1, 5, CV_64F);
    }
    int w = 0, h = 0;
    fs["image_width"] >> w;
    fs["image_height"] >> h;
    model.image_size = cv::Size(w, h);
    fs["height_m"] >> model.height_m;
    fs["pitch_rad"] >> model.pitch_rad;
    fs["yaw_rad"] >> model.yaw_rad;
    int ic = 0, mc = 0;
    fs["intrinsics_calibrated"] >> ic;
    fs["mount_calibrated"] >> mc;
    model.intrinsics_calibrated = ic != 0;
    model.mount_calibrated = mc != 0;
    model.source = path;
    model.update();
    return true;
}

bool save_camera_model(const std::string& path, const CameraModel& model, std::string& error) {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        error = "cannot write camera model " + path;
        return false;
    }
    fs << "image_width" << model.image_size.width;
    fs << "image_height" << model.image_size.height;
    fs << "K" << cv::Mat(model.K);
    fs << "dist" << model.dist;
    fs << "height_m" << model.height_m;
    fs << "pitch_rad" << model.pitch_rad;
    fs << "yaw_rad" << model.yaw_rad;
    fs << "intrinsics_calibrated" << (model.intrinsics_calibrated ? 1 : 0);
    fs << "mount_calibrated" << (model.mount_calibrated ? 1 : 0);
    return true;
}
