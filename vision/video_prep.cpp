#include "video_prep.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

cv::Size processing_size(cv::Size native, int process_width) {
    if (process_width <= 0 || native.width <= 0 || process_width >= native.width) {
        return native;
    }
    double s = static_cast<double>(process_width) / native.width;
    int h = std::max(1, static_cast<int>(std::lround(native.height * s)));
    return {process_width, h};
}

cv::Mat resize_for_processing(const cv::Mat& in, cv::Size size) {
    if (in.empty() || in.size() == size) {
        return in;
    }
    cv::Mat out;
    cv::resize(in, out, size, 0, 0, cv::INTER_AREA);
    return out;
}

namespace {

std::string lower_extension(const std::string& name) {
    std::string ext = fs::path(name).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

}  // namespace

bool is_video_file_name(const std::string& name) {
    std::string e = lower_extension(name);
    return e == ".mov" || e == ".mp4" || e == ".m4v";
}

bool is_image_file_name(const std::string& name) {
    std::string e = lower_extension(name);
    return e == ".jpg" || e == ".jpeg" || e == ".png";
}

std::vector<std::string> list_files(const std::string& dir, bool (*accept)(const std::string&)) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec) && accept(it->path().filename().string())) {
            out.push_back(it->path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

double laplacian_variance(const cv::Mat& gray, const cv::Rect& roi) {
    cv::Rect r = roi & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.width < 3 || r.height < 3) {
        return 0.0;
    }
    cv::Mat lap;
    cv::Laplacian(gray(r), lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

double board_pose_distance(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b,
                           cv::Size image_size) {
    if (a.empty() || a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double fwd = 0.0, rev = 0.0;
    std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        cv::Point2f d1 = a[i] - b[i];
        cv::Point2f d2 = a[i] - b[n - 1 - i];
        fwd += std::hypot(d1.x, d1.y);
        rev += std::hypot(d2.x, d2.y);
    }
    double diag = std::hypot(image_size.width, image_size.height);
    return std::min(fwd, rev) / n / std::max(diag, 1.0);
}

ViewSelection select_calibration_views(const std::vector<ChessboardView>& views, cv::Size image_size,
                                       const ViewSelectParams& params) {
    ViewSelection sel;
    std::vector<double> sharp;
    for (const ChessboardView& v : views) sharp.push_back(v.sharpness);
    sel.sharpness_threshold = std::max(params.min_sharpness, params.relative_sharpness * median_of(sharp));

    std::vector<int> pool;
    for (int i = 0; i < static_cast<int>(views.size()); ++i) {
        if (views[i].sharpness >= sel.sharpness_threshold) {
            pool.push_back(i);
        } else {
            ++sel.blurry;
        }
    }
    if (pool.empty() || params.max_views <= 0) {
        sel.surplus = static_cast<int>(pool.size());
        return sel;
    }

    int first = *std::max_element(pool.begin(), pool.end(),
                                  [&](int a, int b) { return views[a].sharpness < views[b].sharpness; });
    sel.selected.push_back(first);
    // nearest[i]: pose distance from pool[i] to the closest selected view.
    std::vector<double> nearest(pool.size(), std::numeric_limits<double>::infinity());
    std::vector<bool> taken(pool.size(), false);
    auto refresh = [&](int chosen) {
        for (std::size_t i = 0; i < pool.size(); ++i) {
            if (pool[i] == chosen) taken[i] = true;
            if (taken[i]) continue;
            nearest[i] = std::min(nearest[i],
                                  board_pose_distance(views[pool[i]].corners, views[chosen].corners, image_size));
        }
    };
    refresh(first);

    while (static_cast<int>(sel.selected.size()) < params.max_views) {
        int best = -1;
        for (std::size_t i = 0; i < pool.size(); ++i) {
            if (!taken[i] && (best < 0 || nearest[i] > nearest[best])) best = static_cast<int>(i);
        }
        if (best < 0 || nearest[best] < params.min_pose_distance) break;
        sel.selected.push_back(pool[best]);
        refresh(pool[best]);
    }
    for (std::size_t i = 0; i < pool.size(); ++i) {
        if (taken[i]) continue;
        if (nearest[i] < params.min_pose_distance) {
            ++sel.duplicates;
        } else {
            ++sel.surplus;
        }
    }
    return sel;
}

std::vector<int> outlier_views(const std::vector<double>& per_view_errors, double factor) {
    std::vector<int> out;
    double med = median_of(per_view_errors);
    for (int i = 0; i < static_cast<int>(per_view_errors.size()); ++i) {
        if (per_view_errors[i] > factor * med) out.push_back(i);
    }
    return out;
}
