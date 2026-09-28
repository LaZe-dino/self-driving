#include "ground_view.hpp"
#include <opencv2/imgproc.hpp>

cv::Matx33d GroundGrid::grid_to_ground() const {
    // X = x_max - (row + 0.5) r,  Y = y_half - (col + 0.5) r
    double r = resolution_m;
    return cv::Matx33d(0, -r, x_max - 0.5 * r,
                       -r, 0, y_half - 0.5 * r,
                       0, 0, 1);
}

void GroundView::configure(const CameraModel& camera, const GroundGrid& grid) {
    grid_ = grid;
    image_size_ = camera.image_size;
    H_grid_to_image_ = camera.H_ground_to_image * grid.grid_to_ground();

    cv::Mat white(camera.image_size, CV_8UC1, cv::Scalar(255));
    cv::warpPerspective(white, valid_, cv::Mat(H_grid_to_image_),
                        cv::Size(grid.cols(), grid.rows()),
                        cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                        cv::BORDER_CONSTANT, cv::Scalar(0));
    // Shrink so image-border edges never look like lane paint.
    cv::erode(valid_, valid_, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
}

void GroundView::warp(const cv::Mat& undistorted, cv::Mat& bev) const {
    cv::warpPerspective(undistorted, bev, cv::Mat(H_grid_to_image_),
                        cv::Size(grid_.cols(), grid_.rows()),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_CONSTANT, cv::Scalar::all(0));
}
