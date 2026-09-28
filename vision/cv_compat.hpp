#pragma once

// OpenCV 5 split calib3d and features2d into calib, geometry (3d/2d),
// features and objdetect. OpenCV 4.x keeps them in calib3d and imgproc.
// Covers: calibrateCamera, findChessboardCorners, initUndistortRectifyMap,
// estimateAffinePartial2D, fitLine, goodFeaturesToTrack, cornerSubPix.
#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/calib.hpp>
#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/objdetect.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif
