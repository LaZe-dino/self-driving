cmake_minimum_required(VERSION 3.16)
project(atlas_autonomy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    # Without optimisation the per-pixel vision code runs several times slower.
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Vehicle model and controller: shared by the waypoint sim and Atlas Vision's
# path prediction.
add_library(atlas_core STATIC
    vehicle.cpp
    control.cpp
)
target_include_directories(atlas_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(atlas_core PRIVATE -Wall -Wextra)

add_executable(atlas
    position.cpp
    viz.cpp
    perception.cpp
    planning.cpp
    camera.cpp
)
target_compile_options(atlas PRIVATE -Wall -Wextra)
target_link_libraries(atlas PRIVATE atlas_core)

find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} — Atlas Vision enabled")
    if(OpenCV_VERSION VERSION_LESS 4.7)
        message(WARNING "Atlas Vision is tested with OpenCV 4.7+ and 5.x; ${OpenCV_VERSION} may not build")
    endif()

    add_library(atlas_vision STATIC
        vision/frame_source.cpp
        vision/camera_model.cpp
        vision/ground_view.cpp
        vision/calibration.cpp
        vision/process_stats.cpp
        vision/lane_detector.cpp
        vision/lane_tracker.cpp
        vision/ego_motion.cpp
        vision/road_geometry.cpp
        vision/horizon_estimator.cpp
        vision/object_detector.cpp
        vision/object_tracker.cpp
        vision/pipeline.cpp
        vision/dashboard.cpp
        vision/data_logger.cpp
        vision/review.cpp
    )
    target_compile_options(atlas_vision PRIVATE -Wall -Wextra)
    target_include_directories(atlas_vision PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(atlas_vision PUBLIC atlas_core ${OpenCV_LIBS})
    if(WIN32)
        target_link_libraries(atlas_vision PUBLIC psapi)
    endif()

    target_sources(atlas PRIVATE vision/app.cpp)
    target_compile_definitions(atlas PRIVATE ATLAS_HAS_OPENCV)
    target_link_libraries(atlas PRIVATE atlas_vision)

    add_executable(atlas_tests
        tests/test_main.cpp
        tests/test_geometry.cpp
        tests/test_lanes.cpp
        tests/test_tracker.cpp
        tests/test_horizon.cpp
    )
    target_compile_options(atlas_tests PRIVATE -Wall -Wextra)
    target_link_libraries(atlas_tests PRIVATE atlas_vision)
else()
    message(STATUS "OpenCV not found — sim only. Install OpenCV and reconfigure for Atlas Vision")
endif()
