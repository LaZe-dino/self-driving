cmake_minimum_required(VERSION 3.16)
project(atlas_autonomy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(atlas
    position.cpp
    vehicle.cpp
    control.cpp
    viz.cpp
    perception.cpp
    planning.cpp
    camera.cpp
    webcam.cpp
)

target_compile_options(atlas PRIVATE -Wall -Wextra)

find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} — ./atlas --webcam enabled")
    target_compile_definitions(atlas PRIVATE ATLAS_HAS_OPENCV)
    target_include_directories(atlas PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(atlas PRIVATE ${OpenCV_LIBS})
else()
    message(STATUS "OpenCV not found — sim only. Install OpenCV and reconfigure for --webcam")
endif()
