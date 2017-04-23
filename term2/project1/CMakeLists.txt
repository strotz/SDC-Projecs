cmake_minimum_required(VERSION 3.7)
project(project1)

include_directories(/Users/Shared/SDC/eigen-eigen-67e894c6cd8f/Eigen)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp kalman_filter.cpp kalman_filter.h tracking.cpp tracking.h measurement_package.h)
add_executable(project1 ${SOURCE_FILES})