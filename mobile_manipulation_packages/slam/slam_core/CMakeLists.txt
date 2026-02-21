cmake_minimum_required(VERSION 3.8)
project(slam_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(rtabmap_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(visualization_msgs REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${DBoW3_INCLUDE_DIRS}
  /usr/local/include
)

add_executable(flann_feature_extractor src/flann_feature_extractor.cpp)

ament_target_dependencies(flann_feature_extractor
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  vision_msgs
  rtabmap_msgs
  cv_bridge
  visualization_msgs
)

target_link_libraries(flann_feature_extractor
  ${OpenCV_LIBRARIES}
  ${DBoW3_LIBRARIES}
  gtsam 
  Eigen3::Eigen
)

install(TARGETS 
  flann_feature_extractor 
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()