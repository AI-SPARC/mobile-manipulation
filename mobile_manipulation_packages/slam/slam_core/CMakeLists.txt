cmake_minimum_required(VERSION 3.8)
project(slam_core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(rtabmap_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(faiss REQUIRED)


set(ONNXRUNTIME_DIR "/home/momesso/onnxruntime-linux-x64-1.18.0")

add_library(dino_loop_component SHARED
  src/dino_loop_node.cpp
)

target_include_directories(dino_loop_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  /usr/local/include
  ${ONNXRUNTIME_DIR}/include
)

ament_target_dependencies(dino_loop_component
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  cv_bridge
  OpenCV
)

target_link_libraries(dino_loop_component
  faiss
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

rclcpp_components_register_node(dino_loop_component
  PLUGIN "slam_core::DinoLoopNode"
  EXECUTABLE dino_loop_node_exe
)

add_executable(slam_core
  src/slam_core.cpp
)

target_include_directories(slam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  /usr/local/include
  ${ONNXRUNTIME_DIR}/include
)

ament_target_dependencies(slam_core
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

target_link_libraries(slam_core
  dino_loop_component
  ${OpenCV_LIBRARIES}
  gtsam 
  Eigen3::Eigen
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

install(TARGETS dino_loop_component
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS slam_core
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()