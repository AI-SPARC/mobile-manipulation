cmake_minimum_required(VERSION 3.8)
project(vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# --- DEPENDENCIAS ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED) 
find_package(vision_msgs REQUIRED)
find_package(mobile_manipulation_interfaces REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp_components REQUIRED)


add_library(vision_lib SHARED
  src/generate_scan_poses.cpp
  src/object_mapping.cpp
)

target_include_directories(vision_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS} 
)

ament_target_dependencies(vision_lib
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  vision_msgs
  mobile_manipulation_interfaces
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen
  Eigen3
  PCL
  pcl_conversions
  rclcpp_components
)

target_link_libraries(vision_lib
  ${PCL_LIBRARIES}
)


add_executable(combined_semantic_pcl src/combined_semantic_pcl.cpp)
target_link_libraries(combined_semantic_pcl vision_lib) 

add_executable(generate_grasp_poses src/generate_grasp_poses.cpp)
target_link_libraries(generate_grasp_poses vision_lib) 

add_executable(generate_grasp_poses_test src/generate_grasp_poses_test.cpp)
target_link_libraries(generate_grasp_poses_test vision_lib) 

target_include_directories(combined_semantic_pcl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)

ament_target_dependencies(combined_semantic_pcl
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
  mobile_manipulation_interfaces
  PCL
  pcl_conversions
  rclcpp_components
)

ament_target_dependencies(generate_grasp_poses
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
  mobile_manipulation_interfaces
  PCL
  pcl_conversions
  rclcpp_components
)


ament_target_dependencies(generate_grasp_poses_test
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
  mobile_manipulation_interfaces
  PCL
  pcl_conversions
  rclcpp_components
)

install(DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS vision_lib
  EXPORT vision_targets
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(
  TARGETS 
  combined_semantic_pcl
  generate_grasp_poses
  generate_grasp_poses_test
  DESTINATION lib/${PROJECT_NAME}
)

install(
  EXPORT vision_targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

install(
  DIRECTORY
  launch
 
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(vision_targets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(vision_lib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()