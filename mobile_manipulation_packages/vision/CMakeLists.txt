cmake_minimum_required(VERSION 3.8)
project(vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

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
find_package(moveit_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(fcl REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_library(vision_lib SHARED
  src/generate_scan_poses.cpp
  src/object_mapping.cpp
  src/generate_grasp_poses.cpp
)

target_include_directories(vision_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS} 
  ${fcl_INCLUDE_DIRS} 
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
  moveit_msgs   
  octomap_msgs  
  octomap
  geometric_shapes
  ament_index_cpp
)

target_link_libraries(vision_lib
  ${PCL_LIBRARIES}
  fcl 
)

add_executable(combined_semantic_pcl src/combined_semantic_pcl.cpp)
target_link_libraries(combined_semantic_pcl vision_lib) 

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

add_executable(generate_grasp_poses_main src/generate_grasp_poses_main.cpp)
target_link_libraries(generate_grasp_poses_main vision_lib) 

ament_target_dependencies(generate_grasp_poses_main
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
  rclcpp_components
  moveit_msgs   
  octomap_msgs  
  octomap
  geometric_shapes
  ament_index_cpp
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
  generate_grasp_poses_main
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
ament_export_dependencies(geometric_shapes ament_index_cpp) 

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()