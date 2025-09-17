cmake_minimum_required(VERSION 3.5)
project(panda)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(${PROJECT_NAME}_description REQUIRED)
find_package(${PROJECT_NAME}_moveit_config REQUIRED)

# Convert xacros of description and moveit_config packages to URDF, SDF and SRDF
get_filename_component(${PROJECT_NAME}_REPO_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(${PROJECT_NAME}_DESCRIPTION_SCRIPTS_DIR ${${PROJECT_NAME}_REPO_DIR}/${PROJECT_NAME}_description/scripts)
set(${PROJECT_NAME}_MOVEIT_CONFIG_SCRIPTS_DIR ${${PROJECT_NAME}_REPO_DIR}/${PROJECT_NAME}_moveit_config/scripts)
execute_process(
  COMMAND ${${PROJECT_NAME}_DESCRIPTION_SCRIPTS_DIR}/xacro2urdf.bash
)
execute_process(
  COMMAND ${${PROJECT_NAME}_DESCRIPTION_SCRIPTS_DIR}/xacro2sdf.bash
)
execute_process(
  COMMAND ${${PROJECT_NAME}_MOVEIT_CONFIG_SCRIPTS_DIR}/xacro2srdf.bash
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(visualization_msgs REQUIRED) 
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED) 

# Install directories
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

add_executable(ros2_bridge_to_pose_array src/ros2_bridge_to_pose_array.cpp)
ament_target_dependencies(
  ros2_bridge_to_pose_array 
  trajectory_msgs 
  rclcpp nav_msgs 
  geometry_msgs 
  PCL 
  pcl_conversions 
  sensor_msgs 
  tf2_geometry_msgs 
  tf2 
  tf2_ros 
  tf2_sensor_msgs 
  tf2_ros 
  tf2_eigen 
  PCL
  vision_msgs
  moveit_ros_planning_interface 
  Eigen3
)

target_link_libraries(ros2_bridge_to_pose_array
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

install(TARGETS

  ros2_bridge_to_pose_array
  DESTINATION lib/${PROJECT_NAME}
)

# Setup the project
ament_package()
