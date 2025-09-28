cmake_minimum_required(VERSION 3.8)
project(object_manipulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
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
find_package(object_manipulation_interfaces REQUIRED)
find_package(control_msgs REQUIRED)

add_executable(basic_algorithm src/basic_algorithm.cpp)

ament_target_dependencies(basic_algorithm 
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

add_executable(multiple_objects src/multiple_objects.cpp)

ament_target_dependencies(
  multiple_objects 
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
  object_manipulation_interfaces
  control_msgs
  Eigen3
)

add_executable(add_colision_objects src/add_colision_objects.cpp)

ament_target_dependencies(
  add_colision_objects 
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
  object_manipulation_interfaces 
  Eigen3
)

add_executable(evaluate_reward src/evaluate_reward.cpp)

ament_target_dependencies(
  evaluate_reward 
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
  object_manipulation_interfaces 
  Eigen3
)

add_executable(rounded_points src/rounded_points.cpp)

ament_target_dependencies(
  rounded_points 
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
  object_manipulation_interfaces 
  Eigen3
)


target_link_libraries(basic_algorithm
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

target_link_libraries(multiple_objects
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

target_link_libraries(add_colision_objects
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

target_link_libraries(evaluate_reward
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

target_link_libraries(rounded_points
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  basic_algorithm
  multiple_objects
  add_colision_objects
  evaluate_reward
  rounded_points
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
