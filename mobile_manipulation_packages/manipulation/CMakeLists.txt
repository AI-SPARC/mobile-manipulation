cmake_minimum_required(VERSION 3.8)
project(manipulation)

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
find_package(mobile_manipulation_interfaces REQUIRED)
find_package(control_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(behaviortree_cpp REQUIRED)

add_executable(add_collision src/add_collision.cpp)

ament_target_dependencies(add_collision 
  trajectory_msgs 
  rclcpp 
  nav_msgs 
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
  std_msgs
  moveit_ros_planning_interface 
  Eigen3
  mobile_manipulation_interfaces
)

add_executable(simple_manipulation_node src/simple_manipulation_node.cpp)

ament_target_dependencies(simple_manipulation_node 
  trajectory_msgs 
  rclcpp 
  nav_msgs 
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
  std_msgs
  moveit_ros_planning_interface 
  Eigen3
  mobile_manipulation_interfaces
)

add_executable(is_gripper_holding src/is_gripper_holding.cpp)

ament_target_dependencies(is_gripper_holding 
  trajectory_msgs 
  rclcpp 
  nav_msgs 
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
  std_msgs
  moveit_ros_planning_interface 
  Eigen3
  mobile_manipulation_interfaces
)

target_link_libraries(add_collision
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

target_link_libraries(simple_manipulation_node
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)

target_link_libraries(is_gripper_holding
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)

install(TARGETS
  add_collision
  simple_manipulation_node
  is_gripper_holding
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
