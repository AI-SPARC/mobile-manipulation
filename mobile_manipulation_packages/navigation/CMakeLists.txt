cmake_minimum_required(VERSION 3.8)
project(navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- FIND DEPENDENCIES ---
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
find_package(message_filters REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(rclcpp_components REQUIRED)

# =========================================================
# 1. CRIAÇÃO DA BIBLIOTECA COMPARTILHADA (SharedObstacleGraph)
# =========================================================

add_library(navigation_lib SHARED
  src/shared_obstacle_graph.cpp 
)

target_include_directories(navigation_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(navigation_lib
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  PCL
  pcl_conversions
  rclcpp_components
)

target_link_libraries(navigation_lib
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS} 
  yaml-cpp
)

add_executable(obstacle_graph_with_occupancy_grid src/obstacle_graph_with_occupancy_grid.cpp)

ament_target_dependencies(obstacle_graph_with_occupancy_grid 
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


add_executable(a_star src/a_star.cpp)
ament_target_dependencies(a_star 
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
target_link_libraries(a_star ${GLEW_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

add_executable(controller src/controller.cpp)
ament_target_dependencies(controller 
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
target_link_libraries(controller ${GLEW_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

target_link_libraries(obstacle_graph_with_occupancy_grid
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)


install(DIRECTORY include/
  DESTINATION include
)


install(TARGETS
  navigation_lib
  EXPORT navigation_targets  
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)


install(TARGETS
  a_star
  controller
  obstacle_graph_with_occupancy_grid
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)


install(
  EXPORT navigation_targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_targets(navigation_targets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(navigation_lib)
ament_export_dependencies(rclcpp PCL pcl_conversions)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()