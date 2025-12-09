cmake_minimum_required(VERSION 3.8)
project(task_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)

find_package(moveit_ros_planning_interface REQUIRED)
find_package(mobile_manipulation_interfaces REQUIRED)
find_package(behaviortree_cpp REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(manipulation REQUIRED)
find_package(storage_manager REQUIRED) 
find_package(navigation REQUIRED)

message(STATUS "storage_manager_INCLUDE_DIRS = ${storage_manager_INCLUDE_DIRS}")
message(STATUS "storage_manager_LIBRARIES    = ${storage_manager_LIBRARIES}")

add_executable(server_node src/server_node.cpp)

target_include_directories(server_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${manipulation_INCLUDE_DIRS}
    ${navigation_INCLUDE_DIRS}
    ${storage_manager_INCLUDE_DIRS}  
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  ${manipulation_LIBRARY_DIRS}
  ${storage_manager_LIBRARY_DIRS} 
)

ament_target_dependencies(server_node
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen
  pcl_conversions
  trajectory_msgs
  vision_msgs
  message_filters
  moveit_ros_planning_interface
  mobile_manipulation_interfaces
  behaviortree_cpp
  Eigen3
  storage_manager 
)

target_link_libraries(server_node
  ${PCL_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${OpenGL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  yaml-cpp
  ${manipulation_LIBRARIES}
  ${storage_manager_LIBRARIES} 
  ${navigation_LIBRARIES} 
)

install(
  TARGETS server_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY bt
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()