cmake_minimum_required(VERSION 3.8)
project(fleet_task_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
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
find_package(rclcpp_components REQUIRED)
find_package(OpenMP REQUIRED)
find_package(mobile_manipulation_interfaces REQUIRED)
find_package(mobile_manipulation_interfaces_fleet REQUIRED)
find_package(navigation REQUIRED) 

add_executable(fleet_server_node src/fleet_server_node.cpp)

ament_target_dependencies(fleet_server_node
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  mobile_manipulation_interfaces
  mobile_manipulation_interfaces_fleet
  tf2
  tf2_geometry_msgs
  std_msgs
  std_srvs
  navigation
)
target_include_directories(fleet_server_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_features(fleet_server_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS 
fleet_server_node
  DESTINATION lib/${PROJECT_NAME})

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
