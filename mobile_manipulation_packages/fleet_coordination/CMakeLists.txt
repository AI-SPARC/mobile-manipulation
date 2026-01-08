cmake_minimum_required(VERSION 3.8)
project(fleet_coordination)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- FIND DEPENDENCIES ---
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

add_executable(fleet_management src/fleet_management.cpp)

target_include_directories(fleet_management
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${navigation_INCLUDE_DIRS} 
)

ament_target_dependencies(fleet_management
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

target_link_libraries(fleet_management
  ${navigation_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fleet_management OpenMP::OpenMP_CXX)
endif()

target_compile_features(fleet_management PUBLIC c_std_99 cxx_std_17)



add_executable(adjust_velocities_fleet_management src/adjust_velocities_fleet_management.cpp)
ament_target_dependencies(adjust_velocities_fleet_management
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  mobile_manipulation_interfaces_fleet
  tf2
  tf2_geometry_msgs
  std_msgs
  std_srvs
)

add_executable(generate_and_validate_tests src/generate_and_validate_tests.cpp)
ament_target_dependencies(generate_and_validate_tests
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  mobile_manipulation_interfaces_fleet
  tf2
  tf2_geometry_msgs
  std_msgs
  std_srvs
)

add_executable(generate_tests src/generate_tests.cpp)
ament_target_dependencies(generate_tests
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  mobile_manipulation_interfaces_fleet
  tf2
  tf2_geometry_msgs
  std_msgs
  std_srvs
)

install(TARGETS 
  fleet_management
  adjust_velocities_fleet_management
  generate_and_validate_tests
  generate_tests
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()