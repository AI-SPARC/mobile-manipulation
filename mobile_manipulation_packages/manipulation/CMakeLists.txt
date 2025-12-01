cmake_minimum_required(VERSION 3.8)
project(manipulation)

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

find_package(moveit_ros_planning_interface REQUIRED)
find_package(mobile_manipulation_interfaces REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(message_filters REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(yaml-cpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)


add_library(manipulation_lib SHARED
  src/add_collision_impl.cpp
  src/manipulation_impl.cpp
  src/is_gripper_holding_impl.cpp
)

target_include_directories(manipulation_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(manipulation_lib
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  trajectory_msgs
  moveit_ros_planning_interface
  mobile_manipulation_interfaces
  vision_msgs
  message_filters

  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen

  Eigen3
  PCL
  pcl_conversions
  behaviortree_cpp
)

target_link_libraries(manipulation_lib
  ${GLEW_LIBRARIES}
  ${OpenGL_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
  ${PCL_LIBRARIES}
)


add_executable(add_collision src/add_collision_main.cpp)
add_executable(manipulation src/manipulation_main.cpp)
add_executable(is_gripper_holding src/is_gripper_holding_main.cpp)

foreach(target add_collision manipulation is_gripper_holding)
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${target} PUBLIC manipulation_lib)
endforeach()


install(DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
    manipulation_lib
    add_collision
    manipulation
    is_gripper_holding
  EXPORT manipulation_targets
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(
  EXPORT manipulation_targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)


ament_export_targets(manipulation_targets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(manipulation_lib)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
