cmake_minimum_required(VERSION 3.5)
project(gazebo)

set(CMAKE_CXX_STANDARD 20)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
# add_executable(publish_transform src/publish_transform.cpp)
# ament_target_dependencies(publish_transform
#   rclcpp
#   sensor_msgs
#   tf2
#   tf2_ros
#   tf2_sensor_msgs 
#   geometry_msgs
# )

# install(TARGETS 
#   publish_transform
#   DESTINATION lib/${PROJECT_NAME}
# )

install(DIRECTORY
  models
  urdf
  worlds
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
