cmake_minimum_required(VERSION 3.5)
project(denso_isaacsim)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(moveit REQUIRED)

install(DIRECTORY
  rviz2
  launch
  maps
  DESTINATION share/${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()