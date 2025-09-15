cmake_minimum_required(VERSION 3.8)
project(drl)

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
find_package(Eigen3 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(rounded_points src/rounded_points.cpp)
ament_target_dependencies(rounded_points rclcpp nav_msgs geometry_msgs PCL pcl_conversions sensor_msgs tf2_geometry_msgs tf2 tf2_ros tf2_sensor_msgs tf2_ros tf2_eigen PCL Eigen3)

add_executable(train_model src/train_model.cpp)
ament_target_dependencies(train_model rclcpp nav_msgs geometry_msgs PCL pcl_conversions sensor_msgs tf2_geometry_msgs tf2 tf2_ros tf2_sensor_msgs tf2_ros tf2_eigen PCL Eigen3)

target_link_libraries(rounded_points
  ${rclcpp_LIBRARIES}
  ${std_msgs_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

target_link_libraries(train_model
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
  
  rounded_points
  train_model
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
