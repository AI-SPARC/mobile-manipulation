cmake_minimum_required(VERSION 3.8)
project(drl_to_pick_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(msgpack REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)

# Library
add_library(${PROJECT_NAME} SHARED
  src/bridge_to_inference.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${msgpack_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_components
  geometry_msgs
)

target_link_libraries(${PROJECT_NAME}
  ${msgpack_LIBRARIES}
  ${PCL_LIBRARIES}
)

# Register composable node
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN "drl_to_pick_cpp::BridgeToInference"
  EXECUTABLE bridge_to_inference
)

# Install headers
install(
  DIRECTORY include/
  DESTINATION include
)

# Install library
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Export targets
install(
  EXPORT ${PROJECT_NAME}_targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

# Export for other packages
ament_export_targets(${PROJECT_NAME}_targets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
  rclcpp
  rclcpp_components
  geometry_msgs
  Eigen3
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()