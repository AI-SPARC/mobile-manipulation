cmake_minimum_required(VERSION 3.8)
project(llms)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(CURL REQUIRED)           
find_package(nlohmann_json REQUIRED)   
find_package(SQLite3 REQUIRED)         
find_package(behaviortree_cpp REQUIRED)


add_executable(bridge_to_phi_3_5 src/bridge_to_phi_3_5.cpp)

target_include_directories(bridge_to_phi_3_5 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CURL_INCLUDE_DIRS}
)

target_link_libraries(bridge_to_phi_3_5
  ${CURL_LIBRARIES}
  nlohmann_json::nlohmann_json
  SQLite::SQLite3
)

ament_target_dependencies(bridge_to_phi_3_5
  rclcpp
  std_msgs
  behaviortree_cpp
)


add_library(world_state_node SHARED
  src/world_state_node.cpp
)

target_include_directories(world_state_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(world_state_node
  rclcpp_components::component 
  SQLite::SQLite3             
  nlohmann_json::nlohmann_json
)

ament_target_dependencies(world_state_node
  rclcpp
  rclcpp_components
  std_msgs
  vision_msgs
)

rclcpp_components_register_node(world_state_node
  PLUGIN "llms::WorldStateNode"
  EXECUTABLE world_state_server
)


install(TARGETS bridge_to_phi_3_5
  DESTINATION lib/${PROJECT_NAME}
)


install(TARGETS world_state_node
  EXPORT export_llms
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)


ament_export_include_directories(include)

ament_export_dependencies(
  rclcpp
  rclcpp_components
  std_msgs
  vision_msgs
  SQLite3
  nlohmann_json
)

ament_export_targets(export_llms HAS_LIBRARY_TARGET)

ament_package()