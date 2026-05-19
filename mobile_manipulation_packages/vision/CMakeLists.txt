cmake_minimum_required(VERSION 3.8)
project(vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # 1. Mantemos o AVX ativado para a velocidade máxima!
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
  
  # 2. Matemática rápida segura
  add_compile_options(-fno-math-errno -funsafe-math-optimizations -fno-trapping-math)
  
  # 3. A MÁGICA DA ABI: Dizemos ao Eigen para travar o alinhamento em 16 bytes.
  # Isso impede que o PCL corrompa a memória, mas mantém as instruções AVX do seu código ativas!
  add_compile_definitions(NDEBUG EIGEN_NO_DEBUG EIGEN_MAX_ALIGN_BYTES=16)
endif()
# 4. Habilitar Link-Time Optimization (LTO) / Interprocedural Optimization (IPO)
# O compilador otimiza o código inteiro na hora de "juntar" tudo, resolvendo chamadas de funções
# entre arquivos diferentes. É vital para performance em C++.
cmake_policy(SET CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported)
if(lto_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED) 
find_package(vision_msgs REQUIRED)
find_package(mobile_manipulation_interfaces REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(fcl REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(assimp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

add_library(vision_lib SHARED
  src/generate_scan_poses.cpp
  src/object_mapping.cpp
  src/generate_grasp_poses.cpp
)

target_include_directories(vision_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS} 
  ${fcl_INCLUDE_DIRS} 
  ${ASSIMP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(vision_lib
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  vision_msgs
  mobile_manipulation_interfaces
  tf2
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen
  Eigen3
  PCL
  pcl_conversions
  rclcpp_components
  moveit_msgs   
  octomap_msgs  
  octomap
  geometric_shapes
  ament_index_cpp
  assimp
  cv_bridge
)

target_link_libraries(vision_lib
  ${PCL_LIBRARIES}
  fcl 
  assimp
)

rclcpp_components_register_node(vision_lib
  PLUGIN "vision::GenerateGraspPoses"
  EXECUTABLE generate_grasp_poses_node
)

add_executable(combined_semantic_pcl src/combined_semantic_pcl.cpp)
target_link_libraries(combined_semantic_pcl vision_lib) 

target_include_directories(combined_semantic_pcl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${ASSIMP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(combined_semantic_pcl
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
  mobile_manipulation_interfaces
  PCL
  pcl_conversions
  cv_bridge
  rclcpp_components
)

add_executable(generate_grasp_poses_main src/generate_grasp_poses_main.cpp)
target_link_libraries(generate_grasp_poses_main vision_lib TBB::tbb) 

ament_target_dependencies(generate_grasp_poses_main
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
  mobile_manipulation_interfaces
  PCL
  pcl_conversions
  rclcpp_components
  moveit_msgs   
  octomap_msgs  
  octomap
  geometric_shapes
  ament_index_cpp
  assimp
  cv_bridge
)

add_executable(grasp_projector_node src/grasp_projector_node.cpp)
target_link_libraries(grasp_projector_node vision_lib) 

ament_target_dependencies(grasp_projector_node
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
  mobile_manipulation_interfaces
  PCL
  pcl_conversions
  rclcpp_components
  moveit_msgs   
  octomap_msgs  
  octomap
  geometric_shapes
  ament_index_cpp
  assimp
  cv_bridge
  vision_msgs
)
target_link_libraries(grasp_projector_node ${GLEW_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${EIGEN3_INCLUDE_DIR} yaml-cpp)

add_executable(sam2_and_rtabmap_2 src/sam2_and_rtabmap_2.cpp)
target_link_libraries(sam2_and_rtabmap_2 vision_lib) 

ament_target_dependencies(sam2_and_rtabmap_2
  rclcpp
  sensor_msgs
  std_msgs
  vision_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  tf2_eigen
  mobile_manipulation_interfaces
  PCL
  pcl_conversions
  rclcpp_components
  moveit_msgs   
  octomap_msgs  
  octomap
  geometric_shapes
  ament_index_cpp
  assimp
  cv_bridge
  vision_msgs
)


install(DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS vision_lib
  EXPORT vision_targets
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(
  TARGETS 
  combined_semantic_pcl
  generate_grasp_poses_main
  grasp_projector_node
  sam2_and_rtabmap_2
  DESTINATION lib/${PROJECT_NAME}
)

install(
  EXPORT vision_targets
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

install(
  DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(vision_targets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(vision_lib)
ament_export_dependencies(geometric_shapes ament_index_cpp) 

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()