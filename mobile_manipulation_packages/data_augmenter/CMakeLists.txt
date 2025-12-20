cmake_minimum_required(VERSION 3.8)
project(data_augmenter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fopenmp -O3) # Adicionado -fopenmp e -O3 (otimização máxima)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR})

add_executable(data_augmentation src/data_augmentation.cpp)
target_link_libraries(data_augmentation
  ${yaml-cpp_LIBRARIES}
  Eigen3::Eigen
  yaml-cpp
  gomp # Biblioteca do OpenMP
)
ament_target_dependencies(data_augmentation rclcpp)

install(TARGETS data_augmentation
  DESTINATION lib/${PROJECT_NAME})

ament_package()