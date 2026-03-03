cmake_minimum_required(VERSION 3.8)
project(slam_feature_matching)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()


find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(faiss REQUIRED)


set(ONNXRUNTIME_DIR "/home/momesso/onnxruntime-linux-x64-1.18.0")


add_library(${PROJECT_NAME} SHARED
  src/DinoExtractor.cpp
  src/LightGlueMatcher.cpp
  src/FaissLoopDetector.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  /usr/local/include
  ${ONNXRUNTIME_DIR}/include
)


ament_target_dependencies(${PROJECT_NAME}
  cv_bridge
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  faiss
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)


ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(cv_bridge)


install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()