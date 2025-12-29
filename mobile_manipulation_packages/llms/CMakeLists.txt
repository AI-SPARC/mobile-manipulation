cmake_minimum_required(VERSION 3.8)
project(llms)

# 1. Dependências do ROS
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# 2. Dependências de Sistema (HTTP e JSON)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(bridge_to_llama src/bridge_to_llama.cpp)

# 3. Diretórios de Include
# Garante que o compilador ache o seu "include/OllamaClient.hpp"
target_include_directories(bridge_to_llama PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CURL_INCLUDE_DIRS}
)

# 4. Linkar bibliotecas do SISTEMA (CURL, JSON)
# Use target_link_libraries para coisas que NÃO são ROS
target_link_libraries(bridge_to_llama
  ${CURL_LIBRARIES}
  nlohmann_json::nlohmann_json
)

# 5. Linkar bibliotecas do ROS (A CORREÇÃO ESTÁ AQUI)
# ament_target_dependencies gerencia automaticamente os targets complexos de mensagens
ament_target_dependencies(bridge_to_llama
  rclcpp
  std_msgs
)

# 6. Instalação
install(TARGETS bridge_to_llama
  DESTINATION lib/${PROJECT_NAME})

# 7. Copiar a pasta include (opcional, mas boa prática se o header for usado externamente)
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME})

ament_package()