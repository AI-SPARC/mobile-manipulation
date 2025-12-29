cmake_minimum_required(VERSION 3.8)
project(llms)

# ==========================================
# 1. DEPENDÊNCIAS
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(CURL REQUIRED)           
find_package(nlohmann_json REQUIRED)   
find_package(SQLite3 REQUIRED)         

# ==========================================
# 2. EXECUTÁVEL: bridge_to_phi_3.5 (Cliente Ollama)
# ==========================================
add_executable(bridge_to_phi_3.5 src/bridge_to_phi_3.5.cpp)

target_include_directories(bridge_to_phi_3.5 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CURL_INCLUDE_DIRS}
)

target_link_libraries(bridge_to_phi_3.5
  ${CURL_LIBRARIES}
  nlohmann_json::nlohmann_json
)

ament_target_dependencies(bridge_to_phi_3.5
  rclcpp
  std_msgs
)

# ==========================================
# 3. BIBLIOTECA: world_state_node (Banco de Dados)
# ==========================================
# Esta é a lib que você quer usar no outro pacote
add_library(world_state_node SHARED
  src/world_state_node.cpp
)

# Adiciona o diretório include para a biblioteca
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

# Registra o componente para uso via Composition ou ros2 run
rclcpp_components_register_node(world_state_node
  PLUGIN "llms::WorldStateNode"
  EXECUTABLE world_state_server
)

# ==========================================
# 4. EXPORTAÇÃO E INSTALAÇÃO (CRUCIAL)
# ==========================================

# Instala o executável da ponte (Ollama)
install(TARGETS bridge_to_phi_3.5
  DESTINATION lib/${PROJECT_NAME}
)

# Instala a biblioteca do componente E EXPORTA O TARGET
# O nome "export_llms" cria o vínculo para outros pacotes
install(TARGETS world_state_node
  EXPORT export_llms
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Instala os headers (hpp)
install(DIRECTORY include/
  DESTINATION include
)

# --- AMENT EXPORTS ---
# Isso avisa o CMake do outro pacote onde estão os headers
ament_export_include_directories(include)

# Isso avisa o outro pacote quais libs ele precisa carregar junto
ament_export_dependencies(
  rclcpp
  rclcpp_components
  std_msgs
  vision_msgs
  SQLite3
  nlohmann_json
)

# Isso exporta o target "world_state_node" para ser visível como "llms::world_state_node"
ament_export_targets(export_llms HAS_LIBRARY_TARGET)

ament_package()