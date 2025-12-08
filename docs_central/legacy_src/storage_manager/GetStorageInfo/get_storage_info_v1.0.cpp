#include <storage_manager/GetStorageInfo.hpp>
#include "rclcpp_components/register_node_macro.hpp"

namespace storage_manager 
{

// DOC-START: StorageNode
// Construtor: Inicializa o nó como Componente e carrega os arquivos de configuração.
// Este nó funciona como um banco de dados em memória, mapeando nomes de objetos (Labels)
// para locais físicos de armazenamento (Caixas/Estantes).
StorageNode::StorageNode(const rclcpp::NodeOptions & options)
: Node("storage_manager_node", options)
{
    // Declaração de parâmetros para os arquivos YAML
    this->declare_parameter<std::string>("label_to_storage_yaml_file", "");
    this->declare_parameter<std::string>("storage_poses_yaml_file", "");

    std::string label_file = this->get_parameter("label_to_storage_yaml_file").as_string();
    std::string poses_file = this->get_parameter("storage_poses_yaml_file").as_string();

    // Carrega o mapeamento Lógico (Objeto -> Nome da Caixa)
    if(!label_file.empty()) {
        loadLabelToStorage(label_file);
    } 
    else 
    {
        RCLCPP_WARN(this->get_logger(), "Param 'label_to_storage_yaml_file' is empty.");
    }

    // Carrega o mapeamento Físico (Nome da Caixa -> Pose 3D)
    if(!poses_file.empty()) {
        loadStoragePoses(poses_file);
    } 
    else 
    {
        RCLCPP_WARN(this->get_logger(), "Param 'storage_poses_yaml_file' is empty.");
    }

    RCLCPP_INFO(this->get_logger(), "StorageNode Initialized (Ready for Direct Access).");
}
// DOC-END: StorageNode

// DOC-START: getBestStorage
// Função Principal: Encontra o melhor local para guardar um objeto.
// Lógica:
// 1. Verifica quais caixas aceitam esse tipo de objeto (Label).
// 2. Filtra caixas que ainda têm espaço livre.
// 3. Seleciona a caixa mais próxima do robô (Distância Euclidiana).
StorageResult StorageNode::getBestStorage(const std::string& label, const geometry_msgs::msg::Pose& robot_pose)
{
    std::lock_guard<std::mutex> lock(mutex_); // Protege leitura do mapa

    StorageResult result;
    result.success = false;

    // Verifica se o objeto está cadastrado no sistema
    if (labels_to_storage_.find(label) == labels_to_storage_.end()) {
        RCLCPP_WARN(this->get_logger(), "Label '%s' not found in rules.", label.c_str());
        return result;
    }

    double best_dist = std::numeric_limits<double>::max();
    bool found = false;
    std::string selected_name;
    StorageInfo selected_info;

    // Lista de caixas candidatas para este objeto
    const auto& candidates = labels_to_storage_.at(label);

    for (const auto& storage_name : candidates)
    {
        if (storage_map_.find(storage_name) == storage_map_.end()) continue;

        const auto& infos = storage_map_.at(storage_name);
        
        for (const auto& info : infos)
        {
            // Verifica capacidade (-1 significa infinito)
            bool is_unlimited = (info.max_objects == -1);
            bool has_space = (info.actual_objects < info.max_objects);

            if (!is_unlimited && !has_space) 
            {
                continue; // Caixa cheia, pula para a próxima
            }

            // Calcula distância do robô até a caixa
            double dx = info.pose.position.x - robot_pose.position.x;
            double dy = info.pose.position.y - robot_pose.position.y;
            double dz = info.pose.position.z - robot_pose.position.z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            // Seleciona a mais próxima
            if (dist < best_dist) 
            {
                best_dist = dist;
                selected_name = storage_name;
                selected_info = info;
                found = true;
            }
        }
    }

    if (found) 
    {
        // Preenche o resultado com os dados da caixa vencedora
        result.success = true;
        result.storage_name = selected_name;
        result.pose = selected_info.pose;
        result.current_count = selected_info.actual_objects;
        result.max_count = selected_info.max_objects;
        result.indexes = selected_info.indexes; // Próximos índices livres (i, j, k)

        result.size.x = selected_info.size_x; 
        result.size.y = selected_info.size_y; 
        result.size.z = selected_info.size_z;                  

        // Calcula os limites geométricos da caixa para evitar colisão
        auto lims = calculateLimits(selected_info.pose, selected_info.size_x, selected_info.size_y);
        result.limits = {lims.min_x, lims.max_x, lims.min_y, lims.max_y};
        
        RCLCPP_INFO(this->get_logger(), "Selected '%s' for item '%s'. Count: %d/%d", 
            selected_name.c_str(), label.c_str(), result.current_count, result.max_count);
    } 
    else 
    {
        RCLCPP_WARN(this->get_logger(), "No valid storage found for '%s'", label.c_str());
    }

    return result;
}
// DOC-END: getBestStorage

// DOC-START: incrementStorageCount
// Atualiza o contador de ocupação de uma caixa.
// Usado para reservar espaço (amount +1) ou liberar em caso de falha (amount -1).
void StorageNode::incrementStorageCount(const std::string& storage_name, int amount)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (storage_map_.find(storage_name) != storage_map_.end()) 
    {
        for (auto& info : storage_map_[storage_name]) 
        {
            info.actual_objects += amount;
            if (info.actual_objects < 0) 
            {
                info.actual_objects = 0; // Proteção contra underflow
            }
        }
        RCLCPP_INFO(this->get_logger(), "Updated count for '%s' by %d.", storage_name.c_str(), amount);
    }
}
// DOC-END: incrementStorageCount

// DOC-START: addNewIndexes
// Atualiza os índices de grade (i, j, k) da caixa.
// Chamado pelo OrganizeNode após calcular onde o próximo objeto será colocado.
void StorageNode::addNewIndexes(const std::string& storage_name, const std::vector<int>& new_indexes)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (storage_map_.find(storage_name) != storage_map_.end()) 
    {
        for (auto& info : storage_map_[storage_name]) 
        {
            info.indexes = new_indexes; 

            RCLCPP_INFO(this->get_logger(), 
                        "Indexes atualizados para '%s'. Novo IDX: [%d, %d, %d]", 
                        storage_name.c_str(), 
                        new_indexes.size() > 0 ? new_indexes[0] : -1,
                        new_indexes.size() > 1 ? new_indexes[1] : -1,
                        new_indexes.size() > 2 ? new_indexes[2] : -1);
        }
    }
    else 
    {
        RCLCPP_WARN(this->get_logger(), "Tentativa de atualizar indexes para storage '%s' que não existe.", storage_name.c_str());
    }
}
// DOC-END: addNewIndexes

// DOC-START: loadLabelToStorage
// Carrega o arquivo YAML de regras lógicas (ex: "cola" -> ["box_red", "box_blue"]).
void StorageNode::loadLabelToStorage(const std::string &yaml_file)
{
    std::lock_guard<std::mutex> lock(mutex_);
    try 
    {
        YAML::Node config = YAML::LoadFile(yaml_file);

        for (auto it = config.begin(); it != config.end(); ++it) 
        {
            std::string group = it->first.as<std::string>();
            std::vector<std::string> targets;

            for (const auto &entry : it->second) 
            {
                if (entry["storage"]) targets.push_back(entry["storage"].as<std::string>());
            }

            labels_to_storage_[group] = targets;
        }
    } 
    catch (const YAML::Exception &e) 
    {
        RCLCPP_ERROR(this->get_logger(), "YAML Label Error: %s", e.what());
    }
}
// DOC-END: loadLabelToStorage

// DOC-START: loadStoragePoses
// Carrega o arquivo YAML de definições físicas (Posição, Tamanho, Capacidade).
void StorageNode::loadStoragePoses(const std::string &yaml_file)
{
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        YAML::Node config = YAML::LoadFile(yaml_file);
        for (auto it = config.begin(); it != config.end(); ++it) {
            std::string name = it->first.as<std::string>();
            std::vector<StorageInfo> info_list;

            for (const auto &node : it->second) {
                StorageInfo info;
                info.indexes = {0, 0, 0}; // Inicia vazio na posição (0,0,0)
                
                if (node["position"]) {
                    info.pose.position.x = node["position"][0].as<double>();
                    info.pose.position.y = node["position"][1].as<double>();
                    info.pose.position.z = node["position"][2].as<double>();
                }
                
                if (node["orientation"]) {
                    tf2::Quaternion q;
                    q.setRPY(node["orientation"][0].as<double>(), 
                             node["orientation"][1].as<double>(), 
                             node["orientation"][2].as<double>());
                    info.pose.orientation = tf2::toMsg(q);
                } else {
                    info.pose.orientation.w = 1.0;
                }

                if (node["size"]) {
                    info.size_x = node["size"][0].as<double>();
                    info.size_y = node["size"][1].as<double>();
                    info.size_z = node["size"][1].as<double>();
                } else {
                    info.size_x = 0.5; info.size_y = 0.5;
                }

                if (node["max_objects"]) {
                    info.max_objects = node["max_objects"].as<int>();
                }

                info_list.push_back(info);
            }
            storage_map_[name] = info_list;
        }
    } catch (const YAML::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "YAML Storage Error: %s", e.what());
    }
}
// DOC-END: loadStoragePoses

// DOC-START: calculateLimits
// Calcula a bounding box (min_x, max_x, etc.) da área de armazenamento no mundo.
// Considera a rotação (Yaw) da caixa para criar limites alinhados aos eixos globais.
StorageNode::StorageLimits StorageNode::calculateLimits(const geometry_msgs::msg::Pose& pose, double sx, double sy)
{
    double yaw = tf2::getYaw(pose.orientation);
    double c = std::cos(yaw);
    double s = std::sin(yaw);
    double dx = sx/2.0; 
    double dy = sy/2.0;

    // Vértices do retângulo da caixa (local)
    double lx[4] = {dx, dx, -dx, -dx};
    double ly[4] = {dy, -dy, dy, -dy};

    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();

    // Rotaciona e translada cada vértice para o sistema global
    for(int i=0; i<4; i++) 
    {
        double gx = pose.position.x + (lx[i]*c - ly[i]*s);
        double gy = pose.position.y + (lx[i]*s + ly[i]*c);
        
        if (gx < min_x) { min_x = gx; }
        if (gx > max_x) { max_x = gx; }
        
        if (gy < min_y) { min_y = gy; }
        if (gy > max_y) { max_y = gy; }
    }
    return {min_x, max_x, min_y, max_y};
}
// DOC-END: calculateLimits

} // namespace storage_manager

RCLCPP_COMPONENTS_REGISTER_NODE(storage_manager::StorageNode)