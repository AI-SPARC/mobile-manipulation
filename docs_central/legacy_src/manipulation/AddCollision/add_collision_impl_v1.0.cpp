#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <array>

#include <manipulation/AddCollision.hpp>

#include "yaml-cpp/yaml.h"

using namespace std::chrono_literals;

namespace manipulation {

// DOC-START: AddCollision
// Construtor: Inicializa o nó, carrega regras de filtro (YAML) e configura a comunicação.
AddCollision::AddCollision()
 : Node("add_collision_objects")
{
    // Carrega parâmetro do arquivo YAML que define o que é obstáculo e o que não é.
    this->declare_parameter<std::string>("yaml_file", "");
    std::string labels_path = this->get_parameter("yaml_file").as_string();

    load_labels_from_yaml(labels_path);

    // Assina as detecções 3D (Bounding Boxes) do sistema de visão.
    sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
        "/boxes_detection_array", 10,
        std::bind(&AddCollision::detectionCallback, this, std::placeholders::_1));
    
    // Serviço para "congelar" a atualização de um obstáculo específico (ex: quando o robô vai pegá-lo).
    service_ = this->create_service<mobile_manipulation_interfaces::srv::MobileObjectCollision>(
        "/object_collision",
        std::bind(&AddCollision::handleStopService, this, std::placeholders::_1, std::placeholders::_2));

    // Adiciona o chão (Ground Plane) após 2 segundos para dar tempo do MoveIt carregar.
    init_timer_ = this->create_wall_timer(
        std::chrono::seconds(2), 
        [this]() {
            this->add_ground_plane();
            this->init_timer_->cancel(); 
        });        
}   
// DOC-END: AddCollision

// DOC-START: load_labels_from_yaml
// Carrega listas de "Permitidos" e "Proibidos" do arquivo YAML.
// Isso filtra quais objetos viram colisão no MoveIt (ex: ignorar pessoas, adicionar mesas).
void AddCollision::load_labels_from_yaml(const std::string& file_path)
{
    std::ifstream f(file_path.c_str());
    if (!f.good()) {
        RCLCPP_WARN(this->get_logger(), "YAML não encontrado ou vazio: %s", file_path.c_str());
        return;
    }
    try {
        YAML::Node config = YAML::LoadFile(file_path);
        // Helper lambda para carregar vetores de regras
        auto load_rules = [&](const YAML::Node& node, std::vector<LabelRule>& target) {
            for (const auto& label_node : node) {
                std::string label = label_node.as<std::string>();
                // Se terminar com '_', é tratado como prefixo (ex: "box_" pega "box_1", "box_2")
                bool is_prefix = (!label.empty() && label.back() == '_');
                target.push_back({label, is_prefix});
            }
        };
        if (config["authorized_labels"]) load_rules(config["authorized_labels"], authorized_labels_);
        if (config["unauthorized_labels"]) load_rules(config["unauthorized_labels"], unauthorized_labels_);
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Erro parsing YAML: %s", e.what());
    }
}
// DOC-END: load_labels_from_yaml

// DOC-START: add_ground_plane
// Adiciona um plano de chão fixo na cena do MoveIt para evitar que o braço colida com o solo.
void AddCollision::add_ground_plane()
{
    moveit_msgs::msg::CollisionObject ground;
    ground.id = "ground_plane";
    ground.header.frame_id = "world"; // Referência global
    
    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    // Dimensões grandes (20x20m) mas finas (1cm)
    primitive.dimensions = {20.0, 20.0, 0.01}; 
    
    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;
    
    ground.primitives.push_back(primitive);
    ground.primitive_poses.push_back(pose);
    ground.operation = ground.ADD;
    
    // Envia para a Interface de Cena do MoveIt (PSI)
    planning_scene_interface.applyCollisionObjects({ground});
    RCLCPP_INFO(this->get_logger(), "Ground Plane adicionado à cena.");
}
// DOC-END: add_ground_plane

// DOC-START: is_significant_change
// Filtro de histerese espacial: Só atualiza a posição do objeto no MoveIt se ele se moveu mais que 5mm.
// Isso evita spam de mensagens na Planning Scene Interface (PSI) causado por ruído na detecção.
bool AddCollision::is_significant_change(const std::string& id, const geometry_msgs::msg::Pose& new_pose)
{
    if (last_known_poses_.find(id) == last_known_poses_.end()) return true; // Novo objeto

    const auto& old_pose = last_known_poses_[id];
    
    // Distância Euclidiana 3D
    double dist = std::sqrt(
        std::pow(new_pose.position.x - old_pose.position.x, 2) +
        std::pow(new_pose.position.y - old_pose.position.y, 2) +
        std::pow(new_pose.position.z - old_pose.position.z, 2)
    );

    return (dist > 0.005); // Limiar de 5mm
}
// DOC-END: is_significant_change

// DOC-START: add_collision_box
// Cria um novo objeto de colisão no MoveIt pela primeira vez.
void AddCollision::add_collision_box(const std::string &id, const std::array<double, 3> &dimensions, const geometry_msgs::msg::Pose &pose)
{
    // Verifica cache local para não readicionar
    if (added.find(id) != added.end()) return;

    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.id = id;
    collision_object.header.frame_id = "world"; 

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {dimensions[0], dimensions[1], dimensions[2]};

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(pose);
    collision_object.operation = collision_object.ADD; // Operação de Adição

    planning_scene_interface.applyCollisionObjects({collision_object});
    
    added.insert(id);
    last_known_poses_[id] = pose; 
    
    RCLCPP_INFO(this->get_logger(), "Objeto adicionado: %s", id.c_str());
}
// DOC-END: add_collision_box

// DOC-START: move_collision_box
// Atualiza a pose de um objeto existente no MoveIt.
// Usado para rastrear objetos que estão sendo empurrados ou movidos.
void AddCollision::move_collision_box(const std::string &id, const geometry_msgs::msg::Pose &pose)
{
    if (!is_significant_change(id, pose)) return;

    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.id = id;
    collision_object.header.frame_id = "world";
    collision_object.primitive_poses.push_back(pose);
    collision_object.operation = collision_object.MOVE; // Operação de Movimento

    planning_scene_interface.applyCollisionObjects({collision_object});
    
    last_known_poses_[id] = pose; 
}
// DOC-END: move_collision_box

// DOC-START: is_authorized
// Verifica se a label detectada (ex: "coke_can") deve ser inserida na cena baseada nas regras do YAML.
bool AddCollision::is_authorized(const std::string& label)
{
    // 1. Verifica lista negra (unauthorized)
    for (const auto& rule : unauthorized_labels_) 
        if ((rule.is_prefix && label.rfind(rule.label, 0) == 0) || (!rule.is_prefix && label == rule.label)) 
            return false; 

    if (authorized_labels_.empty()) return true; // Se lista branca vazia, permite tudo (menos proibidos)

    // 2. Verifica lista branca (authorized)
    for (const auto& rule : authorized_labels_) 
        if ((rule.is_prefix && label.rfind(rule.label, 0) == 0) || (!rule.is_prefix && label == rule.label)) 
            return true;

    return false;
}
// DOC-END: is_authorized

// DOC-START: detectionCallback
// Callback principal: Recebe detecções, filtra e chama Add ou Move.
void AddCollision::detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
{
    if (msg->detections.empty()) return;

    for (const auto &det : msg->detections)
    {
        if (det.results.empty()) continue;

        std::string object_id = det.results[0].hypothesis.class_id;
        if (!is_authorized(object_id)) continue;

        // Ajuste de Z: MoveIt usa o centro geométrico, enquanto algumas detecções podem vir na base.
        // Aqui assumimos que 'bbox.center' já é o centro geométrico, mas fazemos um ajuste fino se necessário.
        geometry_msgs::msg::Pose pose = det.bbox.center;
        pose.position.z += det.bbox.size.z / 2.0; 

        std::array<double, 3> size_array = {det.bbox.size.x, det.bbox.size.y, det.bbox.size.z};

        // Se é novo, adiciona. Se já existe, move.
        if (added.find(object_id) == added.end()) 
        {
            add_collision_box(object_id, size_array, pose);
        } 
        else 
        {
            // Lógica de Congelamento:
            // Se o objeto for o alvo atual da manipulação ('stop_moving_obstacle'), 
            // só atualizamos sua posição se 'activate_movement' for true.
            // Isso evita que o objeto "pule" no MoveIt no momento exato em que o robô vai pegá-lo.
            if (object_id == stop_moving_obstacle)
            {
                if (activate_movement) move_collision_box(object_id, pose);
            }
            else 
            {
                move_collision_box(object_id, pose);
            }
        }
    }
}
// DOC-END: detectionCallback

// DOC-START: handleStopService
// Serviço chamado pelo ServerNode para travar a posição de um objeto antes do Pick.
void AddCollision::handleStopService(
    const std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Request> request,
    std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Response> response)
{
    stop_moving_obstacle = request->obstacle_id;
    activate_movement = request->activate_movement;
    
    RCLCPP_INFO(this->get_logger(), "Serviço Move: ID '%s' -> Ativo: %d", 
        stop_moving_obstacle.c_str(), activate_movement);
    
    response->success = true;
}
// DOC-END: handleStopService

} // namespace manipulation