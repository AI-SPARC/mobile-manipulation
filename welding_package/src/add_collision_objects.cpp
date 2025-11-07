/**
 * @file add_collision_objects.cpp
 * @brief Nó ROS2 responsável por adicionar objetos de colisão no ambiente MoveIt com base em detecções 3D.
 *
 * Este nó escuta mensagens de detecção 3D (topic `/boxes_detection_array`), 
 * verifica se os objetos detectados estão autorizados de acordo com um arquivo YAML,
 * e os adiciona (ou atualiza) como objetos de colisão na cena MoveIt.
 *
 * O nó também adiciona um plano de chão e pode inicializar grupos de movimento (braço e garra).
 *
 * @version 1.0
 * @date 07-11-2025
 * @author Lucas Momesso
 */
#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "yaml-cpp/yaml.h"
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include "object_manipulation_interfaces/srv/object_collision.hpp"

using namespace std::chrono_literals;

namespace std 
{
    template <>
    struct hash<std::tuple<float, float, float>> 
    {
        size_t operator()(const std::tuple<float, float, float>& t) const 
        {
            size_t h1 = hash<float>()(std::get<0>(t));
            size_t h2 = hash<float>()(std::get<1>(t));
            size_t h3 = hash<float>()(std::get<2>(t));
            
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

namespace std {
    template<>
    struct hash<std::tuple<std::pair<int, int>, bool>> {
        size_t operator()(const std::tuple<std::pair<int, int>, bool>& t) const {
            const auto& p = std::get<0>(t);
            bool b = std::get<1>(t);
            size_t h1 = std::hash<int>{}(p.first);
            size_t h2 = std::hash<int>{}(p.second);
            size_t h3 = std::hash<bool>{}(b);
            size_t seed = h1;
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}

template <typename T1, typename T2>
struct pair_hash {
    std::size_t operator ()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);  
    }
};

template<typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, const std::tuple<T1, T2, T3>& t) {
    os << "(" << std::get<0>(t) << ", " 
       << std::get<1>(t) << ", " 
       << std::get<2>(t) << ")";
    return os;
}

struct TupleHash {
    std::size_t operator()(const std::tuple<float, float, float>& t) const {
        auto h1 = std::hash<float>{}(std::get<0>(t));
        auto h2 = std::hash<float>{}(std::get<1>(t));
        auto h3 = std::hash<float>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct TupleEqual {
    bool operator()(const std::tuple<float,float,float>& a,
                    const std::tuple<float,float,float>& b) const noexcept {
        return std::get<0>(a) == std::get<0>(b) &&
               std::get<1>(a) == std::get<1>(b) &&
               std::get<2>(a) == std::get<2>(b);
    }
};

/**
 * @struct LabelRule
 * @brief Estrutura para armazenar regras de labels (autorizadas ou não).
 *
 * Cada regra contém um rótulo (`label`) e um indicador se esse rótulo é um prefixo (`is_prefix`).
 * Se `is_prefix` for verdadeiro, qualquer label que começar com esse prefixo será considerada correspondente.
 */
struct LabelRule {
    std::string label;
    bool is_prefix;
};

/**
 * @class AddCollision
 * @brief Classe principal do nó responsável por adicionar objetos de colisão no ambiente MoveIt.
 *
 * Esta classe herda de `rclcpp::Node` e implementa:
 * - Assinatura de mensagens de detecção 3D.
 * - Leitura de labels autorizadas e não autorizadas de um arquivo YAML.
 * - Adição e atualização de objetos de colisão na cena MoveIt.
 * - Inicialização de MoveGroupInterface para manipulação de grupos robóticos.
 */
class AddCollision : public rclcpp::Node {

private:

    // Subscriptions.
    /**
     * @brief Assinatura para mensagens de detecção 3D (topic: `/boxes_detection_array`).
     */
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;

     /**
     * @brief Interface para manipulação de objetos de colisão no MoveIt.
     */
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    /**
     * @brief Ponteiros para os grupo do braço no Moveit.
     */
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;

    /**
     * @brief Timer usado para tentar inicializar o MoveGroupInterface até que ele esteja disponível.
     */
    rclcpp::TimerBase::SharedPtr init_timer_;
    
    /**
     * @brief Nome do grupo MoveIt usado (padrão: "denso_arm").
     */
    std::string move_group;
    
    /**
     * @brief Conjunto de IDs de objetos já adicionados à cena, para evitar duplicações.
     */
    std::unordered_set<std::string> added;

     /**
     * @brief Vetores com labels autorizadas e não autorizadas.
     */
    std::vector<LabelRule> authorized_labels_;
    std::vector<LabelRule> unauthorized_labels_;


    /**
     * @brief Lê as listas de labels autorizadas e não autorizadas de um arquivo YAML.
     * @param file_path Caminho do arquivo YAML contendo os labels.
     */
    void load_labels_from_yaml(const std::string& file_path)
    {
        std::ifstream f(file_path.c_str());
        if (!f.good()) {
            RCLCPP_ERROR(this->get_logger(), "Arquivo YAML de labels não encontrado em: %s", file_path.c_str());
            return;
        }

        try {
            YAML::Node config = YAML::LoadFile(file_path);

            auto load_rules = [&](const YAML::Node& node, std::vector<LabelRule>& target) {
                for (const auto& label_node : node) {
                    std::string label = label_node.as<std::string>();
                    bool is_prefix = false;

                    // Se termina com '_', marcar como prefixo
                    if (!label.empty() && label.back() == '_') {
                        is_prefix = true;
                    }

                    target.push_back({label, is_prefix});
                }
            };

            if (config["authorized_labels"]) {
                load_rules(config["authorized_labels"], authorized_labels_);
                RCLCPP_INFO(this->get_logger(), "%zu labels autorizados carregados.", authorized_labels_.size());
            }

            if (config["unauthorized_labels"]) {
                load_rules(config["unauthorized_labels"], unauthorized_labels_);
                RCLCPP_INFO(this->get_logger(), "%zu labels não autorizados carregados.", unauthorized_labels_.size());
            }

        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Erro ao processar o arquivo YAML: %s", e.what());
        }
    }

    /**
     * @brief Inicializa o grupo MoveIt do braço (tenta várias vezes até conseguir).
     */
    void initMoveGroup() {
        try {
            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), move_group);  

            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface inicializado com sucesso.");

            init_timer_->cancel();  
        } catch (const std::exception &e) 
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }
    }

    /**
     * @brief Adiciona o plano de chão como objeto de colisão na cena.
     */
    void add_ground_plane()
    {
        moveit_msgs::msg::CollisionObject ground;
        ground.id = "ground_plane";
        ground.header.frame_id = "world";

        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions = {10.0, 10.0, 0.01}; 

        geometry_msgs::msg::Pose pose;
        pose.position.x = 0.0;
        pose.position.y = 0.0;
        pose.position.z = 0.0;  
        pose.orientation.w = 1.0;

        ground.primitives.push_back(primitive);
        ground.primitive_poses.push_back(pose);
        ground.operation = ground.ADD;

        planning_scene_interface.applyCollisionObjects({ground});
    }

     /**
     * @brief Adiciona uma caixa de colisão na cena MoveIt.
     * @param id Identificador único do objeto.
     * @param dimensions Dimensões da caixa [x, y, z].
     * @param pose Posição e orientação da caixa no frame "world".
     */
    void add_collision_box(const std::string &id,const std::array<double, 3> &dimensions, const geometry_msgs::msg::Pose &pose)
    {
        std::vector<std::string> known_objects = planning_scene_interface.getKnownObjectNames();
        if (std::find(known_objects.begin(), known_objects.end(), id) != known_objects.end()) 
        {
            return;
        }

        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.id = id;
        collision_object.header.frame_id = "world";

        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions = {dimensions[0], dimensions[1], dimensions[2]};

        collision_object.primitives.push_back(primitive);
        collision_object.primitive_poses.push_back(pose);
        collision_object.operation = collision_object.ADD;

        planning_scene_interface.applyCollisionObjects({collision_object});
    }

    /**
     * @brief Move (atualiza a posição) de uma caixa de colisão existente.
     * @param id Identificador único do objeto.
     * @param pose Nova pose do objeto.
     */
    void move_collision_box(const std::string &id, const geometry_msgs::msg::Pose &pose)
    {
        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.id = id;
        collision_object.header.frame_id = "world";

        collision_object.primitive_poses.push_back(pose);
        collision_object.operation = collision_object.MOVE;

        planning_scene_interface.applyCollisionObjects({collision_object});
    }

    /**
     * @brief Verifica se um label é autorizado de acordo com as listas carregadas.
     * @param label Nome do label do objeto detectado.
     * @return `true` se o label for autorizado, `false` caso contrário.
     */
    bool is_authorized(const std::string& label)
    {
        for (const auto& rule : unauthorized_labels_) 
        {
            if ((rule.is_prefix && label.rfind(rule.label, 0) == 0) || (!rule.is_prefix && label == rule.label)) 
            {
                return false; 
            }
        }

        if (authorized_labels_.empty()) return true;

        for (const auto& rule : authorized_labels_) 
        {
            if ((rule.is_prefix && label.rfind(rule.label, 0) == 0) || (!rule.is_prefix && label == rule.label)) 
            {
                return true;
            }
        }

        return false;
    }

    /**
     * @brief Callback acionado ao receber uma nova mensagem de detecção 3D.
     *
     * Verifica cada objeto detectado, testa se o label é autorizado e adiciona/move 
     * a caixa de colisão correspondente na cena MoveIt.
     *
     * @param msg Ponteiro para a mensagem `vision_msgs::msg::Detection3DArray` recebida.
     */
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        if (msg->detections.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Detection3DArray vazio recebido.");
            return;
        }

        for (size_t i = 0; i < msg->detections.size(); ++i)
        {
            const auto &det = msg->detections[i];
            std::string object_id = det.results[0].hypothesis.class_id;

            if (!is_authorized(object_id)) continue;

            geometry_msgs::msg::Pose pose = det.bbox.center;
            pose.position.z += det.bbox.size.z / 2;

            std::array<double, 3> size_array = {
                det.bbox.size.x,
                det.bbox.size.y,
                det.bbox.size.z
            };

            if(added.find(object_id) == added.end())
            {
                add_collision_box(object_id, size_array, pose);
                added.insert(object_id);
            }      
            else
            {
                move_collision_box(object_id, pose);
            }  
        }
    }



public:
        /**
     * @brief Construtor da classe AddCollision.
     * 
     * @details
     * Inicializa o nó ROS2 responsável por **gerenciar objetos de colisão no ambiente MoveIt2**
     * com base nas **detecções 3D recebidas de um tópico**.  
     * Este nó integra dados de percepção com o sistema de planejamento de movimento,
     * garantindo que o robô evite colisões com objetos detectados em tempo real.
     * 
     * **Responsabilidades principais:**
     * - Declara e lê o parâmetro `yaml_file`, que define o caminho do arquivo contendo as listas
     *   de *labels* autorizadas e não autorizadas.
     * - Declara e lê o parâmetro `move_group`, que define o nome do grupo MoveIt a ser controlado
     *   (ex.: `"denso_arm"`).
     * - Cria o **subscriber** para o tópico `/boxes_detection_array`, 
     *   que fornece as detecções 3D dos objetos no ambiente.
     * - Inicializa um **timer** para tentar configurar o `MoveGroupInterface` de forma assíncrona, 
     *   evitando erros caso o MoveIt ainda não esteja pronto na inicialização.
     * - Adiciona automaticamente um **plano de chão (ground plane)** como objeto de colisão.
     * - Carrega as **regras de labels autorizadas e não autorizadas** a partir do arquivo YAML especificado.
     * 
     * ### Subscribers
     * - `sub_` → Assina o tópico `/boxes_detection_array` com mensagens do tipo 
     *   `vision_msgs::msg::Detection3DArray`.  
     *   Cada detecção contém o ID do objeto (classe) e sua pose 3D no espaço.  
     *   O callback associado (`detectionCallback`) adiciona ou move caixas de colisão 
     *   correspondentes na cena MoveIt.
     * 
     * ### Timer
     * - `init_timer_` → Cria um temporizador que chama periodicamente `initMoveGroup()`
     *   até que o `MoveGroupInterface` seja inicializado com sucesso.  
     *   Esse método garante que o nó só tente manipular o MoveIt quando o contexto de ROS2
     *   estiver totalmente inicializado.
     * 
     * ### Parâmetros ROS2
     * - `yaml_file` (`std::string`) → Caminho do arquivo YAML contendo as listas de labels autorizadas e não autorizadas.
     * - `move_group` (`std::string`) → Nome do grupo MoveIt2 (por padrão, `"denso_arm"`).
     * 
     * @note O construtor também chama internamente:
     * - `add_ground_plane()` → adiciona o plano de chão como objeto fixo de colisão.
     * - `load_labels_from_yaml()` → carrega as listas de labels a partir do arquivo YAML configurado.
     */
    AddCollision()
     : Node("add_colision_objects")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("", "denso_arm");

        std::string labels_path = this->get_parameter("yaml_file").as_string();
        move_group = this->get_parameter("").as_string();


        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/boxes_detection_array", 10,
            std::bind(&AddCollision::detectionCallback, this, std::placeholders::_1));

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&AddCollision::initMoveGroup, this));
        
        add_ground_plane();
        load_labels_from_yaml(labels_path);
    }   
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AddCollision>());
    rclcpp::shutdown();
    return 0;
}
