/**
 * @file welding.cpp
 * @brief Nó ROS2 responsável pelo controle automatizado de soldagem básica em esteira linear.
 * * @details
 * Este arquivo implementa a classe `BasicWelding`, responsável por controlar um braço robótico
 * e uma **esteira linear** no ambiente de simulação **Isaac Sim**.  
 * * A arquitetura integra **ROS2**, **MoveIt2** e **YAML-CPP**. O nó monitora detecções 3D de 
 * objetos (ex: "firecabinet"), para a esteira quando o objeto está na posição correta,
 * executa uma sequência de pontos de solda pré-configurada, e então retoma o movimento da esteira.
 * * Diferente de implementações com esteiras circulares ou trajetórias complexas, este nó
 * foca em uma operação linear simples:
 * 1. Detecta objeto.
 * 2. Para esteira.
 * 3. Executa *toda* a lista de poses do YAML.
 * 4. Marca objeto como feito (para não soldar de novo).
 * 5. Continua esteira.
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
#include <yaml-cpp/yaml.h>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "sensor_msgs/msg/point_cloud2.hpp"
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
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/msg/float32.hpp>

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
 * @class BasicWelding
 * @brief Classe principal para o controle de soldagem básica em esteira linear.
 * * @details
 * Implementa o nó ROS2 que monitora detecções 3D, controla uma esteira linear 
 * (publicando em `/conveyor_velocity`) e comanda o braço robótico (via MoveIt2) 
 * para executar uma sequência de solda fixa carregada de um arquivo YAML.
 */
class BasicWelding : public rclcpp::Node {

private:

    /// Publisher de trajetória de juntas (declarado mas não inicializado/usado neste nó).
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;
    /// Publisher de velocidade linear da esteira.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;

    /// Subscriber principal de detecções 3D (para /bbox_3d_with_labels).
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    /// Subscriber secundário de detecções 3D (declarado mas não inicializado/usado).
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_1;

    /// Interface MoveIt2 para controle do braço robótico.
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;

    /// Vetor de poses locais de soldagem (relativas ao objeto) carregadas do YAML.
    std::vector<geometry_msgs::msg::Pose> locations;
    /// Caminho do arquivo YAML de poses.
    std::string yaml_file;

    /// Temporizador para inicialização assíncrona do MoveGroup.
    rclcpp::TimerBase::SharedPtr init_timer_;

    /// Conjunto para rastrear IDs de objetos que já foram soldados.
    std::unordered_set<std::string> welding_done;
    /// Flag de estado para indicar se a esteira está parada para soldagem.
    bool stopped = false;

    /**
     * @brief Carrega uma lista de poses de solda de um arquivo YAML.
     * * @details
     * Processa o arquivo YAML especificado, esperando uma estrutura de lista simples
     * de poses (posição e orientação), e as armazena no vetor `locations`.
     * * @param yaml_path Caminho do arquivo YAML a ser carregado.
     * @return Vetor de `geometry_msgs::msg::Pose` contendo as poses de solda locais.
     */
    std::vector<geometry_msgs::msg::Pose> loadLocationsFromYaml(const std::string &yaml_path)
    {
        

        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);

            for (const auto &it : config)
            {
                const std::string key = it.first.as<std::string>();
                const YAML::Node node = it.second;

                if (!node["position"] || !node["orientation"])
                {
                    RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                "Location '%s' missing position or orientation", key.c_str());
                    continue;
                }

                const YAML::Node pos = node["position"];
                const YAML::Node ori = node["orientation"];

                if (pos.size() != 3 || ori.size() != 4)
                {
                    RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                "Invalid size for position/orientation in '%s'", key.c_str());
                    continue;
                }

                geometry_msgs::msg::Pose pose;
                pose.position.x = pos[0].as<double>();
                pose.position.y = pos[1].as<double>();
                pose.position.z = pos[2].as<double>();

                pose.orientation.x = ori[0].as<double>();
                pose.orientation.y = ori[1].as<double>();
                pose.orientation.z = ori[2].as<double>();
                pose.orientation.w = ori[3].as<double>();

                locations.push_back(pose);

                RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                            "Loaded %s -> pos:[%.2f, %.2f, %.2f], ori:[%.2f, %.2f, %.2f, %.2f]",
                            key.c_str(),
                            pose.position.x, pose.position.y, pose.position.z,
                            pose.orientation.x, pose.orientation.y,
                            pose.orientation.z, pose.orientation.w);
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                        "Failed to load YAML file '%s': %s", yaml_path.c_str(), e.what());
        }

        return locations;
    }


    /**
     * @brief Inicializa a interface MoveGroup do MoveIt2 de forma assíncrona.
     * * @details
     * Tenta criar a instância `move_group_arm` para o grupo "denso_arm".
     * Se for bem-sucedido, cancela o temporizador `init_timer_`.
     */
    void initMoveGroup() {
        try 
        {

            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "denso_arm");  

                rclcpp::sleep_for(std::chrono::milliseconds(5000));
            
            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface inicializado com sucesso.");

            init_timer_->cancel();  
        } catch (const std::exception &e) 
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }

    }

    /**
     * @brief Retorna o braço robótico à sua posição inicial (home).
     * * @details
     * Planeja e executa um movimento para a configuração de juntas predefinida
     * de "home" ou "pronto para soldar".
     */
    void return_to_welding_position()
    {
         if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do arm não inicializado.");
            return;
        }

        
        move_group_arm->setJointValueTarget({
            {"joint1", 0.0},
            {"joint2", -1.1288},
            {"joint3", 2.057},
            {"joint4", 0.0},
            {"joint5", 0.658},
            {"joint6", 0.0},        
        });

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            auto exec_result = move_group_arm->execute(plan);
            rclcpp::sleep_for(std::chrono::milliseconds(100));
            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) 
            {
                RCLCPP_INFO(this->get_logger(), "Returned to BasicWelding position.");
            }
        }
    }
    
    /**
     * @brief Move o braço robótico até uma pose alvo.
     * * @details
     * Configura o planejador MoveIt2 e tenta encontrar e executar um plano
     * para mover o efetuador final até a `target_pose` fornecida.
     * * @param target_pose A pose de destino para o efetuador final (em coordenadas globais).
     */
    void positions_for_arm(const geometry_msgs::msg::Pose &target_pose) 
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return;
        }

      
        move_group_arm->setStartStateToCurrentState();
        move_group_arm->setPlannerId("RRTConnect");
        move_group_arm->setPoseTarget(target_pose);

        move_group_arm->setPlanningTime(10.0);
        move_group_arm->setNumPlanningAttempts(200);

        move_group_arm->setMaxVelocityScalingFactor(0.5);
        move_group_arm->setMaxAccelerationScalingFactor(0.5);

        move_group_arm->setGoalTolerance(0.001);
        move_group_arm->setGoalJointTolerance(0.001);
        move_group_arm->setGoalPositionTolerance(0.001);
        move_group_arm->setGoalOrientationTolerance(0.001);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_arm->execute(plan);
    
            rclcpp::sleep_for(std::chrono::milliseconds(50));
        }

        if (plan.trajectory.joint_trajectory.points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Trajetória vazia retornada pelo planejador");
            return;
        }

    }

    /**
     * @brief Publica a velocidade linear da esteira.
     * @param velocity Valor da velocidade linear.
     */
    void publish_velocity(float velocity)
    {
        auto message = std_msgs::msg::Float32();
        message.data = velocity;

        publisher_->publish(message);

    }


    /*
    
        CALLBACKS.

    */
    
 
    /**
     * @brief Callback de detecção de objetos 3D.
     * * @details
     * Este é o núcleo da lógica de controle.
     * 1. Filtra detecções para a classe "firecabinet".
     * 2. Verifica se o objeto está na zona de solda (`y < 0.2 && y > 0.0`).
     * 3. Verifica se o objeto já foi soldado (usando `welding_done`).
     * 4. Se está na zona, não foi soldado, e a esteira está se movendo (`stopped == false`):
     * - Para a esteira (`publish_velocity(0.0)`) e marca `stopped = true`.
     * 5. Se a esteira está parada (`stopped == true`) e o objeto ainda não foi soldado:
     * - Executa *toda* a sequência de poses de `locations`, transformando cada
     * pose local em global usando a pose do objeto detectado.
     * - Após o loop, marca o ID do objeto como soldado (`welding_done.insert`).
     * - Retorna o robô à posição inicial.
     * - Libera a esteira (`stopped = false`) e define a velocidade para continuar.
     * 6. Se nenhuma condição for atendida (ex: objeto fora da zona), apenas continua a esteira.
     * * @param msg Mensagem `Detection3DArray` recebida do tópico `/bbox_3d_with_labels`.
     */
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        for (const auto &det : msg->detections)
        {
            if (det.results.empty() || det.results[0].hypothesis.class_id != "firecabinet")
                continue;

            if(det.bbox.center.position.y < 0.2 && det.bbox.center.position.y > 0.0 && stopped == false && welding_done.find(det.results[0].hypothesis.class_id) == welding_done.end())
            {
                publish_velocity(0.0);
                rclcpp::sleep_for(std::chrono::milliseconds(500));
                stopped = true;
            }
            else if(stopped == true && welding_done.find(det.results[0].hypothesis.class_id) == welding_done.end())
            {
                
                for (size_t i = 0; i < locations.size(); i++)
                {
                    tf2::Vector3 local_corner(locations[i].position.x, locations[i].position.y, locations[i].position.z);

                    const auto &pose = det.bbox.center;
                    tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
                    tf2::Matrix3x3 rot(q);
                    tf2::Vector3 translation(pose.position.x, pose.position.y, pose.position.z);

                    tf2::Vector3 world_corner = rot * local_corner + translation;

                    geometry_msgs::msg::Pose target_pose;
                    target_pose.position.x = world_corner.x();
                    target_pose.position.y = world_corner.y();
                    target_pose.position.z = world_corner.z();

                    
                    target_pose.orientation.x = locations[i].orientation.x;
                    target_pose.orientation.y = locations[i].orientation.y;
                    target_pose.orientation.z = locations[i].orientation.z;
                    target_pose.orientation.w = locations[i].orientation.w;

                    RCLCPP_INFO(this->get_logger(),
                                "Pose %zu - ponto global: x=%.3f, y=%.3f, z=%.3f",
                                i, world_corner.x(), world_corner.y(), world_corner.z());

                    positions_for_arm(target_pose);
                }

                welding_done.insert(det.results[0].hypothesis.class_id);
                return_to_welding_position();
                stopped = false;
                publish_velocity(0.1);
                rclcpp::sleep_for(std::chrono::milliseconds(50));
            }
            else
            {
                publish_velocity(0.1);
                rclcpp::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }


        

public:
    /**
     * @brief Construtor da classe BasicWelding.
     * * @details
     * Inicializa o nó ROS2, configurando os componentes para o processo de soldagem em esteira linear.
     * * ---
     * ### Responsabilidades principais
     * - **Declaração de parâmetros**
     * - Declara e lê o parâmetro `yaml_file` contendo o caminho para o arquivo YAML
     * com a sequência *única* de poses locais de solda.
     * * - **Configuração de publishers**
     * - `publisher_` → Cria o publisher para o tópico `/conveyor_velocity`, controlando a
     * **velocidade linear da esteira**.
     * * - **Assinatura de tópicos**
     * - `sub_` → Cria o subscriber para o tópico `/bbox_3d_with_labels`, que recebe detecções 3D.
     * O callback associado (`detectionCallback`) orquestra a parada da esteira,
     * a execução da sequência de solda e a retomada da esteira.
     * * - **Leitura e carregamento das poses de solda**
     * - Chama o método `loadLocationsFromYaml(yaml_file)` para carregar a lista
     * de poses locais de solda.
     * * - **Inicialização do MoveIt2**
     * - Cria o temporizador `init_timer_`, que chama periodicamente `initMoveGroup()`
     * até que a interface `MoveGroupInterface` seja inicializada com sucesso.
     * * ---
     * @see loadLocationsFromYaml(), initMoveGroup(), detectionCallback()
     */
    BasicWelding()
     : Node("basic_welding")
    {
        this->declare_parameter<std::string>("yaml_file", "");
   
        yaml_file = this->get_parameter("yaml_file").as_string();
        
        publisher_ = this->create_publisher<std_msgs::msg::Float32>("/conveyor_velocity", 10);
       
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&BasicWelding::detectionCallback, this, std::placeholders::_1));
        
        loadLocationsFromYaml(yaml_file);

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&BasicWelding::initMoveGroup, this));

    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BasicWelding>());
  rclcpp::shutdown();
  return 0;
}