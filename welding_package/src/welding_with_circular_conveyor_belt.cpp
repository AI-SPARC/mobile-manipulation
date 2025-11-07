/**
 * @file welding_with_circular_conveyor_belt.cpp
 * @brief Nó ROS2 responsável pelo controle automatizado de soldagem.
 * 
 * @details
 * Este arquivo implementa a classe `WeldingWithCircularConveyorBelt`, responsável por controlar o braço robótico e a esteira
 * no ambiente de simulação **Isaac Sim**, além de processar detecções 3D de objetos e executar
 * operações de soldagem automatizadas.  
 * 
 * A arquitetura integra **ROS2**, **MoveIt2** e **YAML-CPP**, realizando o planejamento de trajetórias,
 * controle da esteira e execução de poses de solda configuradas externamente via arquivo YAML.
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
 * @class WeldingWithCircularConveyorBelt
 * @brief Classe principal responsável pelo controle automatizado de soldagem.
 * 
 * @details
 * Implementa o nó ROS2 que controla a esteira e o braço robótico no ambiente
 * de simulação Isaac Sim, executando a sequência de soldagem automaticamente.
 */
class WeldingWithCircularConveyorBelt : public rclcpp::Node {

private:

    /// Publisher de velocidade linear da esteira.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
    /// Publisher de velocidade angular da esteira.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_1;
    /// Subscriber de detecções 3D.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    /// Interface MoveIt2 para controle do braço robótico.
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    /// Vetor de poses locais de soldagem carregadas do YAML.
    std::vector<geometry_msgs::msg::Pose> locations;
    /// Caminho do arquivo YAML de poses.
    std::string yaml_file;
    /// Temporizador para inicialização assíncrona do MoveGroup.
    rclcpp::TimerBase::SharedPtr init_timer_;

    /**
     * @brief Carrega poses de solda a partir de um arquivo YAML.
     * 
     * @param yaml_path Caminho do arquivo YAML.
     * @return Vetor de poses carregadas.
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
     * @brief Inicializa a interface MoveGroup do MoveIt2.
     * 
     * @details
     * Tenta criar a instância `move_group_arm` de forma assíncrona até obter sucesso.
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
     * @brief Retorna o braço robótico à posição inicial de soldagem.
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
                RCLCPP_INFO(this->get_logger(), "Returned to WeldingWithCircularConveyorBelt position.");
            }
        }
    }
    
    /**
     * @brief Move o braço robótico até uma pose alvo.
     * 
     * @param target_pose Pose de destino.
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

        move_group_arm->setPlanningTime(5.0);
        move_group_arm->setNumPlanningAttempts(40);

        move_group_arm->setMaxVelocityScalingFactor(0.5);
        move_group_arm->setMaxAccelerationScalingFactor(0.5);

        move_group_arm->setGoalTolerance(0.01);
        move_group_arm->setGoalJointTolerance(0.01);
        move_group_arm->setGoalPositionTolerance(0.01);
        move_group_arm->setGoalOrientationTolerance(0.01);

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

    /**
     * @brief Publica a velocidade angular da esteira.
     * @param velocity Valor da velocidade angular.
     */
    void publish_angular_velocity(float velocity)
    {
        auto message = std_msgs::msg::Float32();
        message.data = velocity;

        publisher_1->publish(message);

    }


    /*
    
        CALLBACKS.

    */
    
    std::string welding_id;
    bool stopped = false, welding_done = false;

     /**
     * @brief Callback de detecção de objetos 3D.
     * 
     * @details
     * Interpreta as mensagens do tópico `/bbox_3d_with_labels` e controla a esteira
     * e o robô conforme a posição e identificação dos objetos detectados.
     */
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        for (const auto &det : msg->detections)
        {
            if (det.results.empty() || det.results[0].hypothesis.class_id.find("firecabinet") == std::string::npos)
            {
                continue;
            }
            
            if(stopped == false)
            {
                publish_velocity(0.2);
                publish_angular_velocity(0.4);
            }


            if(det.bbox.center.position.y < 0.2 && det.bbox.center.position.y > -0.2  && det.bbox.center.position.x > 0.0 && stopped == true && welding_id == det.results[0].hypothesis.class_id)
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

                welding_done = true;
                return_to_welding_position();
                stopped = false;
                publish_velocity(0.2);
                publish_angular_velocity(0.4);
                rclcpp::sleep_for(std::chrono::milliseconds(50));
            }
            else if(det.bbox.center.position.y < 0.2 && det.bbox.center.position.y > -0.2 && det.bbox.center.position.x > 0.0 && stopped == false && welding_id != det.results[0].hypothesis.class_id)
            {
                publish_velocity(0.0);
                publish_angular_velocity(0.0);
                rclcpp::sleep_for(std::chrono::milliseconds(1000));
                welding_id = det.results[0].hypothesis.class_id;
                welding_done = false;
                stopped = true;
            }

        }
    }


        

public:
      /**
     * @brief Construtor da classe WeldingWithCircularConveyorBelt.
     * 
     * @details
     * Inicializa o nó ROS2 responsável pelo controle integrado da esteira circular e do braço robótico
     * no ambiente de simulação **Isaac Sim**, coordenando o processo automatizado de soldagem.  
     * 
     * Este construtor configura todos os componentes ROS2 necessários para o funcionamento completo
     * do sistema, desde a leitura das poses locais de solda até a publicação de comandos de velocidade
     * da esteira e o recebimento de detecções 3D de objetos.
     * 
     * ---
     * ### **Responsabilidades principais**
     * - **Declaração de parâmetros**
     *   - Declara e lê o parâmetro `yaml_file`, que define o caminho para o arquivo YAML contendo
     *     as poses locais de solda (posição e orientação de cada ponto relativo à peça detectada).
     * 
     * - **Configuração de publishers**
     *   - `publisher_` → Cria o publisher responsável por enviar mensagens do tipo `std_msgs::msg::Float32`
     *     para o tópico `/conveyor_velocity`, controlando a **velocidade linear da esteira circular**.
     *   - `publisher_1` → Cria o publisher para o tópico `/conveyor_angular_velocity`, enviando comandos
     *     para ajustar a **velocidade angular** (rotação) da esteira.
     * 
     * - **Assinatura de tópicos**
     *   - `sub_` → Cria o subscriber para o tópico `/bbox_3d_with_labels`, que recebe mensagens
     *     do tipo `vision_msgs::msg::Detection3DArray`.  
     *     Cada mensagem contém uma lista de objetos detectados com posição, orientação e identificador
     *     da classe (ex: “firecabinet”).  
     *     O callback associado (`detectionCallback`) é responsável por:
     *     - Monitorar a posição do objeto detectado;
     *     - Parar a esteira quando o objeto estiver na área de soldagem;
     *     - Acionar o robô para realizar a sequência de solda com base nas poses carregadas do YAML;
     *     - Retomar o movimento da esteira após o término da soldagem.
     * 
     * - **Leitura e carregamento das poses de solda**
     *   - Chama o método `loadLocationsFromYaml(yaml_file)` para carregar as poses locais
     *     de solda (coordenadas relativas à peça) definidas no arquivo de configuração YAML.
     *     Essas poses são transformadas em coordenadas globais quando a peça é detectada,
     *     permitindo que o robô realize a soldagem com precisão em cada ponto.
     * 
     * - **Inicialização do MoveIt2**
     *   - Cria o temporizador `init_timer_`, que chama periodicamente o método `initMoveGroup()`
     *     até que a interface `MoveGroupInterface` do MoveIt2 seja inicializada com sucesso.
     *     Isso garante que o sistema não falhe caso o MoveIt2 ainda não esteja pronto
     *     no momento em que o nó é iniciado.
     * 
     * ---
     * ### **Fluxo resumido**
     * 1. Declara o parâmetro `yaml_file`;
     * 2. Cria os publishers de velocidade linear e angular da esteira;
     * 3. Cria o subscriber de detecções 3D com callback inteligente;
     * 4. Carrega poses de solda do arquivo YAML configurado;
     * 5. Inicia o temporizador que tenta conectar o MoveIt2 até sucesso.
     * 
     * ---
     * ### **Integração geral**
     * - Integração direta com o **Isaac Sim** via tópicos `/conveyor_velocity` e `/conveyor_angular_velocity`;
     * - Comunicação com o **MoveIt2** para planejamento e execução de trajetórias do braço robótico;
     * - Compatibilidade com mensagens do tipo **vision_msgs** (usadas em pipelines de visão 3D);
     * - Arquitetura modular, permitindo fácil substituição da esteira linear por circular sem alterar o restante do código.
     * 
     * @see loadLocationsFromYaml(), initMoveGroup(), detectionCallback()
     */
    WeldingWithCircularConveyorBelt()
     : Node("welding_with_circular_conveyor_belt")
    {
        this->declare_parameter<std::string>("yaml_file", "");
   
        yaml_file = this->get_parameter("yaml_file").as_string();
        
        publisher_ = this->create_publisher<std_msgs::msg::Float32>("/conveyor_velocity", 10);
        publisher_1 = this->create_publisher<std_msgs::msg::Float32>("/conveyor_angular_velocity", 10);
       
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&WeldingWithCircularConveyorBelt::detectionCallback, this, std::placeholders::_1));
        
        loadLocationsFromYaml(yaml_file);

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&WeldingWithCircularConveyorBelt::initMoveGroup, this));

    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WeldingWithCircularConveyorBelt>());
  rclcpp::shutdown();
  return 0;
}