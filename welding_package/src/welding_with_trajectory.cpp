/**
 * @file welding_with_trajectory.cpp
 * @brief Nó ROS2 responsável pela soldagem automatizada com trajetórias complexas usando MoveIt2 e Isaac Sim.
 * * @details
 * Este arquivo implementa o nó **WeldingWithTrajectory**, responsável por realizar o processo completo
 * de soldagem de peças transportadas por uma **esteira circular** simulada no **Isaac Sim**, utilizando
 * controle de trajetória do **MoveIt2** para o braço robótico.
 * * O nó coordena de forma autônoma a movimentação do robô e da esteira, realizando diferentes tipos de
 * trajetórias de soldagem — **normais, lineares ou circulares** — de acordo com os parâmetros definidos
 * em um arquivo YAML.
 * * ---
 * ### Principais funcionalidades
 * - Integração direta com **Isaac Sim** para controle de velocidade da esteira circular;
 * - Planejamento e execução de **trajetórias cartesianas** com MoveIt2;
 * - Leitura de **poses e trajetórias de solda** a partir de um arquivo YAML configurável;
 * - Processamento de **detecções 3D** de objetos via tópico `/bbox_3d_with_labels`;
 * - Controle sincronizado entre **detecção, parada da esteira e execução da solda**;
 * - Suporte a **três tipos de trajetória**:
 * - `"normal"`: movimento até uma única pose de soldagem;
 * - `"line"`: soldagem linear entre o ponto atual e um ponto alvo;
 * - `"circle"`: soldagem circular (definida por raio e ângulos);
 * - Retorno automático do robô à posição inicial após a solda e retomada do movimento da esteira.
 * * ---
 * ### Fluxo geral de operação
 * 1. O nó é iniciado e declara o parâmetro `yaml_file` contendo o caminho do arquivo de configuração;
 * 2. O arquivo YAML é carregado pelo método `loadLocationsFromYaml()`, populando um mapa de operações de solda por objeto;
 * 3. São criados publishers e subscribers ROS2 para comunicação com Isaac Sim e o sistema de visão;
 * 4. Um temporizador (`init_timer_`) tenta inicializar o `MoveGroupInterface` de forma assíncrona;
 * 5. Ao detectar uma peça via `/bbox_3d_with_labels`, o callback `detectionCallback()`:
 * - Identifica o tipo de trajetória associado à peça;
 * - Para a esteira quando a peça está na posição ideal;
 * - Executa a trajetória de solda correspondente (ponto, linha ou arco);
 * - Retoma o movimento da esteira após o término da solda.
 * * ---
 * ### Tópicos ROS2
 * **Publishers**
 * - `/conveyor_velocity` (`std_msgs::msg::Float32`): Controla a **velocidade linear** da esteira.
 * - `/conveyor_angular_velocity` (`std_msgs::msg::Float32`): Controla a **velocidade angular** da esteira.
 * * **Subscriber**
 * - `/bbox_3d_with_labels` (`vision_msgs::msg::Detection3DArray`): Recebe detecções de objetos com pose e rótulo, disparando o processo de soldagem.
 * * ---
 * ### Parâmetros
 * - `yaml_file`: Caminho para o arquivo YAML contendo as poses e definições de trajetória para cada tipo de objeto.
 * * ---
 * ### Integração com MoveIt2
 * O `MoveGroupInterface` para o grupo `"denso_arm"` é usado para planejar e executar movimentos.
 * A função de planejamento **cartesiano** (`computeCartesianPath`) é fundamental para gerar trajetórias
 * suaves e contínuas para as soldas do tipo `"line"` e `"circle"`.
 * * ---
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
 * @class WeldingWithTrajectory
 * @brief Classe principal que gerencia o processo de soldagem automatizada com trajetórias complexas.
 * * @details
 * Esta classe implementa o nó ROS2 que integra a detecção de objetos, o controle da esteira circular
 * e a execução de trajetórias de solda (ponto, linha, arco) com o MoveIt2 no ambiente Isaac Sim.
 * Ela é responsável por carregar configurações de um arquivo YAML, reagir a detecções de objetos,
 * controlar a esteira e comandar o braço robótico para executar as operações de soldagem.
 */
class WeldingWithTrajectory : public rclcpp::Node {

private:
    /// @brief Estrutura para armazenar os dados de uma pose de soldagem, incluindo tipo e parâmetros de trajetória.
    struct WeldingPoseData
    {
        geometry_msgs::msg::Pose pose; ///< A pose de referência local (relativa ao objeto).
        std::string trajectory_type;   ///< O tipo de trajetória ("normal", "line", "circle").
        std::vector<double> trajectory_data; ///< Parâmetros para a trajetória (ex: raio, ângulos).
    };

    /// Publisher para a velocidade linear da esteira.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
    /// Publisher para a velocidade angular da esteira.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_1;
    /// Subscriber para as detecções 3D de objetos.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    /// Interface MoveIt2 para controle do grupo de juntas do braço robótico.
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    /// Caminho para o arquivo YAML de configuração das poses e trajetórias.
    std::string yaml_file;
    /// Temporizador para a inicialização assíncrona do MoveGroupInterface.
    rclcpp::TimerBase::SharedPtr init_timer_;
    /// Mapa que associa o ID de um objeto a um vetor de operações de soldagem.
    std::unordered_map<std::string, std::vector<WeldingPoseData>> welding_poses;
    /// Variáveis de estado para controlar o processo de soldagem.
    std::string welding_id;
    bool stopped = false, welding_done = false;

    /**
     * @brief Carrega as poses e os dados de trajetória de um arquivo YAML.
     * * @details
     * Este método processa um arquivo YAML, extraindo não apenas a posição e orientação, mas também
     * o tipo de trajetória (`normal`, `line`, `circle`) e seus parâmetros associados. Os dados
     * são armazenados no mapa `welding_poses`, usando o rótulo do objeto como chave.
     * * @param yaml_path Caminho do arquivo YAML a ser carregado.
     */
    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);

            for (const auto &label_node : config)
            {
                const std::string label = label_node.first.as<std::string>();
                const YAML::Node &locations_node = label_node.second;

                std::vector<WeldingPoseData> locations;

                for (const auto &loc_item : locations_node)
                {
                    if (!loc_item.IsMap() || loc_item.size() != 1)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] Ignorando entrada inválida de localização.", label.c_str());
                        continue;
                    }

                    const auto &loc_name = loc_item.begin()->first.as<std::string>();
                    const YAML::Node &loc_data = loc_item.begin()->second;

                    if (!loc_data["position"] || !loc_data["orientation"])
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] '%s' missing position/orientation",
                                    label.c_str(), loc_name.c_str());
                        continue;
                    }

                    const YAML::Node &pos = loc_data["position"];
                    const YAML::Node &ori = loc_data["orientation"];

                    if (pos.size() != 3 || ori.size() != 4)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] '%s' invalid position/orientation size",
                                    label.c_str(), loc_name.c_str());
                        continue;
                    }

                    WeldingPoseData wp;
                    wp.pose.position.x = pos[0].as<double>();
                    wp.pose.position.y = pos[1].as<double>();
                    wp.pose.position.z = pos[2].as<double>();
                    wp.pose.orientation.x = ori[0].as<double>();
                    wp.pose.orientation.y = ori[1].as<double>();
                    wp.pose.orientation.z = ori[2].as<double>();
                    wp.pose.orientation.w = ori[3].as<double>();

                    if (loc_data["trajectory"])
                    {
                        const YAML::Node &traj = loc_data["trajectory"];
                        if (traj.IsSequence() && traj.size() >= 1)
                        {
                            wp.trajectory_type = traj[0].as<std::string>();
                            for (size_t i = 1; i < traj.size(); ++i)
                                wp.trajectory_data.push_back(traj[i].as<double>());
                        }
                        else
                        {
                            wp.trajectory_type = "normal";
                        }
                    }
                    else
                    {
                        wp.trajectory_type = "normal";
                    }

                    RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                                "Loaded [%s - %s] -> pos:[%.2f, %.2f, %.2f], ori:[%.2f, %.2f, %.2f, %.2f], traj:%s (%zu values)",
                                label.c_str(), loc_name.c_str(),
                                wp.pose.position.x, wp.pose.position.y, wp.pose.position.z,
                                wp.pose.orientation.x, wp.pose.orientation.y,
                                wp.pose.orientation.z, wp.pose.orientation.w,
                                wp.trajectory_type.c_str(), wp.trajectory_data.size());

                    locations.push_back(wp);
                }

                welding_poses[label] = locations;
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                        "Failed to load YAML file '%s': %s", yaml_path.c_str(), e.what());
        }
    }


    /**
     * @brief Inicializa a interface MoveGroup do MoveIt2 de forma assíncrona.
     * @details Tenta criar a instância `move_group_arm` e cancela o temporizador em caso de sucesso.
     */
    void initMoveGroup() {
        try 
        {

            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "denso_arm");  

            
            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface inicializado com sucesso.");

            init_timer_->cancel();  
        } catch (const std::exception &e) 
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }

    }

    /**
     * @brief Retorna o braço robótico à sua posição inicial de prontidão.
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
                RCLCPP_INFO(this->get_logger(), "Returned to WeldingWithTrajectory position.");
            }
        }
    }
    
    /**
     * @brief Move o robô para uma única pose alvo (trajetória 'normal').
     * @details Configura e executa um plano do MoveIt2 para mover o efetuador final até uma
     * `target_pose` específica. Usado para operações de soldagem em um único ponto.
     * @param target_pose A pose de destino para o efetuador final.
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

        move_group_arm->setPlanningTime(2.0);
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
    
 
    /**
     * @brief Callback principal que processa detecções 3D e orquestra a soldagem.
     * * @details
     * Acionado a cada nova mensagem em `/bbox_3d_with_labels`. A lógica principal é:
     * 1. Extrai o ID do objeto detectado.
     * 2. Se a esteira está em movimento, mantém a velocidade.
     * 3. Se um novo objeto chega à zona de solda, para a esteira.
     * 4. Recupera as operações de solda para esse ID do mapa `welding_poses`.
     * 5. Para cada operação, transforma a pose local em global usando a pose do objeto detectado.
     * 6. Executa a trajetória correspondente: 'normal' (ponto único), 'line' (caminho cartesiano linear) ou
     * 'circle' (gera waypoints em arco e usa caminho cartesiano).
     * 7. Após concluir todas as operações, retorna o robô à posição inicial e reinicia a esteira.
     * * @param msg Ponteiro compartilhado para a mensagem `Detection3DArray` recebida.
     */
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        std::string id;
        for (const auto &det : msg->detections)
        {
            if (det.results.empty())
                continue;

            size_t pos = det.results[0].hypothesis.class_id.find('_'); 
            if (pos != std::string::npos)
                id = det.results[0].hypothesis.class_id.substr(0, pos);
            else
                id = det.results[0].hypothesis.class_id;

            if (welding_poses.find(id) == welding_poses.end())
                continue;

            if (!stopped)
            {
                publish_velocity(0.2);
                publish_angular_velocity(0.4);
            }

            if (det.bbox.center.position.y < 0.3 && det.bbox.center.position.y > -0.1 &&det.bbox.center.position.x > 0.0 && stopped && welding_id == det.results[0].hypothesis.class_id)
            {
                const auto &poses = welding_poses[id];

                for (const auto &wp_data : poses)
                {
                    const auto &pose_local = wp_data.pose;
                    const auto &traj_type = wp_data.trajectory_type;
                    const auto &traj_data = wp_data.trajectory_data;

                    double raio = 0.1;
                    double angulo_inicial = 0.0;
                    double angulo_final = 2 * M_PI;
                    int num_pontos = 40;

                    if (traj_type == "circle" && traj_data.size() == 4)
                    {
                        raio = traj_data[0];
                        angulo_inicial = traj_data[1];
                        angulo_final = traj_data[2];
                        num_pontos = static_cast<int>(traj_data[3]);
                    }

                    const auto &bbox_pose = det.bbox.center;

                    tf2::Quaternion q_bbox(
                        bbox_pose.orientation.x,
                        bbox_pose.orientation.y,
                        bbox_pose.orientation.z,
                        bbox_pose.orientation.w);

                    tf2::Matrix3x3 rot_bbox(q_bbox);
                    tf2::Vector3 translation_bbox(
                        bbox_pose.position.x,
                        bbox_pose.position.y,
                        bbox_pose.position.z);

                    tf2::Vector3 local_center(
                        pose_local.position.x,
                        pose_local.position.y,
                        pose_local.position.z);

                    tf2::Vector3 centro_mundo = rot_bbox * local_center + translation_bbox;

                    tf2::Quaternion q_local(
                        pose_local.orientation.x,
                        pose_local.orientation.y,
                        pose_local.orientation.z,
                        pose_local.orientation.w);

                    tf2::Quaternion q_total = q_bbox * q_local;
                    tf2::Matrix3x3 rot_total(q_total);

                    RCLCPP_INFO(this->get_logger(),
                                "[%s] Trajetória tipo '%s' - Centro: [%.3f, %.3f, %.3f]",
                                id.c_str(), traj_type.c_str(),
                                centro_mundo.x(), centro_mundo.y(), centro_mundo.z());

                    std::vector<geometry_msgs::msg::Pose> waypoints;

                    if (traj_type == "circle")
                    {
                        waypoints.reserve(num_pontos);
                        for (int j = 0; j <= num_pontos; ++j)
                        {
                            double t = static_cast<double>(j) / num_pontos;
                            double ang = angulo_inicial + t * (angulo_final - angulo_inicial);
                            tf2::Vector3 ponto_local(raio * cos(ang), raio * sin(ang), 0.0);
                            tf2::Vector3 ponto_mundo = rot_total * ponto_local + centro_mundo;

                            geometry_msgs::msg::Pose p;
                            p.position.x = ponto_mundo.x();
                            p.position.y = ponto_mundo.y();
                            p.position.z = ponto_mundo.z();
                            p.orientation = tf2::toMsg(q_total);
                            waypoints.push_back(p);
                        }

                        moveit_msgs::msg::RobotTrajectory trajectory;
                        double fraction = move_group_arm->computeCartesianPath(waypoints, 0.01, trajectory);

                        if (fraction > 0.99)
                        {
                            moveit::planning_interface::MoveGroupInterface::Plan plan;
                            plan.trajectory = trajectory;
                            move_group_arm->execute(plan);
                        }
                    }
                    else if (traj_type == "line")
                    {
                        geometry_msgs::msg::Pose start_pose = move_group_arm->getCurrentPose().pose;

                        geometry_msgs::msg::Pose target_pose = pose_local;

                        std::vector<geometry_msgs::msg::Pose> waypoints = {start_pose, target_pose};

                        moveit_msgs::msg::RobotTrajectory trajectory;
                        double fraction = move_group_arm->computeCartesianPath(waypoints, 0.01, trajectory);

                        if (fraction > 0.99)
                        {
                            moveit::planning_interface::MoveGroupInterface::Plan plan;
                            plan.trajectory = trajectory;
                            move_group_arm->execute(plan);
                        }
                        else
                        {
                            RCLCPP_WARN(this->get_logger(),
                                        "Trajetória linear parcial. Fração: %.2f", fraction);
                        }
                    }
                    else if (traj_type == "normal")
                    {   
                        const auto &pose_local = wp_data.pose;

                        tf2::Vector3 local_corner(
                            pose_local.position.x,
                            pose_local.position.y,
                            pose_local.position.z);

                        const auto &bbox_pose = det.bbox.center;

                        tf2::Quaternion q(
                            bbox_pose.orientation.x,
                            bbox_pose.orientation.y,
                            bbox_pose.orientation.z,
                            bbox_pose.orientation.w);

                        tf2::Matrix3x3 rot(q);
                        tf2::Vector3 translation(
                            bbox_pose.position.x,
                            bbox_pose.position.y,
                            bbox_pose.position.z);

                        tf2::Vector3 world_corner = rot * local_corner + translation;

                        geometry_msgs::msg::Pose target_pose;
                        target_pose.position.x = world_corner.x();
                        target_pose.position.y = world_corner.y();
                        target_pose.position.z = world_corner.z();

                        target_pose.orientation = pose_local.orientation;
                     
                        positions_for_arm(target_pose);
                    }
                }

                

                welding_done = true;
                rclcpp::sleep_for(std::chrono::milliseconds(200));
                return_to_welding_position();
                stopped = false;

                publish_velocity(0.2);
                publish_angular_velocity(0.4);
            }
            else if (det.bbox.center.position.y < 0.3 && det.bbox.center.position.y > -0.1 &&
                    det.bbox.center.position.x > 0.0 && !stopped &&
                    welding_id != det.results[0].hypothesis.class_id)
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
     * @brief Construtor da classe WeldingWithTrajectory.
     * * @details
     * Inicializa o nó ROS2, configurando todos os componentes necessários para o funcionamento
     * do sistema de soldagem automatizada com trajetórias complexas.
     * * ---
     * ### **Responsabilidades principais**
     * - **Declaração de parâmetros**: Declara e lê o parâmetro `yaml_file` para a configuração das trajetórias.
     * - **Configuração de publishers**: Cria publishers para controlar as velocidades linear e angular da esteira.
     * - **Assinatura de tópicos**: Cria um subscriber para `/bbox_3d_with_labels`, que aciona o callback `detectionCallback` para iniciar o processo de soldagem.
     * - **Carregamento de Configuração**: Chama `loadLocationsFromYaml` para carregar as operações de solda do arquivo YAML.
     * - **Inicialização do MoveIt2**: Inicia um temporizador que chama `initMoveGroup()` para garantir que a interface com o MoveIt2 seja estabelecida de forma robusta.
     * * ---
     * @see loadLocationsFromYaml(), initMoveGroup(), detectionCallback()
     */
    WeldingWithTrajectory()
     : Node("welding_with_trajectory")
    {
        this->declare_parameter<std::string>("yaml_file", "");
   
        yaml_file = this->get_parameter("yaml_file").as_string();
        
        publisher_ = this->create_publisher<std_msgs::msg::Float32>("/conveyor_velocity", 10);
        publisher_1 = this->create_publisher<std_msgs::msg::Float32>("/conveyor_angular_velocity", 10);
       
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&WeldingWithTrajectory::detectionCallback, this, std::placeholders::_1));
        
        loadLocationsFromYaml(yaml_file);

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&WeldingWithTrajectory::initMoveGroup, this));

    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WeldingWithTrajectory>());
  rclcpp::shutdown();
  return 0;
}