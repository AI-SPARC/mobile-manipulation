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
#include "object_manipulation_interfaces/srv/picked_object.hpp"
#include "object_manipulation_interfaces/srv/goal_reached.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/msg/float32.hpp>
#include <cmath> 
#include "std_msgs/msg/bool.hpp"
#include <thread> 

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


class SimpleManipulation : public rclcpp::Node {

private:

    struct LocationData
    {
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size;
    };

    struct StorageData
    {
        int max_x_objects, max_y_objects, max_z_objects;
        int x;
        int y;
        int z = 0;
        int direction;
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size;
    };

    //Publishers.
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher_2;
    
    //Service.
    rclcpp::Service<object_manipulation_interfaces::srv::PickedObject>::SharedPtr service_;
    rclcpp::Service<object_manipulation_interfaces::srv::GoalReached>::SharedPtr service_1;

    //Timer.
    rclcpp::TimerBase::SharedPtr init_timer_;

   
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;

    rclcpp::Node::SharedPtr moveit_node_;
    rclcpp::Executor::SharedPtr executor_;
    std::thread executor_thread_;

    std::string yaml_file, storages_yaml_file;
    std::unordered_map<std::string, std::vector<geometry_msgs::msg::Pose>> pick_and_place_poses;
    std::pair<std::string, geometry_msgs::msg::Pose> object;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);

            for (const auto &label_node : config)
            {
                const std::string label = label_node.first.as<std::string>();
                const YAML::Node &locations_node = label_node.second;

                std::vector<geometry_msgs::msg::Pose> locations;

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
                                "Loaded [%s - %s] -> pos:[%.2f, %.2f, %.2f], ori:[%.2f, %.2f, %.2f, %.2f]",
                                label.c_str(), loc_name.c_str(),
                                pose.position.x, pose.position.y, pose.position.z,
                                pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w);
                }

                pick_and_place_poses[label] = locations;
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                        "Failed to load YAML file '%s': %s", yaml_path.c_str(), e.what());
        }
    }



    void initMoveGroup() 
    {
        try 
        {
           
            move_group_arm = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
                moveit_node_, "panda_arm"); 
            
            
            move_group_arm->startStateMonitor(); 

            RCLCPP_INFO(this->get_logger(), "MoveGroup (arm) inicializados com sucesso.");

            init_timer_->cancel();  
        } 
        catch (const std::exception &e) 
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s. Tentando novamente...", e.what());
        }

    }

    
    void ready()
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do arm não inicializado.");
            return;
        }

        
        move_group_arm->setJointValueTarget({
            {"panda_joint1", 0.0},
            {"panda_joint2", -0.750},
            {"panda_joint3", 0.0},
            {"panda_joint4", -2.827},
            {"panda_joint5", 0.0},
            {"panda_joint6", 2.077},
            {"panda_joint7", 0.785},
        
        });

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            auto exec_result = move_group_arm->execute(plan);
            rclcpp::sleep_for(std::chrono::milliseconds(100));
            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) 
            {
                RCLCPP_INFO(this->get_logger(), "Gripper fechou (MoveIt).");
            }
        }
    }

   
    // 1. Altere o tipo de retorno de 'void' para 'bool'
bool positions_for_arm(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool computeCartesian)
{
    if (!move_group_arm)
    {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
        return false; // Retorna falha se não estiver inicializado
    }

    const int MAX_PLANNING_CYCLES = 100;
    const int MAX_CARTESIAN_ATTEMPTS = 5;
    const double MIN_CARTESIAN_FRACTION = 0.99;

    bool task_success = false; // Variável que será retornada

    for (int cycle = 1; cycle <= MAX_PLANNING_CYCLES; ++cycle)
    {
        RCLCPP_INFO(this->get_logger(), "Ciclo de Planejamento Externo: Tentativa %d de %d", cycle, MAX_PLANNING_CYCLES);

        if (computeCartesian)
        {
            RCLCPP_INFO(this->get_logger(), "Tentando planejamento Cartesiano...");

            std::vector<geometry_msgs::msg::Pose> waypoints;
            waypoints.push_back(target_pose);

            moveit_msgs::msg::RobotTrajectory trajectory;
            const double eef_step = 0.01;

            move_group_arm->setStartStateToCurrentState();
            move_group_arm->setMaxVelocityScalingFactor(maxVelocity);
            move_group_arm->setMaxAccelerationScalingFactor(maxVelocity);

            for (int cart_attempt = 1; cart_attempt <= MAX_CARTESIAN_ATTEMPTS; ++cart_attempt)
            {
                double fraction = move_group_arm->computeCartesianPath(waypoints, eef_step, trajectory);

                if (fraction >= MIN_CARTESIAN_FRACTION)
                {
                    RCLCPP_INFO(this->get_logger(), "Planejamento Cartesiano bem-sucedido (%.1f%%). Executando...", fraction * 100.0);
                    auto exec_result = move_group_arm->execute(trajectory);

                    if (exec_result == moveit::core::MoveItErrorCode::SUCCESS)
                    {
                        task_success = true;
                        break;
                    }
                    else
                    {
                        RCLCPP_WARN(this->get_logger(), "Execução Cartesiana falhou. Tentando novamente...");
                    }
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "Planejamento Cartesiano falhou (fração: %.2f). Tentativa %d/%d", fraction, cart_attempt, MAX_CARTESIAN_ATTEMPTS);
                }
            }

            if (!task_success)
            {
                RCLCPP_ERROR(this->get_logger(), "Falha no planejamento Cartesiano após %d tentativas. Recorrendo ao planejamento normal.", MAX_CARTESIAN_ATTEMPTS);
            }
        }

        if (task_success)
        {
            break;
        }

        RCLCPP_INFO(this->get_logger(), "Tentando planejamento normal (free-space)...");

        move_group_arm->setStartStateToCurrentState();
        move_group_arm->setPlannerId("RRTConnectkConfigDefault");
        move_group_arm->setPoseTarget(target_pose, "suction_tip");
        move_group_arm->setPlanningTime(10.0);
        move_group_arm->setNumPlanningAttempts(100);
        move_group_arm->setMaxVelocityScalingFactor(maxVelocity);
        move_group_arm->setMaxAccelerationScalingFactor(maxVelocity);
        move_group_arm->setGoalPositionTolerance(0.0001);
        move_group_arm->setGoalOrientationTolerance(0.0001);

        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        auto plan_result = move_group_arm->plan(my_plan);

        if (plan_result != moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_WARN(this->get_logger(), "Planejamento normal falhou nesta tentativa.");
            continue;
        }

        RCLCPP_INFO(this->get_logger(), "Planejamento normal bem-sucedido. Executando...");
        auto exec_result = move_group_arm->execute(my_plan);

        if (exec_result != moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_WARN(this->get_logger(), "Execução normal falhou nesta tentativa.");
            continue;
        }

        RCLCPP_INFO(this->get_logger(), "Execução normal bem-sucedida.");
        // 2. [CORREÇÃO DE BUG] Defina task_success como true aqui
        task_success = true; 
        break;
    }

    // 3. Adicione o log de falha final e retorne o status
    if (!task_success)
    {
        RCLCPP_ERROR(this->get_logger(), "Falha no planejamento e execução após %d ciclos.", MAX_PLANNING_CYCLES);
    }

    return task_success;
}

    void calculate_global_pose(std::string received_id, geometry_msgs::msg::Pose pose)
    {
        std::string id = received_id;
        size_t pos = received_id.find('_');
        if (pos != std::string::npos) {
            id = received_id.substr(0, pos);
        }

        if (pick_and_place_poses.find(id) == pick_and_place_poses.end()) {
            RCLCPP_WARN(this->get_logger(), "ID '%s' não encontrado no YAML.", id.c_str());
            return;
        }

        const auto &poses = pick_and_place_poses[id];

        // --- Itera sobre as poses de aproximação e coleta do objeto ---
        for (size_t i = 0; i < poses.size(); ++i) {
            const auto &pose_local = poses[i];

            // --- Converte posição local do YAML em posição global (frame 'world') ---
            tf2::Vector3 local_point(pose_local.position.x, pose_local.position.y, pose_local.position.z);
            tf2::Quaternion q_object(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
            tf2::Vector3 t_object(pose.position.x, pose.position.y, pose.position.z);
            tf2::Transform object_transform(q_object, t_object);

            tf2::Vector3 world_point = object_transform * local_point;

            geometry_msgs::msg::Pose target_pose_world;
            target_pose_world.position.x = world_point.x();
            target_pose_world.position.y = world_point.y();
            target_pose_world.position.z = world_point.z();
            // A orientação da garra é definida pelo YAML, não pela orientação do objeto
            target_pose_world.orientation = pose_local.orientation; 

            RCLCPP_INFO(this->get_logger(),
                        "Pose de Coleta %zu (world): x=%.3f, y=%.3f, z=%.3f",
                        i, world_point.x(), world_point.y(), world_point.z());

            // --- Transforma a pose alvo do frame 'world' para 'panda_link0' ---
            geometry_msgs::msg::PoseStamped pose_in_world, pose_in_base;
            pose_in_world.header.frame_id = "world";
            // Use um timestamp zerado para obter a transformação mais recente disponível
            pose_in_world.header.stamp = tf2_ros::toMsg(tf2::TimePointZero);
            pose_in_world.pose = target_pose_world;

            try {
                // Aumenta o timeout para dar mais tempo à busca da transformação
                pose_in_base = tf_buffer_->transform(pose_in_world, "panda_link0", tf2::durationFromSec(2.0));
            } catch (const tf2::TransformException &ex) {
                RCLCPP_ERROR(this->get_logger(), "Falha ao transformar de 'world' -> 'panda_link0' para coleta: %s", ex.what());
                // Se uma pose falhar, pula para a próxima iteração
                continue;
            }

            // --- Move o braço até a pose e ativa a sucção ---
            if (positions_for_arm(pose_in_base.pose, 0.5, false)) 
            {
                rclcpp::sleep_for(std::chrono::milliseconds(1000));
                publish_suction_activation(true);
                move_group_arm->attachObject(received_id, "suction_tip"); 
                rclcpp::sleep_for(std::chrono::milliseconds(500));
            }
        }

        // --- Lógica simplificada para o movimento de descarte ---
        // Defina uma pose de descarte diretamente no frame 'world'
        geometry_msgs::msg::Pose place_pose_world;
        place_pose_world.position.x = 0.5;  // Posição de descarte fixa no mundo
        place_pose_world.position.y = -0.5;
        place_pose_world.position.z = 0.4;
        place_pose_world.orientation.w = 1.0; // Orientação neutra

        RCLCPP_INFO(this->get_logger(), "Movendo para a pose de descarte.");

        // --- Transforma a pose de descarte do frame 'world' para 'panda_link0' ---
        geometry_msgs::msg::PoseStamped final_pose_world, final_pose_base;
        final_pose_world.header.frame_id = "world";
        final_pose_world.header.stamp = tf2_ros::toMsg(tf2::TimePointZero);
        final_pose_world.pose = place_pose_world;

        try {
            final_pose_base = tf_buffer_->transform(final_pose_world, "panda_link0", tf2::durationFromSec(2.0));
            
            // --- Move para a posição e solta o objeto ---
            if (positions_for_arm(final_pose_base.pose, 0.5, false)) 
            {
                rclcpp::sleep_for(std::chrono::milliseconds(1000));
                publish_suction_activation(false);
                move_group_arm->detachObject(received_id);
                rclcpp::sleep_for(std::chrono::milliseconds(500));
            }
        } catch (const tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Falha ao transformar pose de descarte de 'world' -> 'panda_link0': %s", ex.what());
        }
    }
    
    geometry_msgs::msg::Pose transform_pose_to_base(const geometry_msgs::msg::Pose &pose_in_tool)
    {
        geometry_msgs::msg::PoseStamped pose_in, pose_out;
        pose_in.header.frame_id = "suction_tip";
        pose_in.header.stamp = this->get_clock()->now();
        pose_in.pose = pose_in_tool;

        try {
            pose_out = tf_buffer_->transform(pose_in, "panda_link0", tf2::durationFromSec(1.0));
            RCLCPP_INFO(this->get_logger(),
                        "Pose transformada para base: x=%.3f, y=%.3f, z=%.3f",
                        pose_out.pose.position.x, pose_out.pose.position.y, pose_out.pose.position.z);
            return pose_out.pose;
        } catch (const tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Falha ao transformar para base: %s", ex.what());
            return pose_in.pose; // fallback
        }
    }
        
        geometry_msgs::msg::Pose transform_pose_to_world(const geometry_msgs::msg::Pose &pose_in_base)
    {
        geometry_msgs::msg::PoseStamped pose_in, pose_out;
        pose_in.header.frame_id = "panda_link0";
        pose_in.header.stamp = this->get_clock()->now();
        pose_in.pose = pose_in_base;

        try {
            pose_out = tf_buffer_->transform(pose_in, "world", tf2::durationFromSec(1.0));
            RCLCPP_INFO(this->get_logger(),
                        "Pose transformada para global: x=%.3f, y=%.3f, z=%.3f",
                        pose_out.pose.position.x, pose_out.pose.position.y, pose_out.pose.position.z);
            return pose_out.pose;
        } catch (const tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "Falha ao transformar para global: %s", ex.what());
            return pose_in.pose;
        }
    }


    /*
    
        PUBLISHERS.
    
    */

 

    void publish_suction_activation(bool activation)
    {
        auto msg = std_msgs::msg::Bool();
        msg.data = activation;

        publisher_2->publish(msg);
    }

    

    /*
    
        Servers.

    */

    void handle_request(const std::shared_ptr<object_manipulation_interfaces::srv::PickedObject::Request> request, std::shared_ptr<object_manipulation_interfaces::srv::PickedObject::Response> response)
    {

        std::get<0>(object) = request->id;
        std::get<1>(object) = request->pose;

        bool success = true;  

        response->success = success;

        if (success)
        {
            RCLCPP_INFO(this->get_logger(), "Processamento concluído com sucesso!");
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Falha ao processar o pedido!");
        }
    }

    void handle_controller_request(const std::shared_ptr<object_manipulation_interfaces::srv::GoalReached::Request> request, std::shared_ptr<object_manipulation_interfaces::srv::GoalReached::Response> response)
    {

        calculate_global_pose(std::get<0>(object), std::get<1>(object));


        std::cout << "toma" << std::endl;

        bool success = true;  

        response->success = success;

        if (success)
        {
            RCLCPP_INFO(this->get_logger(), "Processamento concluído com sucesso!");
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Falha ao processar o pedido!");
        }
    }


public:
    SimpleManipulation()
     : Node("pick_and_organize")
    {
        this->declare_parameter<std::string>("yaml_file", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
    
        moveit_node_ = std::make_shared<rclcpp::Node>("pick_and_organize_moveit_node");

        executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
        
        executor_->add_node(moveit_node_);
        executor_thread_ = std::thread([this]() { this->executor_->spin(); });
      


        publisher_2 = this->create_publisher<std_msgs::msg::Bool>("/surface_gripper", 10);
    
        service_ = this->create_service<object_manipulation_interfaces::srv::PickedObject>("/picked_object",std::bind(&SimpleManipulation::handle_request, this, std::placeholders::_1, std::placeholders::_2));
        
        service_1 = this->create_service<object_manipulation_interfaces::srv::GoalReached>("/goal_reached",std::bind(&SimpleManipulation::handle_controller_request, this, std::placeholders::_1, std::placeholders::_2));

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&SimpleManipulation::initMoveGroup, this));
        
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        loadLocationsFromYaml(yaml_file);
    }   

    ~SimpleManipulation()
    {
        executor_->cancel();
        if (executor_thread_.joinable())
        {
            executor_thread_.join();
        }
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SimpleManipulation>());
  rclcpp::shutdown();
  return 0;
}