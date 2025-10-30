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


class PickAndOrganize : public rclcpp::Node {

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
    
    //Timer.
    rclcpp::TimerBase::SharedPtr init_timer_;

   
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;

    rclcpp::Node::SharedPtr moveit_node_;
    rclcpp::Executor::SharedPtr executor_;
    std::thread executor_thread_;

    std::string yaml_file, storages_yaml_file;
    std::unordered_map<std::string, std::pair<std::string, std::vector<geometry_msgs::msg::Pose>>> pick_and_place_poses;
    std::unordered_map<std::string, LocationData> storages;
    std::unordered_map<std::string, StorageData> storage_axis_counts;


    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);

            for (const auto &label_node : config)
            {
                const std::string label = label_node.first.as<std::string>();
                const YAML::Node &items = label_node.second;

                std::string storage_id = "none";
                std::vector<geometry_msgs::msg::Pose> poses;

                if (!items.IsSequence())
                {
                    RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                "[%s] Estrutura inválida: esperado sequence (lista).", label.c_str());
                    continue;
                }

                // 🔹 Percorre a sequência do YAML
                for (const auto &item : items)
                {
                    if (!item.IsMap() || item.size() != 1)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] Ignorando entrada inválida.", label.c_str());
                        continue;
                    }

                    const std::string key = item.begin()->first.as<std::string>();
                    const YAML::Node &value = item.begin()->second;

                    // 🔹 Detecta storage
                    if (key == "storage")
                    {
                        storage_id = value.as<std::string>();
                        RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                                    "[%s] Storage definido: %s", label.c_str(), storage_id.c_str());
                        continue;
                    }

                    // 🔹 Caso seja uma localização
                    if (!value["position"] || !value["orientation"])
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] '%s' sem position/orientation", label.c_str(), key.c_str());
                        continue;
                    }

                    const YAML::Node &pos = value["position"];
                    const YAML::Node &ori = value["orientation"];

                    if (pos.size() != 3 || ori.size() != 4)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] '%s' posição/orientação inválida", label.c_str(), key.c_str());
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

                    poses.push_back(pose);

                    RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                                "Loaded [%s - %s] -> pos:[%.2f, %.2f, %.2f], ori:[%.2f, %.2f, %.2f, %.2f]",
                                label.c_str(), key.c_str(),
                                pose.position.x, pose.position.y, pose.position.z,
                                pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w);
                }

                pick_and_place_poses[label] = std::make_pair(storage_id, poses);
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                        "Erro ao carregar YAML '%s': %s", yaml_path.c_str(), e.what());
        }
    }

    void loadStoragesLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);

            for (const auto &label_node : config)
            {
                const std::string label = label_node.first.as<std::string>();
                const YAML::Node &locations_node = label_node.second;

                if (!locations_node.IsSequence())
                {
                    RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                "[%s] A entrada não é uma sequência (lista). Ignorando.", label.c_str());
                    continue;
                }

                
                for (const auto &loc_item : locations_node)
                {
                    
                    if (!loc_item.IsMap() || loc_item.size() != 1)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] Ignorando entrada inválida de localização (não é mapa ou tem mais de uma chave).", label.c_str());
                        continue;
                    }

                    
                    const auto &loc_name = loc_item.begin()->first.as<std::string>();
                    const YAML::Node &loc_data = loc_item.begin()->second;

                  
                    if (!loc_data["position"] || !loc_data["orientation"] || !loc_data["size"])
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] '%s' não possui 'position', 'orientation' ou 'size'",
                                    label.c_str(), loc_name.c_str());
                        continue;
                    }

                    const YAML::Node &pos = loc_data["position"];
                    const YAML::Node &ori = loc_data["orientation"];
                    const YAML::Node &size = loc_data["size"];

                    
                    if (!pos.IsSequence() || pos.size() != 3 ||
                        !ori.IsSequence() || ori.size() != 4 ||
                        !size.IsSequence() || size.size() != 3)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] '%s' tamanho inválido para 'position'(3), 'orientation'(4) ou 'size'(3)",
                                    label.c_str(), loc_name.c_str());
                        continue;
                    }

                    LocationData current_data;
                    current_data.pose.position.x = pos[0].as<double>();
                    current_data.pose.position.y = pos[1].as<double>();
                    current_data.pose.position.z = pos[2].as<double>();
                    current_data.pose.orientation.x = ori[0].as<double>();
                    current_data.pose.orientation.y = ori[1].as<double>();
                    current_data.pose.orientation.z = ori[2].as<double>();
                    current_data.pose.orientation.w = ori[3].as<double>();

                    current_data.size.x = size[0].as<double>();
                    current_data.size.y = size[1].as<double>();
                    current_data.size.z = size[2].as<double>();

                  
                    storages[label] = current_data;

                    RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                                "Carregado (sob categoria [%s]) [%s] -> pos:[%.2f, %.2f, %.2f], ori:[%.2f, %.2f, %.2f, %.2f], size:[%.2f, %.2f, %.2f]",
                                label.c_str(), loc_name.c_str(), 
                                current_data.pose.position.x, current_data.pose.position.y, current_data.pose.position.z,
                                current_data.pose.orientation.x, current_data.pose.orientation.y,
                                current_data.pose.orientation.z, current_data.pose.orientation.w,
                                current_data.size.x, current_data.size.y, current_data.size.z);
                }
       
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                        "Falha ao carregar o arquivo YAML '%s': %s", yaml_path.c_str(), e.what());
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

   
    void positions_for_arm(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool computeCartesian) 
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return;
        }

    
        const int MAX_PLANNING_CYCLES = 100; 
        const int MAX_CARTESIAN_ATTEMPTS = 5; 
        const double MIN_CARTESIAN_FRACTION = 0.99; 

   
        for (int cycle = 1; cycle <= MAX_PLANNING_CYCLES; ++cycle)
        {
            RCLCPP_INFO(this->get_logger(), "Ciclo de Planejamento Externo: Tentativa %d de %d", cycle, MAX_PLANNING_CYCLES);

            bool task_success = false; 

            
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
                    else {
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
            move_group_arm->setPlanningTime(4.0);
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
            break; 
        
        } 
    }

   
    geometry_msgs::msg::Pose random_pose(double x_min, double x_max, double y_min, double y_max, double z_min, double z_max)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::uniform_real_distribution<double> dist_x(x_min, x_max);
        std::uniform_real_distribution<double> dist_y(y_min, y_max);
        std::uniform_real_distribution<double> dist_z(z_min, z_max);
        std::uniform_real_distribution<double> dist_angle(0.0, 2 * M_PI);

        geometry_msgs::msg::Pose pose;

        pose.position.x = dist_x(gen);
        pose.position.y = dist_y(gen);
        pose.position.z = dist_z(gen);

        tf2::Quaternion q;
        q.setRPY(dist_angle(gen), dist_angle(gen), dist_angle(gen));
        q.normalize();

        pose.orientation.x = 1.0;
        pose.orientation.y = 0.0;
        pose.orientation.z = 0.0;
        pose.orientation.w = 0.0;

        return pose;
    }

    void calculate_global_pose(std::string received_id, geometry_msgs::msg::Pose pose, geometry_msgs::msg::Vector3 size)
    {
        std::string storage_id, id;

        size_t pos = received_id.find('_');
         
        if (pos != std::string::npos) 
        {
            id = received_id.substr(0, pos);  
        } 
        else
        {
            id = received_id;  
        }

      
        if (pick_and_place_poses.find(id) != pick_and_place_poses.end())  
        {
            const auto &storage_and_poses = pick_and_place_poses.at(id);
            storage_id = storage_and_poses.first;
            const auto &poses = storage_and_poses.second;
            geometry_msgs::msg::Pose target_pose;

            RCLCPP_INFO(this->get_logger(),
                        "Iniciando pick-and-place para '%s' (storage: %s)",
                        id.c_str(), storage_id.c_str());

            for (size_t i = 0; i < poses.size(); ++i)
            {
                const auto &pose_local = poses[i];

                tf2::Vector3 local_corner(
                    pose_local.position.x,
                    pose_local.position.y,
                    pose_local.position.z);

                tf2::Quaternion q(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w);

                tf2::Matrix3x3 rot(q);
                tf2::Vector3 translation(
                    pose.position.x,
                    pose.position.y,
                    pose.position.z);

                tf2::Vector3 world_corner = local_corner + translation;

                
                target_pose.position.x = world_corner.x();
                target_pose.position.y = world_corner.y();
                target_pose.position.z = world_corner.z();
                target_pose.orientation = pose_local.orientation;

                RCLCPP_INFO(this->get_logger(),
                            "Pose %zu - global point: x=%.6f, y=%.6f, z=%.6f",
                            i, world_corner.x(), world_corner.y(), world_corner.z());
                
                
                rclcpp::sleep_for(std::chrono::milliseconds(100));
                positions_for_arm(target_pose, 0.25, false);
                
                rclcpp::sleep_for(std::chrono::milliseconds(500));
                publish_suction_activation(true);

                move_group_arm->attachObject(received_id, "suction_tip");
                
                rclcpp::sleep_for(std::chrono::milliseconds(150));
            }
            target_pose.position.z += 0.25;

            positions_for_arm(target_pose, 0.25, false);

            rclcpp::sleep_for(std::chrono::milliseconds(300));

            geometry_msgs::msg::Pose storage_pose = storages[storage_id].pose;
            geometry_msgs::msg::Vector3 storage_size;
            storage_size.x = static_cast<float>(storages[storage_id].size.x);
            storage_size.y = static_cast<float>(storages[storage_id].size.y);
            storage_size.z = static_cast<float>(storages[storage_id].size.z);

            geometry_msgs::msg::Vector3 object_size = size;


            geometry_msgs::msg::Pose final_pose = placeObjectInBox(storage_id, storage_pose, storage_size, pose.orientation, object_size);

            RCLCPP_INFO(this->get_logger(),
                            "Pose - global point: x=%.3f, y=%.3f, z=%.3f, orientation.x=%.3f, orientation.y=%.3f, orientation.z=%.3f, orientation.w=%.3f",
            final_pose.position.x, final_pose.position.y, final_pose.position.z, final_pose.orientation.x, final_pose.orientation.y, final_pose.orientation.z, final_pose.orientation.w);

            geometry_msgs::msg::Pose temp_pose = final_pose;
            temp_pose.position.z += 0.1;
            positions_for_arm(temp_pose, 0.25, false);

            rclcpp::sleep_for(std::chrono::milliseconds(300));

            positions_for_arm(final_pose, 0.25, true);
            publish_suction_activation(false);
            move_group_arm->detachObject(received_id);
            rclcpp::sleep_for(std::chrono::milliseconds(750));

           
            
            rclcpp::sleep_for(std::chrono::milliseconds(50));

            
            RCLCPP_INFO(this->get_logger(), "Objeto '%s' será armazenado em: %s",
                        received_id.c_str(),
                        storage_id.c_str());
            

            

            positions_for_arm(temp_pose, 0.25, true);
            // ready();

            rclcpp::sleep_for(std::chrono::milliseconds(100));

            
        }
    }


    

    geometry_msgs::msg::Pose placeObjectInBox(
        const std::string &storage_id,
        const geometry_msgs::msg::Pose &storage_pose,
        const geometry_msgs::msg::Vector3 &storage_size,
        const geometry_msgs::msg::Quaternion &object_orientation,
        const geometry_msgs::msg::Vector3 &object_size)
    {
        geometry_msgs::msg::Pose object_pose;

        float compensation = 0.025;
    
        if (storage_axis_counts.find(storage_id) == storage_axis_counts.end() || storage_axis_counts[storage_id].z >= 1)
        {
            object_pose.position.x = storage_pose.position.x - (storage_size.x / 2.0);
            object_pose.position.y = storage_pose.position.y - (storage_size.y / 2.0);
            object_pose.position.z = object_size.z * (2.5 + storage_axis_counts[storage_id].z);


            storage_axis_counts[storage_id].max_x_objects = storage_size.x / (object_size.x + (compensation / 2));
            storage_axis_counts[storage_id].max_y_objects = storage_size.y / (object_size.y + (compensation / 2));
            storage_axis_counts[storage_id].max_z_objects = 1.0 / (object_size.z + (compensation / 2));
            storage_axis_counts[storage_id].x = 1;
            storage_axis_counts[storage_id].y = 1;
            storage_axis_counts[storage_id].z += 0;
            storage_axis_counts[storage_id].direction = 1;
            storage_axis_counts[storage_id].pose = object_pose;
            storage_axis_counts[storage_id].size = object_size;

            
        }
        else
        {
            if(storage_axis_counts[storage_id].x < storage_axis_counts[storage_id].max_x_objects)
            {
                object_pose.position.x = storage_axis_counts[storage_id].pose.position.x + (((storage_axis_counts[storage_id].size.x / 2.0) + (object_size.x / 2.0) + compensation) * storage_axis_counts[storage_id].direction);
                object_pose.position.y = storage_axis_counts[storage_id].pose.position.y;
                object_pose.position.z = object_size.z * (2.5 + storage_axis_counts[storage_id].z);

                storage_axis_counts[storage_id].x += 1;

                storage_axis_counts[storage_id].pose = object_pose;
                storage_axis_counts[storage_id].size = object_size;
            }
            else if(storage_axis_counts[storage_id].y < storage_axis_counts[storage_id].max_y_objects)
            {
                object_pose.position.x = storage_axis_counts[storage_id].pose.position.x;
                object_pose.position.y = storage_axis_counts[storage_id].pose.position.y + (storage_axis_counts[storage_id].size.y / 2.0) + (object_size.y / 2.0) + compensation;
                object_pose.position.z = object_size.z * (2.5 + storage_axis_counts[storage_id].z);
                storage_axis_counts[storage_id].y += 1;
                
                if(storage_axis_counts[storage_id].direction == 1)
                {
                    storage_axis_counts[storage_id].direction = -1;
                }
                else
                {
                    storage_axis_counts[storage_id].direction = 1;
                }
                
                storage_axis_counts[storage_id].pose = object_pose;
                storage_axis_counts[storage_id].size = object_size;

                storage_axis_counts[storage_id].x = 1;
            }
            else if(storage_axis_counts[storage_id].z < storage_axis_counts[storage_id].max_z_objects)
            {
                storage_axis_counts[storage_id].z += 1;
                object_pose.position.x = storage_axis_counts[storage_id].pose.position.x;
                object_pose.position.y = storage_axis_counts[storage_id].pose.position.y + (storage_axis_counts[storage_id].size.y / 2.0) + (object_size.y / 2.0) + compensation;
                object_pose.position.z = object_size.z * (2.5 + storage_axis_counts[storage_id].z);
                storage_axis_counts[storage_id].z += 1;
              
        
                storage_axis_counts[storage_id].x = 1;
                storage_axis_counts[storage_id].y = 1;
               
            }
           
        }

        tf2::Quaternion q_obj(
            object_orientation.x,
            object_orientation.y,
            object_orientation.z,
            object_orientation.w);

        double roll_obj, pitch_obj, yaw_obj;
        tf2::Matrix3x3(q_obj).getRPY(roll_obj, pitch_obj, yaw_obj);

      
        double roll = M_PI;
        double pitch = 0.0;
        double yaw = -yaw_obj;  

        tf2::Quaternion q_final;
        q_final.setRPY(roll, pitch, yaw);
        q_final.normalize();

        object_pose.orientation = tf2::toMsg(q_final);

        return object_pose;
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

        calculate_global_pose(request->id, request->pose, request->size);

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
    PickAndOrganize()
     : Node("pick_and_organize")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("storages_yaml_file", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        storages_yaml_file = this->get_parameter("storages_yaml_file").as_string();
    
        moveit_node_ = std::make_shared<rclcpp::Node>("pick_and_organize_moveit_node");

        executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
        
        executor_->add_node(moveit_node_);
        executor_thread_ = std::thread([this]() { this->executor_->spin(); });
      


        publisher_2 = this->create_publisher<std_msgs::msg::Bool>("/surface_gripper", 10);
    
        service_ = this->create_service<object_manipulation_interfaces::srv::PickedObject>("/picked_object",std::bind(&PickAndOrganize::handle_request, this, std::placeholders::_1, std::placeholders::_2));
        
    
        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&PickAndOrganize::initMoveGroup, this));
        
        loadLocationsFromYaml(yaml_file);
        loadStoragesLocationsFromYaml(storages_yaml_file);
    }   

    ~PickAndOrganize()
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
  rclcpp::spin(std::make_shared<PickAndOrganize>());
  rclcpp::shutdown();
  return 0;
}