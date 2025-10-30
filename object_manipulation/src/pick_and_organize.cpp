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



    //Publishers.
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher_2;
    
    //Service.
    rclcpp::Service<object_manipulation_interfaces::srv::PickedObject>::SharedPtr service_;
    
    //Timer.
    rclcpp::TimerBase::SharedPtr init_timer_;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;

    std::string yaml_file, storages_yaml_file;

    std::unordered_map<std::string, std::pair<std::string, std::vector<geometry_msgs::msg::Pose>>> pick_and_place_poses;
    std::unordered_map<std::string, LocationData> storages;


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
            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "panda_arm"); 
            

            RCLCPP_INFO(this->get_logger(), "MoveGroup (arm e gripper) inicializados com sucesso.");

            init_timer_->cancel();  
        } 
        catch (const std::exception &e) 
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }

    }

    void positions_for_arm(const geometry_msgs::msg::Pose &target_pose) 
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return;
        }

        const int MAX_PLANNING_CYCLES = 100;
    

        for (int cycle = 1; cycle <= MAX_PLANNING_CYCLES; ++cycle)
        {
            RCLCPP_INFO(this->get_logger(), "Ciclo de Planejamento Externo: Tentativa %d de %d", cycle, MAX_PLANNING_CYCLES);

        
            // move_group_arm->setWorkspace(-1.5, -1.5, 0.1, 1.5, 1.5, 1.5);
            move_group_arm->setStartStateToCurrentState(); 
            move_group_arm->setPlannerId("RRTConnectkConfigDefault");
            move_group_arm->setPoseTarget(target_pose, "suction_tip"); 
            move_group_arm->setPlanningTime(2.0);
            move_group_arm->setNumPlanningAttempts(100); 
            move_group_arm->setMaxVelocityScalingFactor(1.0);
            move_group_arm->setMaxAccelerationScalingFactor(1.0);
            move_group_arm->setGoalTolerance(0.001);


            moveit::planning_interface::MoveGroupInterface::Plan my_plan;
            auto plan_result = move_group_arm->plan(my_plan);

           
            if (plan_result != moveit::core::MoveItErrorCode::SUCCESS) 
            {
                
                
                continue; 
            }

            
            auto exec_result = move_group_arm->execute(my_plan);

            if (exec_result != moveit::core::MoveItErrorCode::SUCCESS)
            {
                continue; 
            }

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

                geometry_msgs::msg::Pose target_pose;
                target_pose.position.x = world_corner.x();
                target_pose.position.y = world_corner.y();
                target_pose.position.z = world_corner.z();
                target_pose.orientation = pose_local.orientation;

                RCLCPP_INFO(this->get_logger(),
                            "Pose %zu - global point: x=%.3f, y=%.3f, z=%.3f",
                            i, world_corner.x(), world_corner.y(), world_corner.z());
                
                
                rclcpp::sleep_for(std::chrono::milliseconds(100));
                positions_for_arm(target_pose);
                
                rclcpp::sleep_for(std::chrono::milliseconds(1000));
                publish_suction_activation(true);

                move_group_arm->attachObject(received_id, "suction_tip");
                
                rclcpp::sleep_for(std::chrono::milliseconds(150));
            }

            geometry_msgs::msg::Pose storage_pose = storages[storage_id].pose;
            geometry_msgs::msg::Vector3 storage_size;
            storage_size.x = static_cast<float>(storages[storage_id].size.x);
            storage_size.y = static_cast<float>(storages[storage_id].size.y);
            storage_size.z = static_cast<float>(storages[storage_id].size.z);

         

            moveit::planning_interface::PlanningSceneInterface psi;
            auto objects = psi.getObjects({received_id});
            double x, y, z;

            if (!objects.empty())
            {
                const auto &obj = objects.at(received_id);
                if (!obj.primitives.empty())
                {
                    const shape_msgs::msg::SolidPrimitive &primitive = obj.primitives[0];
                    if (primitive.type == shape_msgs::msg::SolidPrimitive::BOX)
                    {
                        x = primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_X];
                        y = primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Y];
                        z = primitive.dimensions[shape_msgs::msg::SolidPrimitive::BOX_Z];

                        RCLCPP_INFO(this->get_logger(), "Tamanho do objeto: %.3f, %.3f, %.3f", x, y, z);
                    }
                }
            }

            // geometry_msgs::msg::Pose object_pose = pose;
            geometry_msgs::msg::Vector3 object_size;
            object_size.x = static_cast<float>(x);
            object_size.y = static_cast<float>(y);
            object_size.z = static_cast<float>(z);

           
        

            geometry_msgs::msg::Pose final_pose = placeObjectInBox(storage_pose, storage_size, object_size);

            RCLCPP_INFO(this->get_logger(),
                            "Pose - global point: x=%.3f, y=%.3f, z=%.3f, orientation.x=%.3f, orientation.y=%.3f, orientation.z=%.3f, orientation.w=%.3f",
            final_pose.position.x, final_pose.position.y, final_pose.position.z, final_pose.orientation.x, final_pose.orientation.y, final_pose.orientation.z, final_pose.orientation.w);

            positions_for_arm(final_pose);

            rclcpp::sleep_for(std::chrono::milliseconds(1000));
            RCLCPP_INFO(this->get_logger(), "Objeto '%s' será armazenado em: %s",
                        received_id.c_str(),
                        storage_id.c_str());
            publish_suction_activation(false);

            move_group_arm->detachObject(received_id);
        }
    }




    std::pair<geometry_msgs::msg::Pose, geometry_msgs::msg::Vector3> last_object;
    int contador = 0;

    geometry_msgs::msg::Pose placeObjectInBox(const geometry_msgs::msg::Pose &box_pose, const geometry_msgs::msg::Vector3 &box_size,
        const geometry_msgs::msg::Vector3 &obj_size)
    {
        geometry_msgs::msg::Pose object_pose;

        if(contador == 0)
        {
            object_pose.position.x = box_pose.position.x - (box_size.x / 2.0);
            object_pose.position.y = box_pose.position.y - (box_size.y / 2.0);
            object_pose.position.z = 0.1;
            
            std::get<0>(last_object) = object_pose;
            std::get<1>(last_object) = obj_size;
            contador = 1;
        }
        else
        {
            object_pose.position.x = std::get<0>(last_object).position.x + (std::get<1>(last_object).x / 2.0) + (obj_size.x / 2.0) + 0.02;
            object_pose.position.y = std::get<0>(last_object).position.y + (std::get<1>(last_object).y / 2.0) + (obj_size.y / 2.0) + 0.02;
            object_pose.position.z = 0.1;
            
            std::cout << std::get<0>(last_object).position.x << " " << std::get<0>(last_object).position.y << std::endl;
            std::cout << "size" << std::get<1>(last_object).x << " " << std::get<1>(last_object).y << std::endl;
            std::get<0>(last_object) = object_pose;
            std::get<1>(last_object) = obj_size;
        }
     
        
    
    
    

        
        

        object_pose.orientation.x = 1.0;
        object_pose.orientation.y = 0.0;
        object_pose.orientation.z = 0.0;
        object_pose.orientation.w = 0.0;
        

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
    
        publisher_2 = this->create_publisher<std_msgs::msg::Bool>("/surface_gripper", 10);
    
        service_ = this->create_service<object_manipulation_interfaces::srv::PickedObject>("/picked_object",std::bind(&PickAndOrganize::handle_request, this, std::placeholders::_1, std::placeholders::_2));
        
        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&PickAndOrganize::initMoveGroup, this));
        
        loadLocationsFromYaml(yaml_file);
        loadStoragesLocationsFromYaml(storages_yaml_file);
    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PickAndOrganize>());
  rclcpp::shutdown();
  return 0;
}

