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


class PickAndPlaceConveyor : public rclcpp::Node {

private:

    struct LocationData
    {
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size;
    };

    //Publishers.
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher_2;

    //Subscriptions.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_1;

    //Service.
    rclcpp::Client<object_manipulation_interfaces::srv::ObjectCollision>::SharedPtr client_;
    
    //Timer.
    rclcpp::TimerBase::SharedPtr init_timer_;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;

    std::string yaml_file, storages_yaml_file;

    std::unordered_map<std::string, std::vector<geometry_msgs::msg::Pose>> pick_and_place_poses;
    std::unordered_map<std::string, std::vector<LocationData>> storages;

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

    void loadStoragesLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);

            for (const auto &label_node : config)
            {
                const std::string label = label_node.first.as<std::string>();
                const YAML::Node &locations_node = label_node.second;

                
                std::vector<LocationData> locations;

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
                                    "[%s] Ignorando entrada inválida de localização.", label.c_str());
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

                    // Valida os tamanhos dos vetores
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

                    locations.push_back(current_data);

                    RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                                "Carregado [%s - %s] -> pos:[%.2f, %.2f, %.2f], ori:[%.2f, %.2f, %.2f, %.2f], size:[%.2f, %.2f, %.2f]",
                                label.c_str(), loc_name.c_str(),
                                current_data.pose.position.x, current_data.pose.position.y, current_data.pose.position.z,
                                current_data.pose.orientation.x, current_data.pose.orientation.y,
                                current_data.pose.orientation.z, current_data.pose.orientation.w,
                                current_data.size.x, current_data.size.y, current_data.size.z);
                }

              
                storages[label] = locations;
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                        "Falha ao carregar o arquivo YAML '%s': %s", yaml_path.c_str(), e.what());
        }
    }


    void initMoveGroup() {
        try 
        {
            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "panda_arm"); 
            

            RCLCPP_INFO(this->get_logger(), "MoveGroup (arm e gripper) inicializados com sucesso.");

            init_timer_->cancel();  
        } catch (const std::exception &e) 
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

    void return_to_origin()
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do arm não inicializado.");
            return;
        }
        
        move_group_arm->setJointValueTarget({
            {"panda_joint1", 0.0},
            {"panda_joint2", -0.7853981633974483},
            {"panda_joint3", 0.0},
            {"panda_joint4", -2.356194490192345},
            {"panda_joint5", 0.0},
            {"panda_joint6", 1.5707963267948966},
            {"panda_joint7", 0.7853981633974483},
        });

        auto result = move_group_arm->move();

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            RCLCPP_INFO(this->get_logger(), "Braço retornou à origem.");
        } 
        else 
        {
            RCLCPP_ERROR(this->get_logger(), "Falha ao retornar o braço à origem.");
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

    // void calculate_pose()
    // {
       
    // }
        
        
    /*
    
        PUBLISHERS.
    
    */

    void publish_velocity(float velocity)
    {
        auto message = std_msgs::msg::Float32();
        message.data = velocity;

        publisher_->publish(message);

    }

 

    
    void publish_suction_activation(bool activation)
    {
        auto msg = std_msgs::msg::Bool();
        msg.data = activation;

        publisher_2->publish(msg);
    }

    /*
    
        CALLBACKS.

    */
    
    std::string pick_and_place_id;
    bool stopped = false, welding_done = false;

    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        bool encerrado = false;
        std::string id;
        for (const auto &det : msg->detections)
        {
            
            size_t pos = det.results[0].hypothesis.class_id.find('_'); 
            if (pos != std::string::npos) 
            {
                id = det.results[0].hypothesis.class_id.substr(0, pos);  
            } 
            else
            {
                id = det.results[0].hypothesis.class_id;  
            }

            if (det.results.empty() || pick_and_place_poses.find(id) == pick_and_place_poses.end())
            {
                continue;
            }
            
            if(stopped == false)
            {
                publish_velocity(0.2);
            }


            if(det.bbox.center.position.y < 0.1 && det.bbox.center.position.y > -0.1  && det.bbox.center.position.x > 0.25 && stopped == true && pick_and_place_id == det.results[0].hypothesis.class_id)
            {
                send_request(true);
                geometry_msgs::msg::Pose adjust_pose;

                if (pick_and_place_poses.find(id) != pick_and_place_poses.end())  
                {
                    const auto &poses = pick_and_place_poses[id];  

                    for (size_t i = 0; i < poses.size(); ++i)
                    {
                        const auto &pose_local = poses[i];

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
                        

                        // tf2::Vector3 world_corner = rot * local_corner + translation;
                        tf2::Vector3 world_corner = rot * local_corner + translation;
                        geometry_msgs::msg::Pose target_pose;
                        target_pose.position.x = world_corner.x();
                        target_pose.position.y = world_corner.y();
                        target_pose.position.z = world_corner.z();

                        target_pose.orientation = pose_local.orientation;
                        
                        adjust_pose = target_pose;

                        RCLCPP_INFO(this->get_logger(),
                                    "Pose %zu - global point: x=%.3f, y=%.3f, z=%.3f",
                                    i, world_corner.x(), world_corner.y(), world_corner.z());
                       
                        
                        publish_suction_activation(false);
                        rclcpp::sleep_for(std::chrono::milliseconds(100));
                        positions_for_arm(target_pose);
                        
                        
                        rclcpp::sleep_for(std::chrono::milliseconds(1000));
                        publish_suction_activation(true);
                        move_group_arm->attachObject(det.results[0].hypothesis.class_id, "suction_tip");
                        
                        rclcpp::sleep_for(std::chrono::milliseconds(150));
                        
                        
                        
                    }
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "ID '%s' não encontrado em pick_and_place_poses", det.results[0].hypothesis.class_id.c_str());
                }

                welding_done = true;
                stopped = false;
                
                
                auto pose = random_pose(0.0, 0.1, 0.4, 0.6, 0.2, 0.4);
                positions_for_arm(pose);
                // calculate_pose();

                
                
                rclcpp::sleep_for(std::chrono::milliseconds(1000));

                publish_suction_activation(false);
                
                move_group_arm->detachObject(det.results[0].hypothesis.class_id);
                rclcpp::sleep_for(std::chrono::milliseconds(150));
                send_request(false);
                publish_velocity(0.2);
                rclcpp::sleep_for(std::chrono::milliseconds(50));

                //sim, eu fiz isso.
                encerrado = true;
            }
            else if(det.bbox.center.position.y < 0.1 && det.bbox.center.position.y > -0.1 && det.bbox.center.position.x > 0.25 && stopped == false && pick_and_place_id != det.results[0].hypothesis.class_id)
            {
                publish_velocity(0.0);
                rclcpp::sleep_for(std::chrono::milliseconds(1000));
                pick_and_place_id = det.results[0].hypothesis.class_id;
                welding_done = false;
                stopped = true;
                
                
            }

            if(encerrado)
            {
                break;
            }
           
        }
    }

    void send_request(bool stop_flag)
    {
        auto request = std::make_shared<object_manipulation_interfaces::srv::ObjectCollision::Request>();
        request->stop = stop_flag;

      
        client_->async_send_request(request,
            [this](rclcpp::Client<object_manipulation_interfaces::srv::ObjectCollision>::SharedFuture future_response) {
                auto response = future_response.get();  
                if (response->success) {
                    RCLCPP_INFO(this->get_logger(), "Service executado com sucesso!");
                } else {
                    RCLCPP_WARN(this->get_logger(), "Falha ao executar service");
                }
            }
        );
    }


            

public:
    PickAndPlaceConveyor()
     : Node("pick_and_place_suction_gripper")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("storages_yaml_file", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        storages_yaml_file = this->get_parameter("storages_yaml_file").as_string();
        
        publisher_ = this->create_publisher<std_msgs::msg::Float32>("/conveyor_velocity", 10);
        publisher_2 = this->create_publisher<std_msgs::msg::Bool>("/surface_gripper", 10);

        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&PickAndPlaceConveyor::detectionCallback, this, std::placeholders::_1));

        client_ = this->create_client<object_manipulation_interfaces::srv::ObjectCollision>(
            "/object_collision");

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&PickAndPlaceConveyor::initMoveGroup, this));


        loadLocationsFromYaml(yaml_file);
        loadStoragesLocationsFromYaml(storages_yaml_file);
    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PickAndPlaceConveyor>());
  rclcpp::shutdown();
  return 0;
}

