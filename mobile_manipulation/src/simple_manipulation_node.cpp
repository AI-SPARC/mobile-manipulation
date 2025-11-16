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
#include "object_manipulation_interfaces/srv/object_collision.hpp"
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
    
    //Client.
    rclcpp::Client<object_manipulation_interfaces::srv::ObjectCollision>::SharedPtr client_;

    //Service.
    rclcpp::Service<object_manipulation_interfaces::srv::PickedObject>::SharedPtr service_;
    rclcpp::Service<object_manipulation_interfaces::srv::GoalReached>::SharedPtr service_1;

    //Timer.
    rclcpp::TimerBase::SharedPtr init_timer_;

   
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;

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

                    if (pos.size() != 3 || ori.size() != 3) 
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

                    double roll  = ori[0].as<double>();
                    double pitch = ori[1].as<double>();
                    double yaw   = ori[2].as<double>();

                    tf2::Quaternion q;
                    q.setRPY(roll, pitch, yaw);
                    q.normalize();

                    pose.orientation.x = q.x();
                    pose.orientation.y = q.y();
                    pose.orientation.z = q.z();
                    pose.orientation.w = q.w();

                    locations.push_back(pose);

                    RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                                "Loaded [%s - %s] -> pos:[%.2f, %.2f, %.2f], RPY:[%.2f, %.2f, %.2f], quat:[%.2f, %.2f, %.2f, %.2f]",
                                label.c_str(), loc_name.c_str(),
                                pose.position.x, pose.position.y, pose.position.z,
                                roll, pitch, yaw,
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
            
            move_group_gripper = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
                moveit_node_, "hand"); 

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

    void close_gripper() 
    {
        if (!move_group_gripper) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do GRIPPER não inicializado.");
            return;
        }
        move_group_gripper->setStartStateToCurrentState();
        
        move_group_gripper->setJointValueTarget({
            {"panda_finger_joint1", 0.003},
            {"panda_finger_joint2", 0.003},
        });

        move_group_gripper->allowReplanning(true);
        
        auto result = move_group_gripper->move();

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            RCLCPP_INFO(this->get_logger(), "Gripper fechou (MoveIt).");
        } 
        else 
        {
            RCLCPP_ERROR(this->get_logger(), "Falha ao fechar o gripper.");
        }
    }

    void open_gripper() 
    {
        if (!move_group_gripper) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do GRIPPER não inicializado.");
            return;
        }

        move_group_gripper->setStartStateToCurrentState();

       move_group_gripper->setJointValueTarget({
             {"panda_finger_joint1", 0.038},
             {"panda_finger_joint2", 0.038},
        });
        move_group_gripper->allowReplanning(true);


        auto result = move_group_gripper->move();

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            RCLCPP_INFO(this->get_logger(), "Gripper fechou (MoveIt).");
        } 
        else 
        {
            RCLCPP_ERROR(this->get_logger(), "Falha ao fechar o gripper.");
        }
    }

       
    bool positions_for_arm(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool computeCartesian)
    {
        if (!move_group_arm)
        {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return false;
        }

        const int MAX_PLANNING_CYCLES = 100;
        const int MAX_CARTESIAN_ATTEMPTS = 5;
        const double MIN_CARTESIAN_FRACTION = 0.99;

        bool task_success = false; 

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
            move_group_arm->setPoseTarget(target_pose, "panda_link8");
            move_group_arm->setPlanningTime(4.0);
            move_group_arm->setNumPlanningAttempts(100);
            move_group_arm->setMaxVelocityScalingFactor(maxVelocity);
            move_group_arm->setMaxAccelerationScalingFactor(maxVelocity);
            move_group_arm->setGoalPositionTolerance(0.001);
            move_group_arm->setGoalOrientationTolerance(0.001);


    
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
            task_success = true; 
            break;
        }

        
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
            std::cout << id << std::endl;
        }

        if (pick_and_place_poses.find(id) == pick_and_place_poses.end()) {
            RCLCPP_WARN(this->get_logger(), "ID '%s' não encontrado no YAML.", id.c_str());
            return;
        }

        const auto &poses = pick_and_place_poses[id];

   
        for (size_t i = 0; i < poses.size(); ++i) {
            const auto &pose_local = poses[i];

            tf2::Vector3 local_point(
                pose_local.position.x,
                pose_local.position.y,
                pose_local.position.z
            );

            tf2::Quaternion q_object(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            );

            tf2::Vector3 t_object(
                pose.position.x,
                pose.position.y,
                pose.position.z
            );

            tf2::Transform object_transform(q_object, t_object);
            tf2::Vector3 world_point = object_transform * local_point;

            geometry_msgs::msg::Pose target_pose_world;
            target_pose_world.position.x = world_point.x();
            target_pose_world.position.y = world_point.y();
            target_pose_world.position.z = world_point.z();

            double obj_r, obj_p, obj_y;
            tf2::Matrix3x3(q_object).getRPY(obj_r, obj_p, obj_y);

           
            double off_r, off_p, off_y;
            tf2::Quaternion q_offset(
                pose_local.orientation.x,
                pose_local.orientation.y,
                pose_local.orientation.z,
                pose_local.orientation.w
            );
            tf2::Matrix3x3(q_offset).getRPY(off_r, off_p, off_y);

       
            bool use_roll  = true;
            bool use_pitch = true; 
            bool use_yaw   = true; 

            double final_r = obj_r + (use_roll  ? off_r : 0.0);
            double final_p = obj_p + (use_pitch ? off_p : 0.0);
            double final_y = obj_y + (use_yaw   ? off_y : 0.0);

            tf2::Quaternion q_final;
            q_final.setRPY(final_r, final_p, final_y);
            q_final.normalize();

            target_pose_world.orientation.x = q_final.x();
            target_pose_world.orientation.y = q_final.y();
            target_pose_world.orientation.z = q_final.z();
            target_pose_world.orientation.w = q_final.w();

           
            if (positions_for_arm(target_pose_world, 0.5, false)) 
            {
                rclcpp::sleep_for(std::chrono::milliseconds(200));

                std::vector<std::string> touch_links = move_group_gripper->getLinkNames();
                move_group_arm->attachObject(received_id, "panda_link8", touch_links);

                rclcpp::sleep_for(std::chrono::milliseconds(200));

                close_gripper();
                
                rclcpp::sleep_for(std::chrono::milliseconds(500));
            }
        }


        geometry_msgs::msg::Pose place_pose_base;
        place_pose_base.position.x = -0.25;  
        place_pose_base.position.y = 0.0;
        place_pose_base.position.z = 0.25;
        place_pose_base.orientation.x = 1.0;
        place_pose_base.orientation.y = 0.0;
        place_pose_base.orientation.z = 0.0;
        place_pose_base.orientation.w = 0.0;

        
        geometry_msgs::msg::PoseStamped place_in_base, place_in_world;

        place_in_base.header.frame_id = "panda_link0";
        place_in_base.header.stamp = tf2_ros::toMsg(tf2::TimePointZero);  
        place_in_base.pose = place_pose_base;

        try 
        {
            place_in_world = tf_buffer_->transform(place_in_base, "world", tf2::durationFromSec(1.0));
        } 
        catch (const tf2::TransformException &ex) 
        {
            RCLCPP_ERROR(this->get_logger(),
                        "Falha ao transformar pose de descarte para 'world': %s", ex.what());
            return;
        }

        RCLCPP_INFO(this->get_logger(),
                    "Pose de Descarte (world): x=%.3f, y=%.3f, z=%.3f",
                    place_in_world.pose.position.x,
                    place_in_world.pose.position.y,
                    place_in_world.pose.position.z);

        // --- Mover usando a pose EM WORLD ---
        if (positions_for_arm(place_in_world.pose, 0.5, false)) 
        {
            rclcpp::sleep_for(std::chrono::milliseconds(1000));
            open_gripper();
            move_group_arm->detachObject(received_id);
            rclcpp::sleep_for(std::chrono::milliseconds(500));
        }
        else 
        {
            RCLCPP_ERROR(this->get_logger(), "Falha ao mover para a pose de descarte.");
        }

       
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
        if(request->reached == true)
        {
            open_gripper();
            calculate_global_pose(std::get<0>(object), std::get<1>(object));

            send_request(false);
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
       
    }


    void send_request(bool stop_flag)
    {
        auto request = std::make_shared<object_manipulation_interfaces::srv::ObjectCollision::Request>();
        request->stop = stop_flag;

      
        client_->async_send_request(request,
            [this](rclcpp::Client<object_manipulation_interfaces::srv::ObjectCollision>::SharedFuture future_response) 
            {
                auto response = future_response.get();  

                if (response->success) 
                {
                    RCLCPP_INFO(this->get_logger(), "Service executado com sucesso!");
                } 
                else 
                {
                    RCLCPP_WARN(this->get_logger(), "Falha ao executar service");
                }
            }
        );
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
      
    
        client_ = this->create_client<object_manipulation_interfaces::srv::ObjectCollision>(
            "/object_collision");

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