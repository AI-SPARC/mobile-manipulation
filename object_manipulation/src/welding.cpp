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


class Welding : public rclcpp::Node {

private:

    //Publishers.
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;

    //Subscriptions.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_1;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;

    std::vector<geometry_msgs::msg::Pose> locations;
    std::string yaml_file;

    rclcpp::TimerBase::SharedPtr init_timer_;

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


    
    void positions_for_arm(const geometry_msgs::msg::Pose &target_pose) 
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return;
        }

      
        move_group_arm->setStartStateToCurrentState();
        move_group_arm->setPlannerId("RRTConnect");
        move_group_arm->setPoseTarget(target_pose);

        move_group_arm->setPlanningTime(4.0);
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


    /*
    
        CALLBACKS.

    */
    
    

    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        for (const auto &det : msg->detections)
        {
            if (det.results.empty() || det.results[0].hypothesis.class_id != "firecabinet")
                continue;


            for (size_t i = 0; i < locations.size(); i++)
            {
                tf2::Vector3 local_corner(locations[i].position.x, locations[i].position.y, locations[i].position.z);

                const auto &pose = det.bbox.center;
                tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
                tf2::Matrix3x3 rot(q);
                tf2::Vector3 translation(pose.position.x, pose.position.y, pose.position.z);

                tf2::Vector3 world_corner = rot * local_corner + translation;

                tf2::Vector3 global_offset(-0.13, 0.0, 0.0);
                world_corner += global_offset;

                // Converter tf2::Vector3 → geometry_msgs::msg::Pose
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

            
           
        }
    }


        

public:
    Welding()
     : Node("welding")
    {
        this->declare_parameter<std::string>("yaml_file", "");
   
        yaml_file = this->get_parameter("yaml_file").as_string();
 
       
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&Welding::detectionCallback, this, std::placeholders::_1));
        
        loadLocationsFromYaml(yaml_file);

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&Welding::initMoveGroup, this));

    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Welding>());
  rclcpp::shutdown();
  return 0;
}