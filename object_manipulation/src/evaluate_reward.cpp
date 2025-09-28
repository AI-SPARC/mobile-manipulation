#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>

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
#include "object_manipulation_interfaces/srv/evaluate_reward.hpp"

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


class EvaluateReward : public rclcpp::Node {

private:

    //Publishers.
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;

    //Services.
    rclcpp::Service<object_manipulation_interfaces::srv::EvaluateReward>::SharedPtr service_;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;

    rclcpp::TimerBase::SharedPtr init_timer_;



    void remove_collision_box(const std::string &id)
    {
        static moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

        std::vector<std::string> known_objects = planning_scene_interface.getKnownObjectNames();
        // if (std::find(known_objects.begin(), known_objects.end(), id) == known_objects.end()) 
        // {
        //     RCLCPP_WARN(rclcpp::get_logger("remove_collision_box"), 
        //                 "Objeto %s não encontrado no planning scene.", id.c_str());
        //     return;
        // }

        planning_scene_interface.removeCollisionObjects({id});

        RCLCPP_INFO(rclcpp::get_logger("remove_collision_box"), 
                    "Objeto %s removido do planning scene.", id.c_str());
    }

    void initMoveGroup() {
        try {
            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "arm");  

            move_group_gripper = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "gripper");

            // geometry_msgs::msg::Pose pose;
            // pose.position.x -= 0.4;
            // pose.position.z -= 1.016;
            // pose.orientation.x = 0.0;
            // pose.orientation.y = 0.0;
            // pose.orientation.z = 0.0;
            // pose.orientation.w = 1.0;
            

            // add_collision_box(std::to_string(i), {0.12, 0.12, 0.12}, pose);

            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface inicializado com sucesso.");

            init_timer_->cancel();  
        } catch (const std::exception &e) 
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }

    }


    
    float positions_for_arm(const geometry_msgs::msg::Pose &target_pose) 
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return 0;
        }

        move_group_arm->setWorkspace(
            -2.0, -2.0, 0.01,   
            2.0,  2.0, 2.0     
        );
        move_group_arm->setStartStateToCurrentState();
        move_group_arm->setPlannerId("RRTConnectkConfigDefault");
        move_group_arm->setPoseTarget(target_pose);

        move_group_arm->setPlanningTime(2.0);
        move_group_arm->setNumPlanningAttempts(40);

        move_group_arm->setMaxVelocityScalingFactor(1.0);
        move_group_arm->setMaxAccelerationScalingFactor(1.0);

        move_group_arm->setGoalTolerance(0.002);
        move_group_arm->setGoalJointTolerance(0.002);
        move_group_arm->setGoalPositionTolerance(0.002);
        move_group_arm->setGoalOrientationTolerance(0.002);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_arm->execute(plan);
            rclcpp::sleep_for(std::chrono::milliseconds(50));
        }

        if (plan.trajectory.joint_trajectory.points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Trajetória vazia retornada pelo planejador");
            return -100;
        }

        return 0;
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
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do gripper não inicializado.");
            return;
        }

        move_group_gripper->setStartStateToCurrentState();
        // rclcpp::sleep_for(std::chrono::milliseconds(300));
        
        move_group_gripper->setJointValueTarget({
            {"panda_finger_joint1", 0.01},
            {"panda_finger_joint2", 0.01}
        });

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_gripper->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            auto exec_result = move_group_gripper->execute(plan);
            
            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) 
            {
                rclcpp::sleep_for(std::chrono::milliseconds(50));
                RCLCPP_INFO(this->get_logger(), "Gripper fechou (MoveIt).");
            }
        }
    }

    void open_gripper() 
    {
        if (!move_group_gripper) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do gripper não inicializado.");
            return;
        }

        move_group_gripper->setStartStateToCurrentState();
        // rclcpp::sleep_for(std::chrono::milliseconds(300));
        move_group_gripper->setJointValueTarget({
            {"panda_finger_joint1", 0.037},
            {"panda_finger_joint2", 0.037}
        });

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_gripper->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            auto exec_result = move_group_gripper->execute(plan);

            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) 
            {
                rclcpp::sleep_for(std::chrono::milliseconds(50));
                RCLCPP_INFO(this->get_logger(), "Gripper abriu (MoveIt).");
            }
        }
    }



     void add_ground_plane()
    {
        static moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

        moveit_msgs::msg::CollisionObject ground;
        ground.id = "ground_plane";
        ground.header.frame_id = "world";

        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions = {10.0, 10.0, 0.01}; 

        geometry_msgs::msg::Pose pose;
        pose.position.x = 0.0;
        pose.position.y = 0.0;
        pose.position.z = 0.01;  
        pose.orientation.w = 1.0;

        ground.primitives.push_back(primitive);
        ground.primitive_poses.push_back(pose);
        ground.operation = ground.ADD;

        planning_scene_interface.applyCollisionObjects({ground});
    }

    /*

        PUBLISHERS.

    */

    float send_joint_positions(float x, float y, float z, float qx, float qy, float qz, float qw)
    {
      
        float reward = 0;

        geometry_msgs::msg::Pose pose;
        pose.position.x = x;
        pose.position.y = y;
        pose.position.z = z + 0.15;
        pose.orientation.x = qx; 
        pose.orientation.y = qy;
        pose.orientation.z = qz;
        pose.orientation.w = qw;


        reward = positions_for_arm(pose);

        if(reward == -100)
        {
            return -10000;
        }

        geometry_msgs::msg::Pose pose_1;
        pose_1.position.x = x;
        pose_1.position.y = y;
        pose_1.position.z = z;
        pose_1.orientation.x = qx; 
        pose_1.orientation.y = qy;
        pose_1.orientation.z = qz;
        pose_1.orientation.w = qw;


        reward = positions_for_arm(pose_1);

        if(reward == -100)
        {
            return -10000;
        }

        rclcpp::sleep_for(std::chrono::milliseconds(200));

        close_gripper();
        
        geometry_msgs::msg::Pose pose_2;
        pose_2.position.x = x;
        pose_2.position.y = -0.3;
        pose_2.position.z = 0.1;
        pose_2.orientation.x = qx; 
        pose_2.orientation.y = qy;
        pose_2.orientation.z = qz;
        pose_2.orientation.w = qw;


        reward = positions_for_arm(pose_2);

        if(reward == -100)
        {
            return -10000;
        }

        open_gripper();

        static moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

        auto known_objects = planning_scene_interface.getKnownObjectNames();

        const std::string id = "0";
        
        auto object_poses = planning_scene_interface.getObjectPoses({id});
        auto it = object_poses.find(id);
       

        geometry_msgs::msg::Pose object_pose = it->second;

        std::string frame = "world";  

        RCLCPP_INFO(rclcpp::get_logger("get_object_pose"),
                    "Objeto '%s' posição: x=%f, y=%f, z=%f",
                    id.c_str(),
                    object_pose.position.x,
                    object_pose.position.y,
                    object_pose.position.z);

        
        if(object_pose.position.y <= 0.29)
        {
            float dist = position_distance(object_pose, x, y, z);     
            reward = 0;
            reward = abs((object_pose.position.y * -100) / dist);
            return reward;
        }

        float dist = position_distance(object_pose, x, y, z);

        reward = 0;
        reward = 1 / dist;

        std::cout << "toma" << std::endl;
        return reward;
    }

    float position_distance(const geometry_msgs::msg::Pose& p, float x, float y, float z)
    {
        float dx = p.position.x - x;
        float dy = p.position.y - y;
        float dz = p.position.z - z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }


    void handle_service(
        const std::shared_ptr<object_manipulation_interfaces::srv::EvaluateReward::Request> request,
        std::shared_ptr<object_manipulation_interfaces::srv::EvaluateReward::Response> response)
    {
        // request->pose é float[7]
        float x = request->pose[0];
        float y = request->pose[1];
        float z = request->pose[2];
        float qx = request->pose[3];
        float qy = request->pose[4];
        float qz = request->pose[5];
        float qw = request->pose[6];

        std::cout << x + 0.6 << " " << y + 0.3 << " " << z << std::endl;
        float reward = 0.0;

        if(z > 0.0)
        {
            
            reward = send_joint_positions(x + 0.6, y + 0.3, z, qx, qy, qz, qw);   
            
        }
        else
        {
            reward = -30000;
        }
        std::cout << "toma" << std::endl;

        response->reward = reward;
    }

   

public:
    EvaluateReward()
     : Node("multiple_objects")
    {

        service_ = this->create_service<object_manipulation_interfaces::srv::EvaluateReward>(
            "evaluate_reward",
            std::bind(&EvaluateReward::handle_service, this,
                    std::placeholders::_1, std::placeholders::_2)
        );

        // Timers.

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&EvaluateReward::initMoveGroup, this));
        
        // add_ground_plane();
    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EvaluateReward>());
  rclcpp::shutdown();
  return 0;
}