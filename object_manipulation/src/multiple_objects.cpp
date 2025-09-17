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


class MultipleObjects : public rclcpp::Node {

private:

    //Publishers.
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;

    //Subscriptions.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_1;


    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr parameterTimer;    


    std::vector<std::tuple<float, float, float>> points;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;
    rclcpp::TimerBase::SharedPtr init_timer_;

    vision_msgs::msg::Detection3DArray object_detections;


  
    inline float round_to_multiple(float value, float multiple, int decimals) 
    {
        if (multiple == 0.0) return value; 
        
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0, decimals);
        result = std::round(result * factor) / factor;
        
        return result;
    }
    

    int count_decimals(float number) 
    {
      
        float fractional = std::fabs(number - std::floor(number));
        int decimals = 0;
        const float epsilon = 1e-9; 
    
  
        while (fractional > epsilon && decimals < 20) {
            fractional *= 10;
            fractional -= std::floor(fractional);
            decimals++;
        }
        return decimals;
    }


    void initMoveGroup() {
        try {
            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "arm");  

            move_group_gripper = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "gripper");

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

        move_group_arm->setPlannerId("RRTConnectkConfigDefault");
        move_group_arm->setPoseTarget(target_pose);

        move_group_arm->setMaxVelocityScalingFactor(1.0);
        move_group_arm->setMaxAccelerationScalingFactor(1.0);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
           
            move_group_arm->execute(plan);
        }

        
        if (plan.trajectory.joint_trajectory.points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Trajetória vazia retornada pelo planejador");
            return;
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

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            auto exec_result = move_group_arm->execute(plan);

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

        
        move_group_gripper->setJointValueTarget({
            {"panda_finger_joint1", 0.0185},
            {"panda_finger_joint2", 0.0185}
        });

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_gripper->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            auto exec_result = move_group_gripper->execute(plan);

            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) 
            {
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
                RCLCPP_INFO(this->get_logger(), "Gripper abriu (MoveIt).");
            }
        }
    }



    /*

        PUBLISHERS.

    */



    void send_joint_positions(const geometry_msgs::msg::Pose &object_pose)
    {
        geometry_msgs::msg::Pose pose;
        pose.position.x = object_pose.position.x - 0.4;
        pose.position.y = object_pose.position.y;
        pose.position.z = object_pose.position.z - 1.016 + 0.15;

        tf2::Quaternion q_obj;
        tf2::fromMsg(object_pose.orientation, q_obj);
        tf2::Quaternion q_rot; q_rot.setRPY(M_PI, 0, 0);
        tf2::Quaternion q_final = q_obj * q_rot;
        q_final.normalize();
        q_final.setW(-q_final.w());
        pose.orientation = tf2::toMsg(q_final);

        positions_for_arm(pose);

        geometry_msgs::msg::Pose pose_2;
        pose_2.position.x = object_pose.position.x - 0.4;
        pose_2.position.y = object_pose.position.y;
        pose_2.position.z = object_pose.position.z - 1.016;

        q_obj;
        tf2::fromMsg(object_pose.orientation, q_obj);
        q_rot; q_rot.setRPY(M_PI, 0, 0);
        q_final = q_obj * q_rot;
        q_final.normalize();
        q_final.setW(-q_final.w());
        pose_2.orientation = tf2::toMsg(q_final);

        positions_for_arm(pose_2);

        close_gripper();

        pose_2;

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist_x(-0.5, -0.1); 
        std::uniform_real_distribution<float> dist_y(-0.5, 0.5); 

        pose_2.position.x = dist_x(gen);
        pose_2.position.y = dist_y(gen);
        pose_2.position.z = 0.1; 
                
        tf2::fromMsg(object_pose.orientation, q_obj);

        q_rot.setRPY(M_PI, 0, 0);

        q_final = q_obj * q_rot;
        q_final.normalize();

        q_final.setW(-q_final.w());  

        pose_2.orientation = tf2::toMsg(q_final);

        positions_for_arm(pose_2);

    
        open_gripper();

        return_to_origin();

        // rclcpp::sleep_for(std::chrono::seconds(20));


    }

  
    /*
    
        CALLBACKS.

    */

    // void callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    // {
    //     for (const auto & detection : msg->detections) 
    //     {
        
    //         if (!detection.results.empty() && detection.results[0].hypothesis.class_id == "1")
    //         {
    //             const auto & size = detection.bbox.size;
    //             RCLCPP_INFO(this->get_logger(),
    //             "Class 1 -> size x: %.3f, y: %.3f, z: %.3f",
    //             size.x, size.y, size.z);
    //         }
    //     }
    // }

    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        if (msg->detections.empty()) 
        {
            RCLCPP_WARN(this->get_logger(), "Detection3DArray vazio recebido.");
            return;
        }

        object_detections = *msg;

        RCLCPP_INFO(this->get_logger(), "Recebidas %zu detecções", object_detections.detections.size());

    
        for (size_t i = 0; i < object_detections.detections.size(); ++i) 
        {
            const auto &det = object_detections.detections[i];
            const auto &target_pose = det.bbox.center;

            RCLCPP_INFO(this->get_logger(),
                "Objeto %zu -> x: %.3f, y: %.3f, z: %.3f",
                i, target_pose.position.x, target_pose.position.y, target_pose.position.z);

            send_joint_positions(target_pose);
        }

        
    }



 

public:
    MultipleObjects()
     : Node("multiple_objects")
    {


        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/boxes_detection_array", 10,
            std::bind(&MultipleObjects::detectionCallback, this, std::placeholders::_1));


        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&MultipleObjects::initMoveGroup, this));
        
     
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }   
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("multiple_objects");

    
    rclcpp::spin(std::make_shared<MultipleObjects>());
    rclcpp::shutdown();
    return 0;
}