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
#include "object_manipulation_interfaces/srv/object_collision.hpp"

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

    //Services.

    rclcpp::Client<object_manipulation_interfaces::srv::ObjectCollision>::SharedPtr client_;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;

    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::TimerBase::SharedPtr threads;



    vision_msgs::msg::Detection3DArray object_detections;


    void add_collision_box(const std::string &id,
                       const std::array<double, 3> &size,
                       const geometry_msgs::msg::Pose &pose,
                       const std::string &frame_id = "world")
    {
        static moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

        std::vector<std::string> known_objects = planning_scene_interface.getKnownObjectNames();
        if (std::find(known_objects.begin(), known_objects.end(), id) != known_objects.end()) {
            return;
        }

        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.id = id;
        collision_object.header.frame_id = frame_id;  // usa frame original

        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions = {size[0], size[1], size[2]};

        collision_object.primitives.push_back(primitive);
        collision_object.primitive_poses.push_back(pose);
        collision_object.operation = collision_object.ADD;

        planning_scene_interface.applyCollisionObjects({collision_object});
    }


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





    /*

        PUBLISHERS.

    */



    void send_joint_positions(const std::string &id)
    {
        static moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

        auto known_objects = planning_scene_interface.getKnownObjectNames();
        

        auto object_poses = planning_scene_interface.getObjectPoses({id});
        auto it = object_poses.find(id);
        if (it == object_poses.end()) {
            RCLCPP_WARN(this->get_logger(), "Objeto '%s' não retornou pose", id.c_str());
            return;
        }

        geometry_msgs::msg::Pose object_pose = it->second;

        std::string frame = "world";  


        RCLCPP_INFO(rclcpp::get_logger("get_object_pose"),
                    "Objeto '%s' posição: x=%f, y=%f, z=%f",
                    id.c_str(),
                    object_pose.position.x,
                    object_pose.position.y,
                    object_pose.position.z);


        tf2::Quaternion q_down_orientation;

        q_down_orientation.setRPY(M_PI, 0, 0);
        geometry_msgs::msg::Quaternion orientation_msg = tf2::toMsg(q_down_orientation);

        geometry_msgs::msg::Pose pose;
        pose.position.x = object_pose.position.x;
        pose.position.y = object_pose.position.y;
        pose.position.z = object_pose.position.z + 0.15;
        pose.orientation = orientation_msg; 
        positions_for_arm(pose);

   
        

        geometry_msgs::msg::Pose pose_2;
        pose_2.position.x = object_pose.position.x;
        pose_2.position.y = object_pose.position.y;
        pose_2.position.z = object_pose.position.z;
        pose_2.orientation = orientation_msg;
        positions_for_arm(pose_2);

    
        remove_collision_box(id);

        rclcpp::sleep_for(std::chrono::milliseconds(200));

        close_gripper();
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist_y(-0.4, 0.4); 

        double y_random = dist_y(gen);
        y_random = std::round(y_random * 10.0) / 10.0;

        pose_2.position.x = -0.4;
        pose_2.position.y = y_random;
        pose_2.position.z = 0.15; 
        pose_2.orientation = orientation_msg; 

        positions_for_arm(pose_2);


        open_gripper();

        rclcpp::sleep_for(std::chrono::milliseconds(500));
        
        auto request = std::make_shared<object_manipulation_interfaces::srv::ObjectCollision::Request>();
        request->object_id = id;

            // Checa se o service está disponível
        if (!client_->wait_for_service(1s)) {
            RCLCPP_WARN(this->get_logger(), "Service object_collision não disponível");
            return;
        }

        // Chamada assíncrona com callback
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

        
        // return_to_origin(); 
    }






    /*
    
        CALLBACKS.

    */


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
            geometry_msgs::msg::Pose pose = det.bbox.center;
            pose.position.x -= 0.4;
            pose.position.z -= 1.016;

          
            send_joint_positions(std::to_string(i));
        }
    }
   

public:
    MultipleObjects()
     : Node("multiple_objects")
    {

        // Topics.
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/boxes_detection_array", 10,
            std::bind(&MultipleObjects::detectionCallback, this, std::placeholders::_1));

        // Services.


        client_ = this->create_client<object_manipulation_interfaces::srv::ObjectCollision>("object_collision");

        // Timers.

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&MultipleObjects::initMoveGroup, this));

    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MultipleObjects>());
  rclcpp::shutdown();
  return 0;
}