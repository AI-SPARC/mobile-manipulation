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


class AddCollision : public rclcpp::Node {

private:

    // Publishers.
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;

    // Subscriptions.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_1;

    // Services.

    rclcpp::Service<object_manipulation_interfaces::srv::ObjectCollision>::SharedPtr service_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr parameterTimer;    


    std::vector<std::tuple<float, float, float>> points;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;
    rclcpp::TimerBase::SharedPtr init_timer_;

    vision_msgs::msg::Detection3DArray object_detections;

    std::string id_to_remove = "";



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
        pose.position.z = 0.0;  
        pose.orientation.w = 1.0;

        ground.primitives.push_back(primitive);
        ground.primitive_poses.push_back(pose);
        ground.operation = ground.ADD;

        planning_scene_interface.applyCollisionObjects({ground});
    }



    void add_collision_box(const std::string &id,const std::array<double, 3> &size, const geometry_msgs::msg::Pose &pose)
    {
        static moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

       
        std::vector<std::string> known_objects = planning_scene_interface.getKnownObjectNames();
        if (std::find(known_objects.begin(), known_objects.end(), id) != known_objects.end()) 
        {
            return;
        }

        
        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.id = id;
        collision_object.header.frame_id = "world";

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
        if (std::find(known_objects.begin(), known_objects.end(), id) == known_objects.end()) 
        {
            // RCLCPP_WARN(rclcpp::get_logger("remove_collision_box"), 
            //             "Objeto %s não encontrado no planning scene.", id.c_str());
            return;
        }

        planning_scene_interface.removeCollisionObjects({id});

        // RCLCPP_INFO(rclcpp::get_logger("remove_collision_box"), 
        //             "Objeto %s removido do planning scene.", id.c_str());
    }


    // CALLBACKS.

    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        if (msg->detections.empty()) 
        {
            RCLCPP_WARN(this->get_logger(), "Detection3DArray vazio recebido.");
            return;
        }

        object_detections = *msg;

        // RCLCPP_INFO(this->get_logger(), "Recebidas %zu detecções", object_detections.detections.size());
        add_ground_plane();
    
        for (size_t i = 0; i < object_detections.detections.size(); ++i) 
        {
            const auto &det = object_detections.detections[i];

            // RCLCPP_INFO(this->get_logger(),
            //     "Objeto %zu -> x: %.3f, y: %.3f, z: %.3f",
            //     i, target_pose.position.x, target_pose.position.y, target_pose.position.z);
            
            geometry_msgs::msg::Pose pose = det.bbox.center;
            pose.position.x -= 0.4;
            pose.position.z -= 1.016;
            
            
            remove_collision_box(std::to_string(i));
            
            if(id_to_remove != std::to_string(i))
            {
                add_collision_box(std::to_string(i), {0.06, 0.06, 0.06}, pose);
            }
        }
        
    }

    void handle_service(
        const std::shared_ptr<object_manipulation_interfaces::srv::ObjectCollision::Request> request,
        std::shared_ptr<object_manipulation_interfaces::srv::ObjectCollision::Response> response)
    {
        RCLCPP_INFO(this->get_logger(),
                    "Recebido pedido para objeto '%s' | remove=%s",
                    request->object_id.c_str(),
                    request->remove.c_str());

        if (request->remove == "true") 
        {
            id_to_remove = request->object_id;

            remove_collision_box(id_to_remove);

            RCLCPP_INFO(this->get_logger(), "Removendo objeto '%s'", request->object_id.c_str());
        } 
        else 
        {
            id_to_remove.clear(); 
            RCLCPP_INFO(this->get_logger(), "Adicionando objeto '%s'", request->object_id.c_str());
        }

        response->success = true;
    }

 

public:
    AddCollision()
     : Node("add_colision_objects")
    {


        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/boxes_detection_array", 10,
            std::bind(&AddCollision::detectionCallback, this, std::placeholders::_1));

        service_ = this->create_service<object_manipulation_interfaces::srv::ObjectCollision>(
            "object_collision",
            std::bind(&AddCollision::handle_service, this,
                    std::placeholders::_1, std::placeholders::_2)
        );

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&AddCollision::initMoveGroup, this));
        
        add_ground_plane();
     
    }   
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    
    rclcpp::spin(std::make_shared<AddCollision>());
    rclcpp::shutdown();
    return 0;
}