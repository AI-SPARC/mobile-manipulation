#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "yaml-cpp/yaml.h"
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include "mobile_manipulation_interfaces/srv/mobile_object_collision.hpp"

using namespace std::chrono_literals;



class AddCollision : public rclcpp::Node 
{

public:
    AddCollision()
     : Node("add_colision_objects")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("move_group", "panda_arm");

        std::string labels_path = this->get_parameter("yaml_file").as_string();
        move_group = this->get_parameter("move_group").as_string();


        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/boxes_detection_array", 10,
            std::bind(&AddCollision::detectionCallback, this, std::placeholders::_1));

        
        service_ = this->create_service<mobile_manipulation_interfaces::srv::MobileObjectCollision>(
            "/object_collision",
            std::bind(&AddCollision::handleStopService, this, std::placeholders::_1, std::placeholders::_2));

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&AddCollision::initMoveGroup, this));
        
        
        load_labels_from_yaml(labels_path);
    }   
private:

    struct LabelRule 
    {
        std::string label;
        bool is_prefix;
    };

    // Publishers.
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;

    // Subscriptions.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;

    // Services.
    rclcpp::Service<mobile_manipulation_interfaces::srv::MobileObjectCollision>::SharedPtr service_;
    
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr parameterTimer;    

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;
    rclcpp::TimerBase::SharedPtr init_timer_;
    
    vision_msgs::msg::Detection3DArray object_detections;

    std::string id_to_remove = "", move_group, stop_moving_obstacle;
    std::unordered_set<std::string> added;

    std::vector<LabelRule> authorized_labels_;
    std::vector<LabelRule> unauthorized_labels_;

    bool activate_movement = true;

    void load_labels_from_yaml(const std::string& file_path)
    {
        std::ifstream f(file_path.c_str());

        if (!f.good()) 
        {
            RCLCPP_ERROR(this->get_logger(), "Arquivo YAML de labels não encontrado em: %s", file_path.c_str());
            return;
        }

        try 
        {
            YAML::Node config = YAML::LoadFile(file_path);

            auto load_rules = [&](const YAML::Node& node, std::vector<LabelRule>& target) 
            {
                for (const auto& label_node : node) 
                {
                    std::string label = label_node.as<std::string>();
                    bool is_prefix = false;

                    if (!label.empty() && label.back() == '_') {
                        is_prefix = true;
                    }

                    target.push_back({label, is_prefix});
                }
            };

            if (config["authorized_labels"])
             {
                load_rules(config["authorized_labels"], authorized_labels_);
                RCLCPP_INFO(this->get_logger(), "%zu labels autorizados carregados.", authorized_labels_.size());
            }

            if (config["unauthorized_labels"]) 
            {
                load_rules(config["unauthorized_labels"], unauthorized_labels_);
                RCLCPP_INFO(this->get_logger(), "%zu labels não autorizados carregados.", unauthorized_labels_.size());
            }

        } 
        catch (const YAML::Exception& e) 
        {
            RCLCPP_ERROR(this->get_logger(), "Erro ao processar o arquivo YAML: %s", e.what());
        }
    }

    void initMoveGroup() 
    {
        try 
        {
            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), move_group);
            
                add_ground_plane();

            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface inicializado com sucesso.");

            init_timer_->cancel();  
        } 
        catch (const std::exception &e) 
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }
    }

    void add_ground_plane()
    {
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

    void add_collision_box(const std::string &id,const std::array<double, 3> &dimensions, const geometry_msgs::msg::Pose &pose)
    {
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
        primitive.dimensions = {dimensions[0], dimensions[1], dimensions[2]};

        collision_object.primitives.push_back(primitive);
        collision_object.primitive_poses.push_back(pose);
        collision_object.operation = collision_object.ADD;

        planning_scene_interface.applyCollisionObjects({collision_object});
    }

    void move_collision_box(const std::string &id, const geometry_msgs::msg::Pose &pose)
    {
        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.id = id;
        collision_object.header.frame_id = "world";

        collision_object.primitive_poses.push_back(pose);
        collision_object.operation = collision_object.MOVE;

        planning_scene_interface.applyCollisionObjects({collision_object});
    }

    bool is_authorized(const std::string& label)
    {
        for (const auto& rule : unauthorized_labels_) 
        {
            if ((rule.is_prefix && label.rfind(rule.label, 0) == 0) || (!rule.is_prefix && label == rule.label)) 
            {
                return false; 
            }
        }

        if (authorized_labels_.empty()) return true;

        for (const auto& rule : authorized_labels_) 
        {
            if ((rule.is_prefix && label.rfind(rule.label, 0) == 0) || (!rule.is_prefix && label == rule.label)) 
            {
                return true;
            }
        }

        return false;
    }

    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
       
        if (msg->detections.empty()) 
        {
            RCLCPP_WARN(this->get_logger(), "Detection3DArray vazio recebido.");
            return;
        }

        for (const auto &det : msg->detections)
        {
            std::string object_id = det.results[0].hypothesis.class_id;
            if (!is_authorized(object_id)) continue;

            geometry_msgs::msg::Pose pose = det.bbox.center;
            pose.position.z += det.bbox.size.z / 2;

            std::array<double, 3> size_array = {
                det.bbox.size.x,
                det.bbox.size.y,
                det.bbox.size.z
            };

           
            if (added.find(object_id) == added.end()) 
            {
                add_collision_box(object_id, size_array, pose);
                added.insert(object_id);
            } 
            else 
            {
                if(object_id == stop_moving_obstacle && activate_movement == true)
                {
                    move_collision_box(object_id, pose);
                }
                else if(object_id != stop_moving_obstacle)
                {
                    move_collision_box(object_id, pose);
                }
                
            }
        
        }
    }

    void handleStopService(
        const std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Request> request,
        std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Response> response)
    {
        stop_moving_obstacle = request->obstacle_id;
        activate_movement = request->activate_movement;
        RCLCPP_INFO(this->get_logger(), "Serviço recebido: %s", stop_moving_obstacle);

        response->success = true;
    }

};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AddCollision>());
    rclcpp::shutdown();
    return 0;
}
