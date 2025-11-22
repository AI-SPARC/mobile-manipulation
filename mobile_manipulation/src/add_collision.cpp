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
#include <thread> 

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

        // CORREÇÃO 1: Nó e Executor separados para o MoveIt (Evita travar o callback)
        moveit_node_ = std::make_shared<rclcpp::Node>("add_collision_worker");
        executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
        executor_->add_node(moveit_node_);
        executor_thread_ = std::thread([this]() { this->executor_->spin(); });

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

    ~AddCollision()
    {
        executor_->cancel();
        if (executor_thread_.joinable()) executor_thread_.join();
    }

private:

    struct LabelRule 
    {
        std::string label;
        bool is_prefix;
    };

    // MoveIt
    rclcpp::Node::SharedPtr moveit_node_;
    rclcpp::Executor::SharedPtr executor_;
    std::thread executor_thread_;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Service<mobile_manipulation_interfaces::srv::MobileObjectCollision>::SharedPtr service_;
    
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    rclcpp::TimerBase::SharedPtr init_timer_;
    
    std::string id_to_remove = "", move_group, stop_moving_obstacle;
    std::unordered_set<std::string> added;
    
    std::unordered_map<std::string, geometry_msgs::msg::Pose> last_known_poses_;

    std::vector<LabelRule> authorized_labels_;
    std::vector<LabelRule> unauthorized_labels_;

    bool activate_movement = true;


    void load_labels_from_yaml(const std::string& file_path)
    {
        std::ifstream f(file_path.c_str());
        if (!f.good()) {
            RCLCPP_ERROR(this->get_logger(), "YAML não encontrado: %s", file_path.c_str());
            return;
        }
        try {
            YAML::Node config = YAML::LoadFile(file_path);
            auto load_rules = [&](const YAML::Node& node, std::vector<LabelRule>& target) {
                for (const auto& label_node : node) {
                    std::string label = label_node.as<std::string>();
                    bool is_prefix = (!label.empty() && label.back() == '_');
                    target.push_back({label, is_prefix});
                }
            };
            if (config["authorized_labels"]) load_rules(config["authorized_labels"], authorized_labels_);
            if (config["unauthorized_labels"]) load_rules(config["unauthorized_labels"], unauthorized_labels_);
        } catch (const YAML::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Erro YAML: %s", e.what());
        }
    }

    void initMoveGroup() 
    {
        try {
            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                moveit_node_, move_group);
            add_ground_plane();
            RCLCPP_INFO(this->get_logger(), "MoveGroup inicializado.");
            init_timer_->cancel();  
        } catch (const std::exception &e) {
            RCLCPP_WARN(this->get_logger(), "Erro init MoveGroup: %s", e.what());
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
        pose.orientation.w = 1.0;
        ground.primitives.push_back(primitive);
        ground.primitive_poses.push_back(pose);
        ground.operation = ground.ADD;
        planning_scene_interface.applyCollisionObjects({ground});
    }

    bool is_significant_change(const std::string& id, const geometry_msgs::msg::Pose& new_pose)
    {
        if (last_known_poses_.find(id) == last_known_poses_.end()) return true;

        const auto& old_pose = last_known_poses_[id];
        
        double dist = std::sqrt(
            std::pow(new_pose.position.x - old_pose.position.x, 2) +
            std::pow(new_pose.position.y - old_pose.position.y, 2) +
            std::pow(new_pose.position.z - old_pose.position.z, 2)
        );

        if (dist > 0.01) return true; 

        return false;
    }

    void add_collision_box(const std::string &id, const std::array<double, 3> &dimensions, const geometry_msgs::msg::Pose &pose)
    {
        if (added.find(id) != added.end()) return;

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
        
        added.insert(id);
        last_known_poses_[id] = pose; 
    }

    void move_collision_box(const std::string &id, const geometry_msgs::msg::Pose &pose)
    {
        if (!is_significant_change(id, pose)) return;

        moveit_msgs::msg::CollisionObject collision_object;
        collision_object.id = id;
        collision_object.header.frame_id = "world";
        collision_object.primitive_poses.push_back(pose);
        collision_object.operation = collision_object.MOVE;

        planning_scene_interface.applyCollisionObjects({collision_object});
        
        last_known_poses_[id] = pose; 
    }

    bool is_authorized(const std::string& label)
    {
        for (const auto& rule : unauthorized_labels_) 
            if ((rule.is_prefix && label.rfind(rule.label, 0) == 0) || (!rule.is_prefix && label == rule.label)) 
                return false; 

        if (authorized_labels_.empty()) return true;

        for (const auto& rule : authorized_labels_) 
            if ((rule.is_prefix && label.rfind(rule.label, 0) == 0) || (!rule.is_prefix && label == rule.label)) 
                return true;

        return false;
    }

    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        if (msg->detections.empty()) return;

        for (const auto &det : msg->detections)
        {
            if (det.results.empty()) continue;

            std::string object_id = det.results[0].hypothesis.class_id;
            if (!is_authorized(object_id)) continue;

            geometry_msgs::msg::Pose pose = det.bbox.center;
            pose.position.z += det.bbox.size.z / 2;

            std::array<double, 3> size_array = {det.bbox.size.x, det.bbox.size.y, det.bbox.size.z};

            // Lógica de Atualização
            if (added.find(object_id) == added.end()) 
            {
                add_collision_box(object_id, size_array, pose);
            } 
            else 
            {
                if (object_id == stop_moving_obstacle)
                {
                    if (activate_movement) 
                    {
                        move_collision_box(object_id, pose);
                    }
                }
                else 
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
        RCLCPP_INFO(this->get_logger(), "Controle de movimento: ID '%s' -> Ativo: %d", 
            stop_moving_obstacle.c_str(), activate_movement);
        response->success = true;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AddCollision>());
    rclcpp::shutdown();
    return 0;
}