#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include <shape_msgs/msg/solid_primitive.hpp>
#include "yaml-cpp/yaml.h"

// MoveIt Headers Necessários (Apenas Scene Interface e Mensagens)
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>

// Serviço Customizado
#include "mobile_manipulation_interfaces/srv/mobile_object_collision.hpp"

using namespace std::chrono_literals;

class AddCollision : public rclcpp::Node 
{
public:
    AddCollision()
     : Node("add_collision_objects")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        std::string labels_path = this->get_parameter("yaml_file").as_string();

        load_labels_from_yaml(labels_path);

        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/boxes_detection_array", 10,
            std::bind(&AddCollision::detectionCallback, this, std::placeholders::_1));
        
        service_ = this->create_service<mobile_manipulation_interfaces::srv::MobileObjectCollision>(
            "/object_collision",
            std::bind(&AddCollision::handleStopService, this, std::placeholders::_1, std::placeholders::_2));

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(2), 
            [this]() {
                this->add_ground_plane();
                this->init_timer_->cancel(); 
            });
            
        RCLCPP_INFO(this->get_logger(), "Nó de Colisão Iniciado (MoveGroupInterface removido).");
    }   

private:
    struct LabelRule 
    {
        std::string label;
        bool is_prefix;
    };

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Service<mobile_manipulation_interfaces::srv::MobileObjectCollision>::SharedPtr service_;
    rclcpp::TimerBase::SharedPtr init_timer_;
    
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    std::string stop_moving_obstacle = "";
    std::unordered_set<std::string> added;
    std::unordered_map<std::string, geometry_msgs::msg::Pose> last_known_poses_;
    std::vector<LabelRule> authorized_labels_;
    std::vector<LabelRule> unauthorized_labels_;
    bool activate_movement = true;

    void load_labels_from_yaml(const std::string& file_path)
    {
        std::ifstream f(file_path.c_str());
        if (!f.good()) {
            RCLCPP_WARN(this->get_logger(), "YAML não encontrado ou vazio: %s", file_path.c_str());
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
            RCLCPP_ERROR(this->get_logger(), "Erro parsing YAML: %s", e.what());
        }
    }

    void add_ground_plane()
    {
        moveit_msgs::msg::CollisionObject ground;
        ground.id = "ground_plane";
        ground.header.frame_id = "world";
        
        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions = {20.0, 20.0, 0.01}; 
        
        geometry_msgs::msg::Pose pose;
        pose.orientation.w = 1.0;
        
        ground.primitives.push_back(primitive);
        ground.primitive_poses.push_back(pose);
        ground.operation = ground.ADD;
        
        planning_scene_interface.applyCollisionObjects({ground});
        RCLCPP_INFO(this->get_logger(), "Ground Plane adicionado à cena.");
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

        return (dist > 0.005); 
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
        
        RCLCPP_INFO(this->get_logger(), "Objeto adicionado: %s", id.c_str());
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
            pose.position.z += det.bbox.size.z / 2.0; 

            std::array<double, 3> size_array = {det.bbox.size.x, det.bbox.size.y, det.bbox.size.z};

            if (added.find(object_id) == added.end()) 
            {
                add_collision_box(object_id, size_array, pose);
            } 
            else 
            {
                if (object_id == stop_moving_obstacle)
                {
                    if (activate_movement) move_collision_box(object_id, pose);
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
        RCLCPP_INFO(this->get_logger(), "Serviço Move: ID '%s' -> Ativo: %d", 
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