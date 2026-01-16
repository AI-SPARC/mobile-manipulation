#include "manipulation/AddCollision.hpp"

#include <cmath>
#include <iostream>
#include <functional>
#include <fstream>
#include <sstream>

#include "yaml-cpp/yaml.h"

using namespace std::chrono_literals;

namespace manipulation {

AddCollision::AddCollision(const rclcpp::NodeOptions & options)
 : Node("add_collision_objects", options), db_(nullptr), stop_moving_obstacle(""), activate_movement(false)
{
    this->declare_parameter<std::string>("yaml_file", "");
    this->declare_parameter<std::string>("database_path", "/home/momesso/pibic/src/mobile_manipulation_packages/llms/db/robot_world_data.db");

    std::string labels_path = this->get_parameter("yaml_file").as_string();
    db_path_ = this->get_parameter("database_path").as_string();

    load_labels_from_yaml(labels_path);
    connect_database();

    service_ = this->create_service<mobile_manipulation_interfaces::srv::MobileObjectCollision>(
        "/object_collision",
        std::bind(&AddCollision::handleStopService, this, std::placeholders::_1, std::placeholders::_2));

    init_timer_ = this->create_wall_timer(
        std::chrono::seconds(2), 
        [this]() {
            this->add_ground_plane();
            this->init_timer_->cancel(); 
        });


    db_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&AddCollision::sync_from_database, this));
        
    RCLCPP_INFO(this->get_logger(), "AddCollision iniciado. Lendo DB: %s", db_path_.c_str());
}   

AddCollision::~AddCollision()
{
    if (db_) {
        sqlite3_close(db_);
    }
}

void AddCollision::connect_database()
{
    
    int rc = sqlite3_open_v2(db_path_.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr);
    if (rc != SQLITE_OK) 
    {
        RCLCPP_ERROR(this->get_logger(), "Falha ao abrir DB (pode não ter sido criado ainda): %s", sqlite3_errmsg(db_));
        db_ = nullptr;
    } 
    else 
    {
        RCLCPP_INFO(this->get_logger(), "Conectado ao DB com sucesso.");
    }
}

std::vector<double> AddCollision::parse_string_to_vector(const std::string& s)
{
    std::vector<double> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ';')) {
        try {
            v.push_back(std::stod(item));
        } catch (...) {
            v.push_back(0.0);
        }
    }
    
    while (v.size() < 3) v.push_back(0.0);
    return v;
}

void AddCollision::sync_from_database()
{
    if (!db_) 
    {
        
        static int retry_counter = 0;
        if (retry_counter++ % 50 == 0) connect_database(); 
        return;
    }

    const char* sql = "SELECT id, pose, size FROM objects;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_, sql, -1, &stmt, 0) != SQLITE_OK) 
    {
        
        return; 
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string object_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string pose_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        std::string size_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

        
        if (!is_authorized(object_id)) continue;

        
        std::vector<double> pose_vec = parse_string_to_vector(pose_str);
        std::vector<double> size_vec = parse_string_to_vector(size_str);

        geometry_msgs::msg::Pose pose;
        pose.position.x = pose_vec[0];
        pose.position.y = pose_vec[1];
        pose.position.z = pose_vec[2];
        pose.orientation.w = 1.0; 
   
        pose.position.z += size_vec[2] / 2.0;

        std::array<double, 3> size_array = {size_vec[0], size_vec[1], size_vec[2]};

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

    sqlite3_finalize(stmt);
}

void AddCollision::load_labels_from_yaml(const std::string& file_path)
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

void AddCollision::add_ground_plane()
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

bool AddCollision::is_significant_change(const std::string& id, const geometry_msgs::msg::Pose& new_pose)
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

void AddCollision::add_collision_box(const std::string &id, const std::array<double, 3> &dimensions, const geometry_msgs::msg::Pose &pose)
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

void AddCollision::move_collision_box(const std::string &id, const geometry_msgs::msg::Pose &pose)
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

bool AddCollision::is_authorized(const std::string& label)
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

void AddCollision::handleStopService(
    const std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Request> request,
    std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Response> response)
{
    stop_moving_obstacle = request->obstacle_id;
    activate_movement = request->activate_movement;
    RCLCPP_INFO(this->get_logger(), "Serviço Move: ID '%s' -> Ativo: %d", 
        stop_moving_obstacle.c_str(), activate_movement);
    response->success = true;
}

} // namespace manipulation