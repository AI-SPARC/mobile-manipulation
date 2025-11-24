#include <memory>
#include <vector>
#include <string>
#include <unordered_set>
#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"

#include <yaml-cpp/yaml.h>

#include "mobile_manipulation_interfaces/srv/stop_pose.hpp"

#include "mobile_manipulation_interfaces/action/pick_object.hpp"
#include "mobile_manipulation_interfaces/action/path.hpp"
#include "mobile_manipulation_interfaces/action/controller.hpp"
#include <nav_msgs/msg/odometry.hpp>                       

class ServerNode : public rclcpp::Node 
{
public:
    ServerNode()
     : Node("pick_and_organize")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("label_to_storage_yaml_file", "");
        this->declare_parameter<std::string>("storage_poses_yaml_file", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        label_to_storage_yaml_file = this->get_parameter("label_to_storage_yaml_file").as_string();
        storage_poses_yaml_file = this->get_parameter("storage_poses_yaml_file").as_string();

        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&ServerNode::detection_callback, this, std::placeholders::_1));

        client_ = this->create_client<mobile_manipulation_interfaces::srv::StopPose>(
            "stop_pose");
        
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(
            this, "pick_object");

        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(
            this, "path");

        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(
            this, "controller");

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&ServerNode::odom_callback, this, std::placeholders::_1));

        if(!yaml_file.empty()) 
        {
            loadLocationsFromYaml(yaml_file);
        }

        if(!label_to_storage_yaml_file.empty())
        {
            loadLabelToStorage(label_to_storage_yaml_file);
        }

        if(!storage_poses_yaml_file.empty())
        {
            loadStoragePoses(storage_poses_yaml_file);
        }

        
    } 

private:
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // Service client.
    rclcpp::Client<mobile_manipulation_interfaces::srv::StopPose>::SharedPtr client_;

    // Action clients.
    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;

    std::string yaml_file, label_to_storage_yaml_file, storage_poses_yaml_file;
    std::unordered_set<std::string> authorized_labels;
    std::unordered_set<std::string> picked;
    std::unordered_map<std::string, std::vector<geometry_msgs::msg::Pose>> storage;
    std::unordered_map<std::string, std::vector<std::string>> labels_to_storage;

    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;
    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0;

    bool storing = false;
    bool action_busy = false;

    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try {
            YAML::Node config = YAML::LoadFile(yaml_path);
            for (const auto &label_node : config) {
                authorized_labels.insert(label_node.first.as<std::string>());
            }
        } catch (const YAML::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load YAML: %s", e.what());
        }
    }

    void loadLabelToStorage(const std::string &yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);

        for (auto it = config.begin(); it != config.end(); ++it)
        {
            std::string group_name = it->first.as<std::string>();  
            const YAML::Node &entries = it->second;                

            std::vector<std::string> storages;

            for (const auto &entry : entries)
            {
                const YAML::Node &value = entry["storage"];

                if (value)
                {
                    storages.push_back(value.as<std::string>());
                }
            }

            labels_to_storage[group_name] = storages;
        }
    }

    void loadStoragePoses(const std::string &yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);

        for (auto it = config.begin(); it != config.end(); ++it)
        {
            std::string storage_name = it->first.as<std::string>();
            const YAML::Node &locations = it->second;

            std::vector<geometry_msgs::msg::Pose> poses;

            for (const auto &loc : locations)
            {
                for (auto loc_it = loc.begin(); loc_it != loc.end(); ++loc_it)
                {
                    const YAML::Node &loc_data = loc_it->second;

                    geometry_msgs::msg::Pose pose;

                    const YAML::Node &pos = loc_data["position"];
                    pose.position.x = pos[0].as<double>();
                    pose.position.y = pos[1].as<double>();
                    pose.position.z = pos[2].as<double>();

                    if (loc_data["orientation"])
                    {
                        const YAML::Node &ori = loc_data["orientation"];

                        pose.orientation.x = ori[0].as<double>();
                        pose.orientation.y = ori[1].as<double>();
                        pose.orientation.z = ori[2].as<double>();
                        pose.orientation.w = ori[3].as<double>();
                    }
                    else
                    {
                        pose.orientation.x = 0.0;
                        pose.orientation.y = 0.0;
                        pose.orientation.z = 0.0;
                        pose.orientation.w = 1.0;
                    }

                    poses.push_back(pose);
                }
            }

            storage[storage_name] = poses;
        }
    }

    std::pair<std::string, geometry_msgs::msg::Pose> getClosestStorage(const std::string& label, double px, double py, double pz)
    {
        double best_dist = std::numeric_limits<double>::max();
        std::string best_storage_name;
        geometry_msgs::msg::Pose best_pose;

        if (!labels_to_storage.count(label))
        {
            throw std::runtime_error("Label não encontrada: " + label);
        }

        const auto& storage_list = labels_to_storage[label];

        for (const auto& storage_name : storage_list)
        {
            if (!storage.count(storage_name))
                continue;

            const auto& poses = storage.at(storage_name);

            for (const auto& pose : poses)
            {
                double dx = pose.position.x - px;
                double dy = pose.position.y - py;
                double dz = pose.position.z - pz;

                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_storage_name = storage_name;
                    best_pose = pose;
                }
            }
        }

        if (best_storage_name.empty())
        {
            throw std::runtime_error("Nenhum storage encontrado para a label: " + label);
        }

        return { best_storage_name, best_pose };
    }

    // Callbacks.

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) 
    {
        pose_x = msg->pose.pose.position.x;
        pose_y = msg->pose.pose.position.y;
        pose_z = 0.0;
    }

    void detection_callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        if(action_busy == true) 
        {
            return;
        }

        if(storing == true)
        {
            std::string id;
                
            size_t pos = std::get<0>(pick_pose).find('_'); 

            if (pos != std::string::npos) 
            {
                id = std::get<0>(pick_pose).substr(0, pos);  
            } 
            else 
            {
                id = std::get<0>(pick_pose);  
            }

            auto [name, pose] = getClosestStorage(id, pose_x, pose_y, pose_z);

            std::get<1>(pick_pose) = pose;
            send_path_goal(pose);
            action_busy = true;
        }
        else
        {
            for (const auto &det : msg->detections)
            {
                if (det.results.empty()) continue;

                std::string raw_id = det.results[0].hypothesis.class_id;
                std::string id;
                
                size_t pos = raw_id.find('_'); 

                if (pos != std::string::npos) 
                {
                    id = raw_id.substr(0, pos);  
                } 
                else 
                {
                    id = raw_id;  
                }

                if (authorized_labels.find(id) == authorized_labels.end()) {
                    continue;
                }
                
                if(picked.find(raw_id) == picked.end())
                {
                    action_busy = true;
                    picked.insert(raw_id);

                    geometry_msgs::msg::Pose pose;
                    pose.position = det.bbox.center.position;
                    pose.orientation = det.bbox.center.orientation;

                    pick_pose = std::make_pair(raw_id, pose);
                    send_path_goal(pose);
                    action_busy = true;

                    break;
                }
            }
        }
        
    }

    // Service client (stop_pose).

    void send_request(geometry_msgs::msg::Pose pose)
    {
        auto request = std::make_shared<mobile_manipulation_interfaces::srv::StopPose::Request>();

        request->stop_pose = pose;
      
        client_->async_send_request(request,
            [this](rclcpp::Client<mobile_manipulation_interfaces::srv::StopPose>::SharedFuture future_response) 
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

    // Action client (path).

    void send_path_goal(const geometry_msgs::msg::Pose & target_pose)
    {
        if (!this->path_client->wait_for_action_server(std::chrono::seconds(5))) 
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            action_busy = false;
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::Path::Goal();
        
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "Enviando Goal (Pose) para A*...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SendGoalOptions();
        
        send_goal_options.goal_response_callback = std::bind(&ServerNode::path_goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::path_result_callback, this, std::placeholders::_1);
        send_goal_options.feedback_callback = std::bind(&ServerNode::path_feedback_callback, this, std::placeholders::_1, std::placeholders::_2);

        this->path_client->async_send_goal(goal_msg, send_goal_options);
    }

    void path_feedback_callback(
        rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::SharedPtr,
        const std::shared_ptr<const mobile_manipulation_interfaces::action::Path::Feedback> feedback)
    {
        if (feedback->recalculating_path == true) 
        {
            send_request(feedback->stop_pose);
        }

    }

    void path_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>> & goal_handle)
    {
        if (!goal_handle) 
        {
            action_busy = false;
            RCLCPP_ERROR(this->get_logger(), "Goal foi rejeitado pelo servidor");
        } 
        else 
        {
            RCLCPP_INFO(this->get_logger(), "Goal aceito, executando...");
        }
    }

    void path_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::WrappedResult & result)
    {
        switch (result.code) 
        {
            case rclcpp_action::ResultCode::SUCCEEDED:
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
                return;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
                return;
            default:
                RCLCPP_ERROR(this->get_logger(), "Unknown result code");
                return;
        }
    
        send_controller_goal(result.result->path);
    }

    // Action client (controller).

    void send_controller_goal(const nav_msgs::msg::Path &target_path)
    {
        if (!this->controller_client->wait_for_action_server(std::chrono::seconds(5))) 
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::Controller::Goal();
        
        goal_msg.path = target_path;

        RCLCPP_INFO(this->get_logger(), "Enviando Goal (Pose) para CONTROLLER...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SendGoalOptions();
        
        send_goal_options.goal_response_callback = std::bind(&ServerNode::controller_goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::controller_result_callback, this, std::placeholders::_1);

        this->controller_client->async_send_goal(goal_msg, send_goal_options);
    }

    void controller_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>> & goal_handle)
    {
        if (!goal_handle) 
        {
            RCLCPP_ERROR(this->get_logger(), "Goal foi rejeitado pelo servidor");
        } 
        else 
        {
            RCLCPP_INFO(this->get_logger(), "Goal aceito, executando...");
        }
    }

    void controller_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::WrappedResult & result)
    {
        switch (result.code) 
        {
            case rclcpp_action::ResultCode::SUCCEEDED:
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
                return;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
                return;
            default:
                RCLCPP_ERROR(this->get_logger(), "Unknown result code");
                return;
        }
        
        send_goal(std::get<0>(pick_pose), std::get<1>(pick_pose));
    }

    // Action client (pick_object).

    void send_goal(const std::string id, const geometry_msgs::msg::Pose & target_pose)
    {
        if (!this->client_ptr_->wait_for_action_server(std::chrono::seconds(5))) 
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
           
            return;
        }

      

        auto goal_msg = mobile_manipulation_interfaces::action::PickObject::Goal();
        
        goal_msg.obstacle_id = id;
        goal_msg.pick = storing;
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "Enviando Goal (Pose) para MANIPULATION...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SendGoalOptions();
        
        send_goal_options.goal_response_callback = std::bind(&ServerNode::goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::result_callback, this, std::placeholders::_1);

        this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
    }

    void goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>> & goal_handle)
    {
        if (!goal_handle) 
        {
            RCLCPP_ERROR(this->get_logger(), "Goal foi rejeitado pelo servidor");
            action_busy = false;
        } 
        else 
        {
            RCLCPP_INFO(this->get_logger(), "Goal aceito, executando...");
        }
    }

    void result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>::WrappedResult & result)
    {
        action_busy = false;

        switch (result.code) 
        {
            case rclcpp_action::ResultCode::SUCCEEDED:
                break;
            case rclcpp_action::ResultCode::ABORTED:
                RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
                return;
            case rclcpp_action::ResultCode::CANCELED:
                RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
                return;
            default:
                RCLCPP_ERROR(this->get_logger(), "Unknown result code");
                return;
        }

        if(result.result->success == false)
        {
            picked.erase(std::get<0>(pick_pose));
        }
        else
        {
            if(storing == false)
            {
                storing = true;
            }
            else
            {
                storing = false;
            }
        }

        RCLCPP_INFO(this->get_logger(), "O resultado foi: %s", result.result->success ? "true" : "false");
                
        
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ServerNode>());
  rclcpp::shutdown();
  return 0;
}