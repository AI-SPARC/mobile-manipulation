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

class ServerNode : public rclcpp::Node 
{
public:
    ServerNode()
     : Node("pick_and_organize")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        yaml_file = this->get_parameter("yaml_file").as_string();
        
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&ServerNode::detectionCallback, this, std::placeholders::_1));

        client_ = this->create_client<mobile_manipulation_interfaces::srv::StopPose>(
            "stop_pose");
        
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(
            this, "pick_object");

        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(
            this, "path");

        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(
            this, "controller");

        if(!yaml_file.empty()) 
        {
            loadLocationsFromYaml(yaml_file);
        }
    } 

private:
    bool action_busy = false;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;

    // Service client.
    rclcpp::Client<mobile_manipulation_interfaces::srv::StopPose>::SharedPtr client_;

    // Action clients.
    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;

    std::string yaml_file;
    std::unordered_set<std::string> pick_and_place_poses;
    std::unordered_set<std::string> picked;

    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;

    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try {
            YAML::Node config = YAML::LoadFile(yaml_path);
            for (const auto &label_node : config) {
                pick_and_place_poses.insert(label_node.first.as<std::string>());
            }
        } catch (const YAML::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load YAML: %s", e.what());
        }
    }

    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        if(action_busy == true)
        {
            return;
        }

        for (const auto &det : msg->detections)
        {
            if (det.results.empty()) continue;

            std::string raw_id = det.results[0].hypothesis.class_id;
            std::string id;
            
            size_t pos = raw_id.find('_'); 
            if (pos != std::string::npos) {
                id = raw_id.substr(0, pos);  
            } else {
                id = raw_id;  
            }

            if (pick_and_place_poses.find(id) == pick_and_place_poses.end()) {
                continue;
            }
            
            if(picked.find(id) == picked.end())
            {
                action_busy = true;
                picked.insert(id);

                geometry_msgs::msg::Pose pose;
                pose.position = det.bbox.center.position;
                pose.orientation = det.bbox.center.orientation;

                pick_pose = std::make_pair(id, pose);
                send_path_goal(pose);
                action_busy = true;

                break;
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

        RCLCPP_INFO(this->get_logger(), "Enviando Goal (Pose) para Action...");

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
        if (feedback->recalculating_path) 
        {
            send_request(feedback->stop_pose);
        }

    }

    void path_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>> & goal_handle)
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

        RCLCPP_INFO(this->get_logger(), "Enviando Goal (Pose) para Action...");

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
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "Enviando Goal (Pose) para Action...");

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