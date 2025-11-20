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

#include "mobile_manipulation_interfaces/srv/mobile_picked_object.hpp"
#include "mobile_manipulation_interfaces/srv/mobile_goal_pose.hpp"
#include "mobile_manipulation_interfaces/action/pick_object.hpp"

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
        
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(
            this, "pick_object");

        if(!yaml_file.empty()) {
            loadLocationsFromYaml(yaml_file);
        }
    } 

private:
    bool action_busy = false;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;

    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    
    std::string yaml_file;
    std::unordered_set<std::string> pick_and_place_poses;
    std::unordered_set<std::string> picked;

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
        if (action_busy) {
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

                geometry_msgs::msg::Vector3 size = det.bbox.size;

                send_goal(id, pose);

                break;
            }
        }
    }

    //Actions.

    void send_goal(const std::string id, const geometry_msgs::msg::Pose & target_pose)
    {
        if (!this->client_ptr_->wait_for_action_server(std::chrono::seconds(5))) 
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            action_busy = false;
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::PickObject::Goal();
        
        goal_msg.obstacle_id = id;
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "Enviando Goal (Pose) para Action...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SendGoalOptions();
        
        send_goal_options.goal_response_callback = 
            std::bind(&ServerNode::goal_response_callback, this, std::placeholders::_1);
            
        send_goal_options.result_callback = 
            std::bind(&ServerNode::result_callback, this, std::placeholders::_1);

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

        RCLCPP_INFO(this->get_logger(), "O resultado foi: %s", result.result->success ? "true" : "false");
                
        action_busy = false;
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ServerNode>());
  rclcpp::shutdown();
  return 0;
}