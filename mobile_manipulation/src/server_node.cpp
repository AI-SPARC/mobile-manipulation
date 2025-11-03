#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <yaml-cpp/yaml.h>
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
#include "object_manipulation_interfaces/srv/picked_object.hpp"
#include "object_manipulation_interfaces/srv/goal_pose.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/msg/float32.hpp>
#include <cmath> 
#include "std_msgs/msg/bool.hpp"

using namespace std::chrono_literals;

class PickAndOrganize : public rclcpp::Node {

private:

    //Publishers.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_1;

    //Subscriptions.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;

    //Service.
    rclcpp::Client<object_manipulation_interfaces::srv::ObjectCollision>::SharedPtr client_;
    rclcpp::Client<object_manipulation_interfaces::srv::PickedObject>::SharedPtr client_1;
    rclcpp::Client<object_manipulation_interfaces::srv::GoalPose>::SharedPtr client_2;
    
    //Timer.
    rclcpp::TimerBase::SharedPtr init_timer_;

    std::string yaml_file;

    std::unordered_set<std::string> pick_and_place_poses;
    std::unordered_set<std::string> picked;

    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);

            for (const auto &label_node : config)
            {
                const std::string label = label_node.first.as<std::string>();

                pick_and_place_poses.insert(label);
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                        "Failed to load YAML file '%s': %s", yaml_path.c_str(), e.what());
        }
    }
        
 
    
    
    // Callbacks.

    
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        std::string id;
        for (const auto &det : msg->detections)
        {
            
            size_t pos = det.results[0].hypothesis.class_id.find('_'); 

            if (pos != std::string::npos) 
            {
                id = det.results[0].hypothesis.class_id.substr(0, pos);  
            } 
            else
            {
                id = det.results[0].hypothesis.class_id;  
            }

            if (det.results.empty() || pick_and_place_poses.find(id) == pick_and_place_poses.end())
            {
                continue;
            }
            
            if(picked.find(id) == picked.end())
            {
                rclcpp::sleep_for(std::chrono::milliseconds(100));
                geometry_msgs::msg::Pose pose;
                pose.position.x = det.bbox.center.position.x;
                pose.position.y = det.bbox.center.position.y;
                pose.position.z = det.bbox.center.position.z;
                pose.orientation.x = det.bbox.center.orientation.x;
                pose.orientation.y = det.bbox.center.orientation.y;
                pose.orientation.z = det.bbox.center.orientation.z;
                pose.orientation.w = det.bbox.center.orientation.w;

                geometry_msgs::msg::Vector3 size;
                size.x = det.bbox.size.x;
                size.y = det.bbox.size.y;
                size.z = det.bbox.size.z;


                send_picked_object(det.results[0].hypothesis.class_id, pose, size);
                send_goal_pose(pose);
                send_request(true);

                picked.insert(id);
            }
            


        }
    }

    // Services

    void send_request(bool stop_flag)
    {
        auto request = std::make_shared<object_manipulation_interfaces::srv::ObjectCollision::Request>();
        request->stop = stop_flag;

      
        client_->async_send_request(request,
            [this](rclcpp::Client<object_manipulation_interfaces::srv::ObjectCollision>::SharedFuture future_response) 
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
    
    void send_picked_object(std::string received_id, geometry_msgs::msg::Pose received_pose, geometry_msgs::msg::Vector3 received_size)
    {
        auto request = std::make_shared<object_manipulation_interfaces::srv::PickedObject::Request>();
        request->id = received_id;
        request->pose = received_pose;
        request->size = received_size;
      
        client_1->async_send_request(request,
            [this](rclcpp::Client<object_manipulation_interfaces::srv::PickedObject>::SharedFuture future_response) 
            {
                auto response = future_response.get();  

                if (response->success) 
                {
                    send_request(false);
                    RCLCPP_INFO(this->get_logger(), "Service executado com sucesso!");
                } 
                else 
                {
                    RCLCPP_WARN(this->get_logger(), "Falha ao executar service");
                }
            }
        );
    }

    void send_goal_pose(geometry_msgs::msg::Pose received_pose)
    {
        auto request = std::make_shared<object_manipulation_interfaces::srv::GoalPose::Request>();
    
        request->pose = received_pose;
     
      
        client_2->async_send_request(request,
            [this](rclcpp::Client<object_manipulation_interfaces::srv::GoalPose>::SharedFuture future_response) 
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


            

public:
    PickAndOrganize()
     : Node("pick_and_organize")
    {
        this->declare_parameter<std::string>("yaml_file", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&PickAndOrganize::detectionCallback, this, std::placeholders::_1));

        client_ = this->create_client<object_manipulation_interfaces::srv::ObjectCollision>(
            "/object_collision");

        client_1 = this->create_client<object_manipulation_interfaces::srv::PickedObject>(
            "/picked_object");

        client_2 = this->create_client<object_manipulation_interfaces::srv::GoalPose>(
            "/goal_pose");

        loadLocationsFromYaml(yaml_file);
    }   
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PickAndOrganize>());
  rclcpp::shutdown();
  return 0;
}

