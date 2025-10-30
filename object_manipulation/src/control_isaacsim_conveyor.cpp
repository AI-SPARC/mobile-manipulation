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
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/msg/float32.hpp>
#include <cmath> 
#include "std_msgs/msg/bool.hpp"
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


class PickAndOrganize : public rclcpp::Node {

private:

    struct LocationData
    {
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size;
    };


    struct BoxSpace 
    {
        double length;  
        double width;  
        double height; 
        double origin_x;
        double origin_y;
        double origin_z;
    };

    struct OccupiedCell 
    {
        double x, y, z;
        double sx, sy, sz;
    };

    struct Vec3 
    {
        float x, y, z;
    };

    //Publishers.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_1;

    //Subscriptions.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;

    //Service.
    rclcpp::Client<object_manipulation_interfaces::srv::ObjectCollision>::SharedPtr client_;
    rclcpp::Client<object_manipulation_interfaces::srv::PickedObject>::SharedPtr client_1;
    
    //Timer.
    rclcpp::TimerBase::SharedPtr init_timer_;

    std::string yaml_file;

    std::unordered_set<std::string> pick_and_place_poses;

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
        
    /*
    
        PUBLISHERS.
    
    */

    void publish_velocity(float velocity)
    {
        auto message = std_msgs::msg::Float32();
        message.data = velocity;

        publisher_->publish(message);

    }

    void publish_angular_velocity(float velocity)
    {
        auto message = std_msgs::msg::Float32();
        message.data = velocity;

        publisher_1->publish(message);

    }
 
    /*
    
        CALLBACKS.

    */
    
    std::string pick_and_place_id;
    bool stopped = false, welding_done = false;

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
            
            if(stopped == true && pick_and_place_id == det.results[0].hypothesis.class_id)
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
                pick_and_place_id = ' ';
            }
           

            if(det.bbox.center.position.y < 0.1 && det.bbox.center.position.y > -0.1 && det.bbox.center.position.x > 0.25 && stopped == false && pick_and_place_id != det.results[0].hypothesis.class_id)
            {
                publish_velocity(0.0);
                publish_angular_velocity(0.0);

                rclcpp::sleep_for(std::chrono::milliseconds(1000));
                send_request(true);
                
                pick_and_place_id = det.results[0].hypothesis.class_id;
                stopped = true;

                break;
            }

            if(stopped == false)
            {
                publish_velocity(0.2);
                publish_angular_velocity(0.375);
            }
           

        }
    }

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
                    stopped = false;
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


            

public:
    PickAndOrganize()
     : Node("pick_and_organize")
    {
        this->declare_parameter<std::string>("yaml_file", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        
        publisher_ = this->create_publisher<std_msgs::msg::Float32>("/conveyor_velocity", 10);
        publisher_1 = this->create_publisher<std_msgs::msg::Float32>("/conveyor_angular_velocity", 10);

        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&PickAndOrganize::detectionCallback, this, std::placeholders::_1));

        client_ = this->create_client<object_manipulation_interfaces::srv::ObjectCollision>(
            "/object_collision");

        client_1 = this->create_client<object_manipulation_interfaces::srv::PickedObject>(
            "/picked_object");

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

