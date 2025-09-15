#include <string>
#include <random>
#include <algorithm>
#include <geometry_msgs/msg/point.hpp>
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <map>
#include <stack>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <climits>
#include <iomanip>
#include <thread>
#include <queue>
#include <tuple>
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>                       
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cmath>
#include <cstring>
#include <utility> 
#include <iomanip>
#include <filesystem>
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <set>

using namespace std::chrono_literals;



namespace std {
    template <>
    struct hash<std::tuple<float, float>> {
        size_t operator()(const std::tuple<float, float>& t) const {
            size_t h1 = hash<float>()(std::get<0>(t));
            size_t h2 = hash<float>()(std::get<1>(t));
            return h1 ^ (h2 << 1);  
        }
    };

    template <>
    struct hash<std::tuple<float, float, float>> {
        size_t operator()(const std::tuple<float, float, float>& t) const {
            size_t h1 = hash<float>()(std::get<0>(t));
            size_t h2 = hash<float>()(std::get<1>(t));
            size_t h3 = hash<float>()(std::get<2>(t));
            return h1 ^ (h2 << 1) ^ (h3 << 2); 
        }
    };
}




class AStar : public rclcpp::Node {

private:


    struct PointCompare {
        bool operator()(const std::tuple<float,float,float>& a, const std::tuple<float,float,float>& b) const {
            if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b); // x
            if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b); // y
            return std::get<2>(a) < std::get<2>(b); // z
        }
    };

    //Publishers.
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_rounded_points;

    //Subscriptions.
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr depth_camera_subscription_1;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr depth_camera_subscription_2;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr parameterTimer;    
    float distanceToObstacle_ = 0.01;
    int decimals = 0;



  
    inline float round_to_multiple(float value, float multiple, int decimals) 
    {
        if (multiple == 0.0) return value; 
        
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0, decimals);
        result = std::round(result * factor) / factor;
        
        return result;
    }
    

    int count_decimals(float number) 
    {
      
        float fractional = std::fabs(number - std::floor(number));
        int decimals = 0;
        const float epsilon = 1e-9; 
    
  
        while (fractional > epsilon && decimals < 20) {
            fractional *= 10;
            fractional -= std::floor(fractional);
            decimals++;
        }
        return decimals;
    }

    // std::vector<std::tuple<float,float,float>> points; // seus pontos
    // std::vector<bool> array_of_points;

    // void build_voxel_grid()
    // {
    //     if (points.empty()) return;

    //     // 1️⃣ encontrar limites
    //     float min_x = std::get<0>(points[0]), max_x = min_x;
    //     float min_y = std::get<1>(points[0]), max_y = min_y;
    //     float min_z = std::get<2>(points[0]), max_z = min_z;

    //     for (auto& p : points) {
    //         min_x = std::min(min_x, std::get<0>(p)); max_x = std::max(max_x, std::get<0>(p));
    //         min_y = std::min(min_y, std::get<1>(p)); max_y = std::max(max_y, std::get<1>(p));
    //         min_z = std::min(min_z, std::get<2>(p)); max_z = std::max(max_z, std::get<2>(p));
    //     }

    //     float resolution = 0.02f;
    //     int size_x = static_cast<int>(std::ceil((max_x - min_x) / resolution)) + 1;
    //     int size_y = static_cast<int>(std::ceil((max_y - min_y) / resolution)) + 1;
    //     int size_z = static_cast<int>(std::ceil((max_z - min_z) / resolution)) + 1;

    //     // 2️⃣ criar vetor de voxels
    //     array_of_points.assign(size_x * size_y * size_z, false);

    //     // 3️⃣ preencher voxels ocupados
    //     for (auto& p : points)
    //     {
    //         int ix = static_cast<int>((std::get<0>(p) - min_x) / resolution);
    //         int iy = static_cast<int>((std::get<1>(p) - min_y) / resolution);
    //         int iz = static_cast<int>((std::get<2>(p) - min_z) / resolution);

    //         int index = iz * size_x * size_y + iy * size_x + ix;
    //         array_of_points[index] = true;
    //     }
    // }


    /*

        PUBLISHERS.

    */


  
    /*
    
        CALLBACKS.

    */

    std::set<std::tuple<float,float,float>, PointCompare> points;

        
    void callback_object_point_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
        pcl::fromROSMsg(*msg, pcl_cloud);

        points.clear();

        for (const auto& p : pcl_cloud.points)
        {
            if (p.z > 0.05) 
            {
                points.emplace(p.x, p.y, p.z); 
            }
        }

        
            std::cout << points.size() << std::endl;
        

        
    }






public:
    AStar()
     : Node("Bug_1")
    {
        // this->declare_parameter<double>("path_resolution", 0.05);
        // this->declare_parameter<int>("diagonalEdges", 3);
        // this->declare_parameter<int>("iterations_before_verification", 10);

        // distanceToObstacle_ =  static_cast<float>(this->get_parameter("path_resolution").get_parameter_value().get<double>());
        // diagonalEdges_ = this->get_parameter("diagonalEdges").get_parameter_value().get<int>();
        // iterations_before_verification = this->get_parameter("iterations_before_verification").get_parameter_value().get<int>();

        // RCLCPP_INFO(this->get_logger(), "path_resolution is set to: %f", distanceToObstacle_);
        // RCLCPP_INFO(this->get_logger(), "diagonalEdges is set to: %d", diagonalEdges_);
        // RCLCPP_INFO(this->get_logger(), "iterations_before_verification is set to: %d", iterations_before_verification);

    
        decimals = count_decimals(distanceToObstacle_);


        depth_camera_subscription_1 = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rounded_points", 10, std::bind(&AStar::callback_object_point_cloud, this, std::placeholders::_1));

     
            tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }   
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    rclcpp::spin(std::make_shared<AStar>());
    rclcpp::shutdown();
    return 0;
}