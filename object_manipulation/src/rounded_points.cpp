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
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tuple>
#include <vector>
#include <cmath>

using namespace std::chrono_literals;


class AStar : public rclcpp::Node {

private:

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_rounded_points;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr parameterTimer;    

    float distanceToObstacle_ = 0.01;
    int decimals = 0;

    std::vector<std::tuple<float, float, float>> rounded_point_cloud;

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> depth_camera_sub_1;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> depth_camera_sub_2;
    std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>> sync_;

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

    void publish_created_vertices()
    {
        if (rounded_point_cloud.empty()) return;

        sensor_msgs::msg::PointCloud2 cloud_msgs_created_vertices;
        cloud_msgs_created_vertices.header.stamp = this->get_clock()->now();
        cloud_msgs_created_vertices.header.frame_id = "world";

        cloud_msgs_created_vertices.height = 1; 
        cloud_msgs_created_vertices.width = rounded_point_cloud.size(); 
        cloud_msgs_created_vertices.is_dense = true;
        cloud_msgs_created_vertices.is_bigendian = false;
        cloud_msgs_created_vertices.point_step = 3 * sizeof(float); 
        cloud_msgs_created_vertices.row_step = cloud_msgs_created_vertices.point_step * cloud_msgs_created_vertices.width;

        sensor_msgs::PointCloud2Modifier modifier(cloud_msgs_created_vertices);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(cloud_msgs_created_vertices.width);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msgs_created_vertices, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msgs_created_vertices, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msgs_created_vertices, "z");
        for (const auto& vertex : rounded_point_cloud) 
        {
            *iter_x = std::get<0>(vertex);
            *iter_y = std::get<1>(vertex);
            *iter_z = std::get<2>(vertex);
            ++iter_x;
            ++iter_y;
            ++iter_z;    
        }

        publisher_rounded_points->publish(cloud_msgs_created_vertices);
    }

    void sync_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg1,
                       const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg2)
    {
        rounded_point_cloud.clear();

        // --- Transformar nuvem 1 para "world" ---
        sensor_msgs::msg::PointCloud2 cloud1_world;
        try {
            auto transform_stamped =
                tf_buffer_->lookupTransform("world", "depth_optical_frame_1", msg1->header.stamp, 1ms);
            tf2::doTransform(*msg1, cloud1_world, transform_stamped);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Transform depth_camera_1: %s", ex.what());
            return;
        }

        // --- Transformar nuvem 2 para "world" ---
        sensor_msgs::msg::PointCloud2 cloud2_world;
        try {
            auto transform_stamped =
                tf_buffer_->lookupTransform("world", "depth_optical_frame_2", msg2->header.stamp, 1ms);
            tf2::doTransform(*msg2, cloud2_world, transform_stamped);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Transform depth_camera_2: %s", ex.what());
            return;
        }

        // --- Iterar pontos nuvem 1 ---
        for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud1_world, "x"), 
                                                          iter_y(cloud1_world, "y"),
                                                          iter_z(cloud1_world, "z");
             iter_x != iter_x.end();
             ++iter_x, ++iter_y, ++iter_z)
        {
            if(*iter_z > 0.01 && *iter_z < 0.1)
            {
                float x = round_to_multiple(*iter_x, distanceToObstacle_, decimals);
                float y = round_to_multiple(*iter_y, distanceToObstacle_, decimals);
                float z = round_to_multiple(*iter_z, distanceToObstacle_, decimals);
                rounded_point_cloud.emplace_back(x, y, z);
            }
        }

        // --- Iterar pontos nuvem 2 ---
        for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud2_world, "x"), 
                                                          iter_y(cloud2_world, "y"),
                                                          iter_z(cloud2_world, "z");
             iter_x != iter_x.end();
             ++iter_x, ++iter_y, ++iter_z)
        {
            if(*iter_z > 0.01 && *iter_z < 0.1)
            {
                float x = round_to_multiple(*iter_x, distanceToObstacle_, decimals);
                float y = round_to_multiple(*iter_y, distanceToObstacle_, decimals);
                float z = round_to_multiple(*iter_z, distanceToObstacle_, decimals);
                rounded_point_cloud.emplace_back(x, y, z);
            }
        }
        publish_created_vertices();
    }

    //     void sync_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg1,
    //                    const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg2)
    // {
    //     rounded_point_cloud.clear();

    //     // --- Transformar nuvem 1 para "world" ---
    //     sensor_msgs::msg::PointCloud2 cloud1_world;
    //     try {
    //         auto transform_stamped =
    //             tf_buffer_->lookupTransform("world", "depth_optical_frame_1", msg1->header.stamp, 1ms);
    //         tf2::doTransform(*msg1, cloud1_world, transform_stamped);
    //     } catch (tf2::TransformException &ex) {
    //         RCLCPP_WARN(this->get_logger(), "Transform depth_camera_1: %s", ex.what());
    //         return;
    //     }

    //     // --- Transformar nuvem 2 para "world" ---
    //     sensor_msgs::msg::PointCloud2 cloud2_world;
    //     try {
    //         auto transform_stamped =
    //             tf_buffer_->lookupTransform("world", "depth_optical_frame_2", msg2->header.stamp, 1ms);
    //         tf2::doTransform(*msg2, cloud2_world, transform_stamped);
    //     } catch (tf2::TransformException &ex) {
    //         RCLCPP_WARN(this->get_logger(), "Transform depth_camera_2: %s", ex.what());
    //         return;
    //     }

    //     // --- Iterar pontos nuvem 1 ---
    //     for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud1_world, "x"), 
    //                                       iter_y(cloud1_world, "y"),
    //                                       iter_z(cloud1_world, "z");
    //         iter_x != iter_x.end();
    //         ++iter_x, ++iter_y, ++iter_z)
    //     {
    //         rounded_point_cloud.emplace_back(*iter_x, *iter_y, *iter_z);
    //     }

    //     // --- Iterar pontos nuvem 2 ---
    //     for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud2_world, "x"), 
    //                                             iter_y(cloud2_world, "y"),
    //                                             iter_z(cloud2_world, "z");
    //         iter_x != iter_x.end();
    //         ++iter_x, ++iter_y, ++iter_z)
    //     {
    //         rounded_point_cloud.emplace_back(*iter_x, *iter_y, *iter_z);
    //     }
    //     publish_created_vertices();
    // }

public:
    AStar()
     : Node("Bug_1")
    {
        parameterTimer = this->create_wall_timer(
            std::chrono::milliseconds(200),
            std::bind(&AStar::publish_created_vertices, this));

        decimals = count_decimals(distanceToObstacle_);

        publisher_rounded_points = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points", 10);

        depth_camera_sub_1 = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, "/depth_camera_1/points");
        depth_camera_sub_2 = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(this, "/depth_camera_2/points");

        sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>>(
            *depth_camera_sub_1, *depth_camera_sub_2, 10);
        sync_->registerCallback(std::bind(&AStar::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

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
