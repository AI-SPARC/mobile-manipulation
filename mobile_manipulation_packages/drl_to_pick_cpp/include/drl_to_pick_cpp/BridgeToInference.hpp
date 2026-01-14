#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <mutex>

namespace drl_to_pick_cpp
{

class BridgeToInference : public rclcpp::Node
{
public:
  explicit BridgeToInference(const rclcpp::NodeOptions & options);

 
  std::vector<geometry_msgs::msg::Pose> get_latest_grasps();

private:
  
  void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  std::vector<geometry_msgs::msg::Pose> get_grasps_from_server(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud);

  geometry_msgs::msg::Pose matrix_to_pose(const Eigen::Matrix4f & matrix);
  void publish_grasps(const std::vector<geometry_msgs::msg::Pose> & grasps);

  
  std::vector<geometry_msgs::msg::Pose> latest_grasps_;
  std::mutex grasp_mutex_; 

  
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_grasps_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;

 
  std::string server_host_;
  int server_port_;
  float score_threshold_;
  int max_grasps_;
  std::string target_frame_;
};

}  // namespace drl_to_pick_cpp