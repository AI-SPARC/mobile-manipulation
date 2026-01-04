#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace drl_to_pick_cpp
{

class BridgeToInference : public rclcpp::Node
{
public:
  explicit BridgeToInference(const rclcpp::NodeOptions & options);

  /// Processa point cloud e retorna poses de grasp ordenadas por score (maior primeiro)
  std::vector<geometry_msgs::msg::Pose> process_point_cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud);

private:
  std::vector<geometry_msgs::msg::Pose> get_grasps_from_server(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud);

  geometry_msgs::msg::Pose matrix_to_pose(const Eigen::Matrix4f & matrix);

  void publish_grasps(const std::vector<geometry_msgs::msg::Pose> & grasps);

  // ROS
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_grasps_;

  // Parameters
  std::string server_host_;
  int server_port_;
  float score_threshold_;
  int max_grasps_;
  std::string target_frame_;
};

}  // namespace drl_to_pick_cpp