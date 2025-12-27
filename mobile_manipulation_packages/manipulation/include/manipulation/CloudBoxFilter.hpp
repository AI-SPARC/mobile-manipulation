#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <functional>
#include <mutex>
#include <vector>

namespace manipulation
{

class CloudBoxFilter : public rclcpp::Node
{
public:
  explicit CloudBoxFilter(const rclcpp::NodeOptions & options);

  /// Configura a bounding box (chamado pelo nó pai)
  void set_bounding_box(
    const geometry_msgs::msg::Pose & pose,
    const geometry_msgs::msg::Vector3 & size);

  /// Registra callback para receber a point cloud filtrada (nó pai)
  void register_callback(
    std::function<void(const pcl::PointCloud<pcl::PointXYZ>::Ptr &)> callback);

  /// Getters
  bool has_points() const;
  pcl::PointCloud<pcl::PointXYZ>::Ptr get_filtered_points() const;

private:
  static constexpr size_t TARGET_POINTS = 512;

  void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr filter_points_in_box(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const Eigen::Affine3f & sensor_to_world);
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr upsample_points(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud);

  // ROS
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Parâmetros
  bool debug_filtered_points_;

  // Bounding box
  geometry_msgs::msg::Pose bbox_pose_;
  geometry_msgs::msg::Vector3 bbox_size_;
  Eigen::Affine3f bbox_transform_;
  bool bbox_configured_ = false;

  // Cache e callback
  mutable std::mutex cloud_mutex_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr last_filtered_points_;
  std::vector<std::function<void(const pcl::PointCloud<pcl::PointXYZ>::Ptr &)>> callbacks_;
};

}  // namespace manipulation