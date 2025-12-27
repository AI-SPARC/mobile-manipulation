#include "manipulation/CloudBoxFilter.hpp"

#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <random>

namespace manipulation
{

CloudBoxFilter::CloudBoxFilter(const rclcpp::NodeOptions & options)
: Node("cloud_box_filter", options),
  last_filtered_points_(new pcl::PointCloud<pcl::PointXYZ>())
{
  declare_parameter("debug_filtered_points", false);
  debug_filtered_points_ = get_parameter("debug_filtered_points").as_bool();

  if (debug_filtered_points_) 
  {
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_points", 10);
  }

  subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "/depth_pcl", 10,
    std::bind(&CloudBoxFilter::topic_callback, this, std::placeholders::_1));

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

void CloudBoxFilter::set_bounding_box(
  const geometry_msgs::msg::Pose & pose,
  const geometry_msgs::msg::Vector3 & size)
{
  bbox_pose_ = pose;
  bbox_size_ = size;

  Eigen::Quaternionf q(
    static_cast<float>(pose.orientation.w),
    static_cast<float>(pose.orientation.x),
    static_cast<float>(pose.orientation.y),
    static_cast<float>(pose.orientation.z));
  
  Eigen::Translation3f t(
    static_cast<float>(pose.position.x),
    static_cast<float>(pose.position.y),
    static_cast<float>(pose.position.z));

  bbox_transform_ = t * q;
  bbox_configured_ = true;

  RCLCPP_INFO(get_logger(), 
    "Bounding box set: pos(%.3f, %.3f, %.3f) size(%.3f, %.3f, %.3f)",
    pose.position.x, pose.position.y, pose.position.z,
    size.x, size.y, size.z);
}

void CloudBoxFilter::register_callback(
  std::function<void(const pcl::PointCloud<pcl::PointXYZ>::Ptr &)> callback)
{
  callbacks_.push_back(std::move(callback));
}

bool CloudBoxFilter::has_points() const
{
  std::lock_guard<std::mutex> lock(cloud_mutex_);
  return last_filtered_points_ && !last_filtered_points_->empty();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CloudBoxFilter::get_filtered_points() const
{
  std::lock_guard<std::mutex> lock(cloud_mutex_);

  return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*last_filtered_points_);
}

void CloudBoxFilter::topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!bbox_configured_) {
    return;
  }

  geometry_msgs::msg::TransformStamped t_stamped;
  try {
    t_stamped = tf_buffer_->lookupTransform("world", msg->header.frame_id, tf2::TimePointZero);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "TF error: %s", ex.what());
    return;
  }

  Eigen::Affine3f sensor_to_world = tf2::transformToEigen(t_stamped).cast<float>();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*msg, *cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = filter_points_in_box(cloud, sensor_to_world);

  if (filtered->empty()) {
    return;
  }

  if (filtered->size() < TARGET_POINTS) 
  {
    filtered = upsample_points(filtered);
  }

  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    last_filtered_points_ = filtered;
  }

  for (const auto & callback : callbacks_) 
  {
    callback(filtered);
  }

  if (debug_filtered_points_ && publisher_) 
  {
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*filtered, output);
    output.header.stamp = msg->header.stamp;
    output.header.frame_id = "world";
    publisher_->publish(output);
  }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CloudBoxFilter::filter_points_in_box(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
  const Eigen::Affine3f & sensor_to_world)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_world(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(*cloud, *cloud_world, sensor_to_world);

  Eigen::Vector4f min_pt(
    static_cast<float>(-bbox_size_.x / 2.0),
    static_cast<float>(-bbox_size_.y / 2.0),
    static_cast<float>(-bbox_size_.z / 2.0),
    1.0f);
  
  Eigen::Vector4f max_pt(
    static_cast<float>(bbox_size_.x / 2.0),
    static_cast<float>(bbox_size_.y / 2.0),
    static_cast<float>(bbox_size_.z / 2.0),
    1.0f);

  pcl::CropBox<pcl::PointXYZ> crop;
  crop.setInputCloud(cloud_world);
  crop.setMin(min_pt);
  crop.setMax(max_pt);
  crop.setTransform(bbox_transform_.inverse());

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
  crop.filter(*filtered);

  return filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CloudBoxFilter::upsample_points(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr upsampled(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
  upsampled->reserve(TARGET_POINTS);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> noise(0.0f, 0.0005f);

  const size_t original_size = cloud->size();
  
  while (upsampled->size() < TARGET_POINTS) {
    const pcl::PointXYZ & src = cloud->points[upsampled->size() % original_size];
    pcl::PointXYZ p;
    p.x = src.x + noise(gen);
    p.y = src.y + noise(gen);
    p.z = src.z + noise(gen);
    upsampled->push_back(p);
  }

  return upsampled;
}

}  // namespace manipulation

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(manipulation::CloudBoxFilter)