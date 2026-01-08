#ifndef COMBINED_SEMANTIC_PCL_HPP
#define COMBINED_SEMANTIC_PCL_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// --- TF2 Includes ---
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>

#include <string>
#include <vector>
#include <map>
#include <tuple>

namespace semantic_pcl
{

class CombinedSemanticPCL : public rclcpp::Node
{
public:
    explicit CombinedSemanticPCL(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    virtual ~CombinedSemanticPCL() = default;

private:
    void labelsCallback(const std_msgs::msg::String::SharedPtr msg);
    
    void syncedCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr & seg_msg,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pcl_msg);

    void parseLabelsJson(const std::string & json_str);
    std::tuple<uint8_t, uint8_t, uint8_t> getColorForId(int32_t obj_id);

    void publishSemanticPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

    void publishColoredPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr labels_sub_;
    
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> seg_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pcl_sub_;

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::PointCloud2
    > SyncPolicy;
    
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_semantic_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_colored_;

    // --- TF2 Variables ---
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::string target_frame_; // Frame de destino (ex: "world")

    std::string topic_segmentation_;
    std::string topic_pointcloud_;
    std::string topic_labels_;
    std::string topic_output_semantic_;
    std::string topic_output_colored_;
    std::string frame_id_; // Frame original da câmera

    int downsample_step_;
    size_t frame_count_;
    std::map<int32_t, std::string> id_to_label_;
    std::map<int32_t, std::tuple<uint8_t, uint8_t, uint8_t>> color_map_;
};

} // namespace semantic_pcl

#endif // COMBINED_SEMANTIC_PCL_HPP