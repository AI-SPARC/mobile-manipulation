#ifndef COMBINED_SEMANTIC_PCL_HPP_
#define COMBINED_SEMANTIC_PCL_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp> // Incluído para BBox3D
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <mutex>
#include <map>
#include <vector>
#include <string>
#include <regex>

#include "mobile_manipulation_interfaces/msg/semantic_pcl.hpp"

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
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pcl_msg,
        const vision_msgs::msg::Detection3DArray::ConstSharedPtr & bbox_msg);

    // Helpers
    void parseLabelsJson(const std::string & json_str);
    std::string extractCleanLabel(std::string raw_label);
    int32_t getPixelId(const sensor_msgs::msg::Image::ConstSharedPtr & img, size_t index);
    std::tuple<uint8_t, uint8_t, uint8_t> getColorForId(int32_t obj_id);

  
    sensor_msgs::msg::PointCloud2 createPCLMsg(
        const std::vector<std::array<float, 3>>& points, 
        const std_msgs::msg::Header& header);

    
    void publishSemanticPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

    void publishColoredPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

   
    void publishSplitSemanticPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const vision_msgs::msg::Detection3DArray::ConstSharedPtr & bbox_msg,
        const std_msgs::msg::Header & header);

   
    std::string target_frame_;
    int downsample_step_;

    size_t frame_count_;

    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr labels_sub_;
    
    
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> seg_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pcl_sub_;
    std::shared_ptr<message_filters::Subscriber<vision_msgs::msg::Detection3DArray>> bbox_sub_; 

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::PointCloud2,
        vision_msgs::msg::Detection3DArray
    >;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_semantic_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_colored_;
    rclcpp::Publisher<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr pub_custom_msg_;

    
    std::mutex map_mutex_;
    std::map<int32_t, std::string> id_to_label_;
    std::map<int32_t, std::tuple<uint8_t, uint8_t, uint8_t>> color_map_;
};

} // namespace semantic_pcl

#endif // COMBINED_SEMANTIC_PCL_HPP_