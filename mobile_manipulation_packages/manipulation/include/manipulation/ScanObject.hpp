#ifndef MANIPULATION__SCAN_OBJECT_HPP_
#define MANIPULATION__SCAN_OBJECT_HPP_

#include <string>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/header.hpp>

namespace manipulation {

class ScanObject : public rclcpp::Node
{
public:
    explicit ScanObject(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);

    void appendObjectMarkers(
        const std::string& label,
        int object_index,
        const geometry_msgs::msg::Pose& pose,
        const geometry_msgs::msg::Vector3& size,
        const std_msgs::msg::Header& header,
        visualization_msgs::msg::MarkerArray& marker_array);

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_detections_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    std::string target_frame_;
};

} // namespace manipulation

#endif