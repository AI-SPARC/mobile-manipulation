#include "manipulation/ScanObject.hpp"

#include <cmath>
#include <array>
#include <vector>
#include <string>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Vector3.h>

using namespace std::chrono_literals;

namespace manipulation {

ScanObject::ScanObject(const rclcpp::NodeOptions & options)
 : Node("scan_object_visualizer", options)
{
    // Frame de destino para os markers
    target_frame_ = "world";

    sub_detections_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
        "/bbox_3d_with_labels", 10,
        std::bind(&ScanObject::detectionCallback, this, std::placeholders::_1));

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/visualization_marker_array", 10);

    RCLCPP_INFO(this->get_logger(), "ScanObject Visualizer iniciado. Publicando no frame: %s", target_frame_.c_str());
}   

void ScanObject::detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
{
    visualization_msgs::msg::MarkerArray all_markers;

    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    all_markers.markers.push_back(delete_all);

    std_msgs::msg::Header target_header;
    target_header.stamp = msg->header.stamp;
    target_header.frame_id = target_frame_;  

    int obj_idx = 0;
    for (const auto& detection : msg->detections)
    {
        if (detection.results.empty()) continue;

        std::string label = detection.results[0].hypothesis.class_id;
        
        geometry_msgs::msg::Pose pose = detection.bbox.center;
        geometry_msgs::msg::Vector3 size = detection.bbox.size;

        appendObjectMarkers(label, obj_idx, pose, size, target_header, all_markers);
        obj_idx++;
    }

    if (!all_markers.markers.empty()) {
        marker_pub_->publish(all_markers);
    }
}

void ScanObject::appendObjectMarkers(
    const std::string& label,
    int object_index,
    const geometry_msgs::msg::Pose& pose,
    const geometry_msgs::msg::Vector3& size,
    const std_msgs::msg::Header& header,
    visualization_msgs::msg::MarkerArray& marker_array)
{
    std::string ns_suffix = "_" + label + "_" + std::to_string(object_index);

    visualization_msgs::msg::Marker bbox_marker;
    bbox_marker.header = header;
    bbox_marker.ns = "bbox" + ns_suffix;
    bbox_marker.id = 0;
    bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
    bbox_marker.action = visualization_msgs::msg::Marker::ADD;
    bbox_marker.pose = pose; 
    bbox_marker.scale = size;
    
    bbox_marker.color.r = 0.0f;
    bbox_marker.color.g = 1.0f;
    bbox_marker.color.b = 0.0f;
    bbox_marker.color.a = 0.3f;
    bbox_marker.lifetime = rclcpp::Duration(0, 500000000);

    marker_array.markers.push_back(bbox_marker);

    tf2::Vector3 center(pose.position.x, pose.position.y, pose.position.z);
    tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
    tf2::Matrix3x3 rot_matrix(q);

    double dx = size.x / 2.0;
    double dy = size.y / 2.0;
    double dz = size.z / 2.0;

    std::vector<std::array<double, 3>> local_corners = {
        { dx,  dy,  dz}, { dx,  dy, -dz}, { dx, -dy,  dz}, { dx, -dy, -dz},
        {-dx,  dy,  dz}, {-dx,  dy, -dz}, {-dx, -dy,  dz}, {-dx, -dy, -dz}
    };

    int ray_id = 1;
    for (const auto& c : local_corners)
    {
        tf2::Vector3 local_corner(c[0], c[1], c[2]);
        tf2::Vector3 rotated_corner_offset = rot_matrix * local_corner;
        tf2::Vector3 absolute_corner = center + rotated_corner_offset;
        tf2::Vector3 direction = (absolute_corner - center).normalize();

        double ray_length = 0.125;
        tf2::Vector3 ray_end = absolute_corner + (direction * ray_length);

        visualization_msgs::msg::Marker ray_marker;
        ray_marker.header = header;
        ray_marker.ns = "rays" + ns_suffix; 
        ray_marker.id = ray_id++;
        ray_marker.type = visualization_msgs::msg::Marker::ARROW;
        ray_marker.action = visualization_msgs::msg::Marker::ADD;
        
        geometry_msgs::msg::Point p_start, p_end;
        p_start.x = absolute_corner.x(); p_start.y = absolute_corner.y(); p_start.z = absolute_corner.z();
        p_end.x = ray_end.x(); p_end.y = ray_end.y(); p_end.z = ray_end.z();
        
        ray_marker.points.push_back(p_start);
        ray_marker.points.push_back(p_end);

        ray_marker.scale.x = 0.005;
        ray_marker.scale.y = 0.01;
        ray_marker.scale.z = 0.0; 

        ray_marker.color.r = 1.0f;
        ray_marker.color.g = 0.0f;
        ray_marker.color.b = 0.0f;
        ray_marker.color.a = 1.0f;
        ray_marker.lifetime = rclcpp::Duration(0, 500000000);

        marker_array.markers.push_back(ray_marker);
    }
}

} // namespace manipulation

// int main(int argc, char ** argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<manipulation::ScanObject>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }