#ifndef MANIPULATION__SCAN_OBJECT_HPP_
#define MANIPULATION__SCAN_OBJECT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "mobile_manipulation_interfaces/msg/object_data.hpp"
#include "mobile_manipulation_interfaces/msg/semantic_pcl.hpp" 

#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <memory>

namespace manipulation {

// --- VOXEL HASHING ---
struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& other) const { return x == other.x && y == other.y && z == other.z; }
};
struct VoxelHash {
    std::size_t operator()(const VoxelKey& k) const {
        size_t h1 = std::hash<int>{}(k.x); size_t h2 = std::hash<int>{}(k.y); size_t h3 = std::hash<int>{}(k.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2); 
    }
};

// --- DADOS ---
struct ScanPoint {
    tf2::Vector3 position;
    tf2::Vector3 target_center;
    int face_id; // Identificador da face para agrupamento
};

struct ObjectData {
    std::string label;
    vision_msgs::msg::Detection3D detection;
    std_msgs::msg::Header header;
    std::vector<ScanPoint> valid_scan_grid;
};

class ScanObject : public rclcpp::Node
{
public:
    explicit ScanObject(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    virtual ~ScanObject() = default;

    std::vector<geometry_msgs::msg::Point> getSortedScanPoints(const std::string& label);

private:
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg);
    void animationTimerCallback();

    std::vector<ScanPoint> computeValidScanningGrid(
        const geometry_msgs::msg::Pose& target_pose,
        const geometry_msgs::msg::Vector3& target_size,
        const std::string& target_label);

    bool isRayBlocked(const tf2::Vector3& start, const tf2::Vector3& end, const std::string& target_label);
    VoxelKey pointToVoxel(const tf2::Vector3& pt);

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_detections_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
    rclcpp::Subscription<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr sub_semantic_pcl_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr animation_timer_;

    std::mutex robot_pos_mutex_;
    tf2::Vector3 robot_position_;
    bool robot_pos_received_;

    std::mutex objects_mutex_;
    std::map<std::string, ObjectData> detected_objects_;
    std_msgs::msg::Header last_header_;

    std::mutex voxel_mutex_;
    std::unordered_map<VoxelKey, std::string, VoxelHash> voxel_grid_;

    // Animação
    std::mutex anim_mutex_;
    std::vector<geometry_msgs::msg::Point> points_to_animate_; 
    vision_msgs::msg::Detection3D current_anim_bbox_;
    size_t current_anim_index_; 
    bool is_animating_; 

    // Parâmetros
    std::string target_frame_;
    std::string odom_topic_;
    double ray_length_;
    double grid_resolution_;
    double ray_step_size_;
    std::string target_object_id_;
    double voxel_map_resolution_; 
};

} // namespace manipulation

#endif // MANIPULATION__SCAN_OBJECT_HPP_