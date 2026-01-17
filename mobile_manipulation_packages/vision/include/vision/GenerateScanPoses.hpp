#ifndef MANIPULATION__GENERATE_SCAN_POSES_HPP_
#define MANIPULATION__GENERATE_SCAN_POSES_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <mobile_manipulation_interfaces/msg/semantic_pcl.hpp> 
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <mutex>
#include <string>
#include <vector>
#include <map>

namespace vision {

// Estruturas auxiliares
struct VoxelKey {
    int x, y, z;
    bool operator<(const VoxelKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct ScanPoint {
    geometry_msgs::msg::Pose pose;
    tf2::Vector3 position;
    tf2::Vector3 target_center;
    int face_id; 
};

struct TargetVoxel {
    tf2::Vector3 position;
    tf2::Vector3 normal;
    bool covered = false; 
};

struct ObjectData {
    std::string label;
    vision_msgs::msg::Detection3D detection;
    std_msgs::msg::Header header;
    std::vector<ScanPoint> valid_scan_grid;
};

class GenerateScanPoses : public rclcpp::Node
{
public:
    explicit GenerateScanPoses(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~GenerateScanPoses() = default;

    
    std::optional<std::pair<std::vector<geometry_msgs::msg::Pose>, tf2::Vector3>> getSortedScanPoses(const std::string& label);
    std::vector<geometry_msgs::msg::Pose> getOptimizedScanPoses(
        const std::vector<geometry_msgs::msg::Pose>& sorted_candidates, 
        const std::string& label);
        
private:
    
    std::string target_frame_;
    std::string odom_topic_;
    double ray_length_;
    double grid_resolution_;
    double voxel_map_resolution_;
    double ray_step_size_;
    std::string target_object_id_;
    bool publish_markers_; 
    double max_incidence_angle_rad_;
    
    int num_cameras_;

    double camera_fov_h_rad_;
    double camera_fov_v_rad_;
    double target_surface_res_;
    double min_coverage_percent_;

    
    tf2::Vector3 robot_position_;
    bool robot_pos_received_;
    std::mutex robot_pos_mutex_;

    std::map<VoxelKey, std::string> voxel_grid_;
    std::mutex voxel_mutex_;

    std::map<std::string, ObjectData> detected_objects_;
    std::mutex objects_mutex_;
    std_msgs::msg::Header last_header_;

    
    std::vector<geometry_msgs::msg::Pose> poses_to_animate_; 
    std::vector<TargetVoxel> debug_voxels_;
    std::mutex anim_mutex_;

    std::vector<rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr> sub_detections_;
    
    rclcpp::Subscription<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr sub_semantic_pcl_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr animation_timer_;

    
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
    void semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg);
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void animationTimerCallback();

    
    std::vector<ScanPoint> computeValidScanningGrid(
        const geometry_msgs::msg::Pose& target_pose,
        const geometry_msgs::msg::Vector3& target_size,
        const std::string& target_label);
    
    bool isRayBlocked(const tf2::Vector3& start, const tf2::Vector3& end, const std::string& target_label);
    VoxelKey pointToVoxel(const tf2::Vector3& pt);

  
    std::vector<TargetVoxel> generateTargetVoxels(const vision_msgs::msg::Detection3D& detection);
    bool isVoxelVisible(const geometry_msgs::msg::Pose& pose, const TargetVoxel& voxel);
    std::vector<geometry_msgs::msg::Pose> filterPosesByCoverage(
        const std::vector<geometry_msgs::msg::Pose>& candidates, 
        std::vector<TargetVoxel>& targets);
};

} // namespace vision

#endif