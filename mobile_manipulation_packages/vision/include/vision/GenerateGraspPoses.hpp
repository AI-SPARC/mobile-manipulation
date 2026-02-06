#ifndef VISION_GENERATE_GRASP_POSES_HPP_
#define VISION_GENERATE_GRASP_POSES_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

// Assimp
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <vector>
#include <string>
#include <Eigen/Dense>

namespace vision 
{

struct ScoredGrasp {
    geometry_msgs::msg::Pose pose_center;
    geometry_msgs::msg::Pose pose_finger1;
    geometry_msgs::msg::Pose pose_finger2;
    Eigen::Vector3f raw_ray_dir;
    Eigen::Vector3f raw_p_f1;
    Eigen::Vector3f raw_p_f2;
    double total_score;
    float entry_angle;
    float exit_angle;
    float entry_planarity;
    float exit_planarity;
    Eigen::Vector3f entry_normal;
    Eigen::Vector3f debug_entry_pt;
    Eigen::Vector3f debug_exit_pt;
    pcl::PointCloud<pcl::PointXYZ> debug_inliers;
};

struct StepAnalysis {
    bool valid;
    Eigen::Vector3f center;
    int point_count;
    float curvature;
    float angle_to_normal_deg;
    float symmetry_score;
    Eigen::Vector3f normal_vector;
};

struct LocalBox {
    Eigen::Vector3f min_pt;
    Eigen::Vector3f max_pt;
    Eigen::Vector3f center;     
    Eigen::Vector3f dimensions; 
};

struct VoxelBucket {
    Eigen::Vector3f center;
    std::vector<pcl::PointXYZ> points;
};

class GenerateGraspPoses : public rclcpp::Node 
{
public:
    explicit GenerateGraspPoses(const rclcpp::NodeOptions & options);

    geometry_msgs::msg::PoseArray processCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr target, pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment);

private:
    void timerCallback();
    void loadAndProcess(const std::string& path);
    
    std::vector<geometry_msgs::msg::Pose> generateMultiOrientedRays(
        const Eigen::Vector4f& min, const Eigen::Vector4f& max, float res);
        
    StepAnalysis analyzeLocalCylinder(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const Eigen::Vector3f& center,
        const Eigen::Vector3f& ray_dir,
        float radius,
        float height);
        
    Eigen::Quaternionf findBestOrientation(const Eigen::Vector3f& p_f1, const Eigen::Vector3f& p_f2);
    
    geometry_msgs::msg::PoseArray evaluateGrasps(pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment);

    void extractBoundingBoxesFromOBJ(); 
    bool check_collision(ScoredGrasp& grasp, const pcl::KdTreeFLANN<pcl::PointXYZ>& env_kdtree, bool publish_debug, bool try_rotations);
    
    void publishGripperModel();
    void publishGripperCollisionBoxes();
    void publishBest();

  
    std::string object_mesh_path_;
    std::string gripper_glb_path_;
    bool publish_object_mesh_;
    bool publish_gripper_mesh_;
    bool use_pcd_file;
    std::string pcd_path_;
    std::string gripper_mesh_path_;
    float gripper_mesh_scale_;
    
    float mesh_offset_x_, mesh_offset_y_, mesh_offset_z_;
    float mesh_rot_roll_, mesh_rot_pitch_, mesh_rot_yaw_;

    int num_benchmark_runs_, animation_delay_ms_;
    bool enable_ray_animation_;
    float grid_res_;
    float cloud_voxel_size_;
    float cylinder_radius_;
    float cylinder_height_;
    float analysis_step_size_;
    float max_gripper_width_;
    float finger_offset_;
    int min_points_per_segment_;
    float weight_orientation_;
    float weight_symmetry_;
    std::vector<float> ray_lengths;
    
    bool mean_filter;
    int mean_filter_k_;
    
    int num_best_grasps_;
    float rotation_step_deg_;
    int total_orientations_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr collision_cloud_; 
    pcl::KdTreeFLANN<pcl::PointXYZ> collision_kdtree_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rays_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_bbox_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_inliers_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_collision_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_object_mesh_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_gripper_model_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_gripper_boxes_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_grasps_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_marker_pub_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_debug_cylinder_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_step_inliers_;

    rclcpp::TimerBase::SharedPtr timer_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr gripper_dense_cloud_; 
    
    std::vector<LocalBox> gripper_boxes_; 

    Eigen::Vector4f min_pt_, max_pt_;
    std::vector<geometry_msgs::msg::Pose> all_candidates_;
    std::vector<geometry_msgs::msg::Pose> hit_candidates_;
    std::vector<ScoredGrasp> best_grasps_;
    bool has_best_ = false;
};

} // namespace vision

#endif // VISION_GENERATE_GRASP_POSES_HPP_