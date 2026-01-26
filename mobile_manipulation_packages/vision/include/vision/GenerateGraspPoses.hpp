#ifndef VISION_GENERATE_GRASP_POSES_HPP_
#define VISION_GENERATE_GRASP_POSES_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/octree/octree_search.h>

namespace vision 
{

struct StepAnalysis
{
    bool valid;
    int point_count;
    float angle_to_normal_deg; 
    Eigen::Vector3f normal_vector; 
    float symmetry_score;
    float curvature; 
    Eigen::Vector3f center;
};

struct ScoredGrasp
{
    geometry_msgs::msg::Pose pose_center;
    geometry_msgs::msg::Pose pose_finger1;
    geometry_msgs::msg::Pose pose_finger2;
    
    double total_score;
    double entry_angle;
    double exit_angle;
    double entry_planarity;
    double exit_planarity;
    Eigen::Vector3f entry_normal;

    Eigen::Vector3f raw_ray_dir;
    Eigen::Vector3f raw_p_f1;
    Eigen::Vector3f raw_p_f2;
    Eigen::Vector3f debug_entry_pt;
    Eigen::Vector3f debug_exit_pt;
    pcl::PointCloud<pcl::PointXYZ> debug_inliers; 
};

class GenerateGraspPoses : public rclcpp::Node
{
public:
    explicit GenerateGraspPoses(const rclcpp::NodeOptions & options);

    geometry_msgs::msg::PoseArray processCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr target, pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment);

private:
    
    bool use_pcd_file;
    std::string pcd_path_;
    std::string gripper_mesh_path_;
    float gripper_mesh_scale_;

    float mesh_offset_x_;
    float mesh_offset_y_;
    float mesh_offset_z_;

    float mesh_rot_roll_;
    float mesh_rot_pitch_;
    float mesh_rot_yaw_;

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
    float weight_planarity_;

    bool mean_filter;
    float mean_filter_k_;

    int num_best_grasps_;
    int total_orientations_;
    float rotation_step_deg_;

    
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rays_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_bbox_; 
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_inliers_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_gripper_model_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_grasps_cloud_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_input_cloud_;

 
    pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud_;
    std::vector<geometry_msgs::msg::Pose> all_candidates_;
    std::vector<geometry_msgs::msg::Pose> hit_candidates_;
    std::vector<ScoredGrasp> best_grasps_;
    bool has_best_ = false;
    Eigen::Vector4f min_pt_, max_pt_;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> gripper_octree_{0.002f};
    pcl::PointCloud<pcl::PointXYZ>::Ptr gripper_dense_cloud_;

    void loadAndProcess(const std::string& path);
    
    std::vector<geometry_msgs::msg::Pose> generateMultiOrientedRays(
        const Eigen::Vector4f& min, const Eigen::Vector4f& max, float res, 
        const std::vector<Eigen::Matrix3f>& rotation_matrices);

    StepAnalysis analyzeLocalCylinder(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const Eigen::Vector3f& center,
        const Eigen::Vector3f& ray_dir,
        float radius,
        float height);

    void publishGripperModel();
    void loadGripperAsOctree();
    bool check_collision(const ScoredGrasp& grasp, pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment);
    Eigen::Quaternionf findBestOrientation(const Eigen::Vector3f& p_f1, const Eigen::Vector3f& p_f2);
    geometry_msgs::msg::PoseArray evaluateGrasps(pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment);
    void timerCallback();
    void publishBest();

    struct VoxelBucket 
    {
        Eigen::Vector3f center;
        std::vector<pcl::PointXYZ> points;
    };
};

} // namespace vision

#endif // VISION_GENERATE_GRASP_POSES_HPP_