#ifndef VISION__OBJECT_MAPPING_HPP_
#define VISION__OBJECT_MAPPING_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <mobile_manipulation_interfaces/msg/semantic_pcl.hpp>

// PCL Includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

// Octomap Includes
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/msg/octomap.h>
#include <octomap_msgs/conversions.h>

// TF2
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <mutex>
#include <string>
#include <map>
#include <vector>

namespace vision {



class ObjectMapping : public rclcpp::Node
{
public:
    explicit ObjectMapping(const rclcpp::NodeOptions & options);
    
   
    void ObjectToMap(std::string id);

private:
    struct MappingObjectData
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        geometry_msgs::msg::Pose pose;

        MappingObjectData() : cloud(new pcl::PointCloud<pcl::PointXYZ>) {}
    };
    
    void jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg);

    void publishAccumulatedCloud();
    void publishToPlanningScene();
    void publishSemanticEnvironment(
        const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr & input_msg, 
        const std::string& target_label);

    
    bool areCloudsClose(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target, 
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_env, 
        double threshold);

    
    std::mutex data_mutex_;
    std::string object_to_map_;
    std::map<std::string, MappingObjectData> object_map_;
    
    rclcpp::Time last_motion_time_;
    bool is_robot_stopped_;

    
    double velocity_threshold_;
    double settlement_duration_;
    double voxel_leaf_size_;
    bool publish_octomap_to_moveit_;
    double surrounding_distance_threshold_;

   
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
    rclcpp::Subscription<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr sub_semantic_pcl_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_accumulated_cloud_;
    rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr pub_planning_scene_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_environment_cloud_;
    
    
    rclcpp::Publisher<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr pub_semantic_environment_;
};

} // namespace vision

#endif // VISION__OBJECT_MAPPING_HPP_