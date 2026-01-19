#ifndef VISION__OBJECT_MAPPING_HPP_
#define VISION__OBJECT_MAPPING_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <mobile_manipulation_interfaces/msg/semantic_pcl.hpp>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h> 
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>


#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace vision {

class ObjectMapping : public rclcpp::Node
{
public:
    explicit ObjectMapping(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    virtual ~ObjectMapping() = default;

    void ObjectToMap(std::string id);

private:
    
    struct ObjectData
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        geometry_msgs::msg::Pose pose;

        ObjectData() : cloud(new pcl::PointCloud<pcl::PointXYZ>) {}
    };
  
    void semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg);
    void jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    
    void publishAccumulatedCloud();

    rclcpp::Subscription<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr sub_semantic_pcl_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_accumulated_cloud_;

    
    std::unordered_map<std::string, ObjectData> object_map_;
    std::mutex data_mutex_;

    bool is_robot_stopped_;
    rclcpp::Time last_motion_time_;
    std::string object_to_map_;
    
    double velocity_threshold_;     
    double settlement_duration_;    
    double voxel_leaf_size_;
};

} // namespace vision

#endif // VISION__OBJECT_MAPPING_HPP_