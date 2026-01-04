#ifndef MANIPULATION_IK_VALIDATOR_HPP_
#define MANIPULATION_IK_VALIDATOR_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

// MoveIt
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

#include <vector>
#include <optional>
#include <string>
#include <atomic>
#include <memory>
#include <tuple> // Necessário para std::tuple

// Forward declaration
namespace navigation { class SharedObstacleGraph; }

namespace manipulation {

class IKValidator : public rclcpp::Node
{
public:
    explicit IKValidator(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~IKValidator() = default;

    std::optional<std::tuple<float, float, float>> find_best_base_position(
        const std::vector<std::tuple<float, float, float>>& robot_positions, 
        const geometry_msgs::msg::Pose& target_pose_global,
        bool seed_mode,
        const std::shared_ptr<navigation::SharedObstacleGraph>& graph_provider_node,
        std::string authorized_collision
    );

    bool is_still_reachable(const std::shared_ptr<navigation::SharedObstacleGraph>& graph_provider_node);

private:
    void delayed_init();
    void publish_viable_ik_points(const std::vector<std::tuple<float, float, float>>& results);

    // ROS
    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    
    // MoveIt
    std::string group_name_;
    std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
    moveit::core::RobotModelConstPtr robot_model_;
    planning_scene_monitor::PlanningSceneMonitorPtr psm_;

    std::atomic<bool> initialized_{false};
    std::string virtual_joint_name_;

    std::optional<std::tuple<float, float, float>> selected_ik_position;
    std::string last_authorized_collision;
    geometry_msgs::msg::Pose last_target;
};

} // namespace manipulation

#endif // MANIPULATION_IK_VALIDATOR_HPP_