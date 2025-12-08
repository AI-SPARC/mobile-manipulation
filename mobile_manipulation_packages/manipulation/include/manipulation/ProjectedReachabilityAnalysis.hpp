#ifndef MANIPULATION__PROJECTED_REACHABILITY_ANALYSIS_HPP_
#define MANIPULATION__PROJECTED_REACHABILITY_ANALYSIS_HPP_

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
namespace manipulation {

class ProjectedReachabilityAnalysis : public rclcpp::Node
{
public:
    explicit ProjectedReachabilityAnalysis(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    
    ~ProjectedReachabilityAnalysis() override = default;

    double calculate_max_2d_radius(const geometry_msgs::msg::Pose& pose, const double& ROBOT_BASE_Z = 0.11, const double& MAX_REACH_3D = 0.9);

private:
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
};

} // namespace manipulation

#endif // MANIPULATION__IS_GRIPPER_HOLDING_HPP_