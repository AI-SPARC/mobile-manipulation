#ifndef MANIPULATION__PROJECTED_REACHABILITY_ANALYSIS_HPP_
#define MANIPULATION__PROJECTED_REACHABILITY_ANALYSIS_HPP_

#include <memory>
#include <vector>
#include <cmath>
#include <utility>
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace manipulation {

class ProjectedReachabilityAnalysis : public rclcpp::Node
{
public:
    explicit ProjectedReachabilityAnalysis(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ProjectedReachabilityAnalysis() override = default;

    // --- MUDANÇA: Retorna vetor de pontos ---
    std::vector<std::pair<float, float>> get_reachable_points(
        const geometry_msgs::msg::Pose& origin, 
        const double& ROBOT_BASE_Z, 
        const double& MAX_REACH_3D,
        std::vector<std::pair<float, float>>& valid_candidates
    );

private:
    float distanceToObstacle_ = 0.05, security_distance = 0.2;
    int decimals = 0;

    void publish_reachability_cloud(const std::vector<std::pair<float, float>>& points);
    std::vector<std::array<float, 3>> get_offsets(float distanceToObstacle);
    inline float round_to_multiple(float value, float multiple, int decimals);
    int count_decimals(float number);

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr reachability_cloud_pub_;
};

} // namespace manipulation

#endif