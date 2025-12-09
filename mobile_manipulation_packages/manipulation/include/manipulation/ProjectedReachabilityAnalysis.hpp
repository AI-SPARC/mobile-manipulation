#ifndef MANIPULATION__PROJECTED_REACHABILITY_ANALYSIS_HPP_
#define MANIPULATION__PROJECTED_REACHABILITY_ANALYSIS_HPP_

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include <cmath>
#include <unordered_set>
#include <utility>
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose.hpp"

// Header do pacote navigation para acessar SharedObstacleGraph e PairHash
#include "navigation/SharedObstacleGraph.hpp" 

namespace manipulation {

class ProjectedReachabilityAnalysis : public rclcpp::Node
{
public:
    explicit ProjectedReachabilityAnalysis(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ProjectedReachabilityAnalysis() override = default;

    // Retorna {sucesso, raio}. Recebe o NÓ (Provider) e não o mapa direto.
    std::pair<bool, double> calculate_max_2d_radius(
        const geometry_msgs::msg::Pose& pose, 
        const double& ROBOT_BASE_Z, 
        const double& MAX_REACH_3D,
        const std::shared_ptr<navigation::SharedObstacleGraph>& graph_provider_node
    );

private:
    float distanceToObstacle_ = 0.05;
    int decimals = 0;

    // BFS recebe o snapshot (para calcular) E o nó (para verificar validade)
    std::pair<std::pair<float, float>, bool> bfs_to_calculate_possible_pick_points(
        geometry_msgs::msg::Pose origin, 
        const double& radius,
        const std::shared_ptr<const std::unordered_set<std::pair<float, float>, navigation::PairHash>>& current_map_snapshot,
        const std::shared_ptr<navigation::SharedObstacleGraph>& graph_provider_node
    );

    std::vector<std::array<float, 3>> get_offsets(float distanceToObstacle);
    inline float round_to_multiple(float value, float multiple, int decimals);
    int count_decimals(float number);

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
};

} // namespace manipulation

#endif