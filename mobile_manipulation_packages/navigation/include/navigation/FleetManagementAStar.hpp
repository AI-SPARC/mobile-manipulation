#ifndef NAVIGATION__A_STAR_COMPONENT_HPP_
#define NAVIGATION__A_STAR_COMPONENT_HPP_

#include <vector>
#include <map>
#include <tuple>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <utility>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

// Dependência Externa (Seu Shared Graph)
#include "navigation/SharedObstacleGraph.hpp" 

namespace navigation {



struct RobotState {
    geometry_msgs::msg::Pose current;
    geometry_msgs::msg::Pose target;
    bool has_current = false;
    bool has_target = false;
};

class AStar : public rclcpp::Node 
{
public:
    explicit AStar(const rclcpp::NodeOptions & options);

private:
    // --- Membros Principais ---
    
    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_nav_path_;

    // Subscribers
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> odom_subs_;

    std::unordered_map<int, RobotState> robot_states_;


    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    
    // Mapa de Obstáculos Interno (Cópia Local Otimizada)
    std::unordered_set<std::pair<float, float>, PairHash> obstaclesVertices;

    // Mutexes
    std::mutex odom_mutex;
    std::mutex map_mutex_;

    // Parâmetros
    float distanceToObstacle_;
    float maxSecurityDistance_;
    int iterations_before_verification;
    int num_robots;
    int decimals;

    // --- Métodos de Callbacks ---
    void callback_odom(const nav_msgs::msg::Odometry::SharedPtr msg, int robot_id);
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // --- Métodos Algorítmicos (A*) ---
    std::vector<std::pair<float, float>> run_a_star(
        std::pair<float, float> start_tuple, 
        std::pair<float, float> goal_tuple);

    std::pair<std::pair<float, float>, bool> find_nearest_free_point(
        std::pair<float, float> origin, 
        int max_steps);

    std::vector<std::pair<float, float>> straight_line(
        std::pair<float, float> start_tuple, 
        std::pair<float, float> goal_tuple);

    std::pair<nav_msgs::msg::Path, nav_msgs::msg::Path> filter_path(
        std::vector<std::pair<float, float>>& path, 
        std::vector<std::pair<float, float>>& path_without_filter, 
        const geometry_msgs::msg::Pose& goal_pose);

    // --- Helpers ---
    std::vector<std::array<float, 3>> get_offsets(float distanceToObstacle);
    
    inline float round_to_multiple(float value, float multiple, int decimals);
    int count_decimals(float number);
};

} // namespace navigation

#endif  // NAVIGATION__A_STAR_COMPONENT_HPP_