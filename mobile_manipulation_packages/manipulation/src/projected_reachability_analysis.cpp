#include <manipulation/ProjectedReachabilityAnalysis.hpp>
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <queue> 
#include <algorithm>
#include <set> 

using namespace std::chrono_literals;

namespace manipulation {

ProjectedReachabilityAnalysis::ProjectedReachabilityAnalysis(const rclcpp::NodeOptions & options)
: Node("projected_reachability_node", options) 
{
    RCLCPP_INFO(this->get_logger(), "Projected Reachability Analysis (All Points) inicializado.");

    this->declare_parameter<double>("path_resolution", 0.05);
    this->declare_parameter<double>("security_distance", 0.2);

    distanceToObstacle_ =  static_cast<float>(this->get_parameter("path_resolution").get_parameter_value().get<double>());
    security_distance = static_cast<float>(this->get_parameter("security_distance").get_parameter_value().get<double>());
    
    decimals = count_decimals(distanceToObstacle_);

    marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("reachability_visualization", 10);
    
    rclcpp::QoS qos(10);
    qos.reliable(); 
    qos.transient_local();
    reachability_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("reachability_zone", qos);
}   

std::vector<std::pair<float, float>> ProjectedReachabilityAnalysis::get_reachable_points(
    const geometry_msgs::msg::Pose& origin, 
    const double& ROBOT_BASE_Z, 
    const double& MAX_REACH_3D, std::vector<std::pair<float, float>>& valid_candidates)
{

    // 1. Cálculo do Raio
    double vertical_dist = std::abs(origin.position.z - ROBOT_BASE_Z);

    if (vertical_dist > MAX_REACH_3D) 
    {
        RCLCPP_WARN(this->get_logger(), "Objeto inalcançável! Altura (%.2f) > Alcance Max (%.3f)", vertical_dist, MAX_REACH_3D);
        return valid_candidates; 
    }

    double radius_2d = std::sqrt(std::pow(MAX_REACH_3D, 2) - std::pow(vertical_dist, 2));
    const float radius_sq = static_cast<float>(radius_2d * radius_2d);
    const float security_distance_squared = security_distance * security_distance;

    RCLCPP_INFO(this->get_logger(), "Calculando pontos acessíveis. Raio 2D: %.4f m", radius_2d);

    // 2. BFS
    std::pair<float,float> origin_pair = {
        round_to_multiple(origin.position.x, distanceToObstacle_, decimals), 
        round_to_multiple(origin.position.y, distanceToObstacle_, decimals)
    };
    
    valid_candidates.push_back(origin_pair);

    std::queue<std::pair<float, float>> q;
    q.push(origin_pair);

    std::set<std::pair<float, float>> visited;
    visited.insert(origin_pair);

    auto offsets = get_offsets(distanceToObstacle_);
    int steps = 0;
    int max_steps = 30000; 

    while(!q.empty())
    {
        steps++;
        if(steps > max_steps) break;
   
        std::pair<float, float> current_pos = q.front();
        q.pop();

        for(int i = 0; i < 8; i++)
        {
            float nx = round_to_multiple(current_pos.first + offsets[i][0], distanceToObstacle_, decimals);
            float ny = round_to_multiple(current_pos.second + offsets[i][1], distanceToObstacle_, decimals);
            std::pair<float, float> neighbor = {nx, ny};

            if(visited.find(neighbor) != visited.end())
            {
                continue;
            }

            float dx = neighbor.first - origin_pair.first;
            float dy = neighbor.second - origin_pair.second;
            float dist_sq = dx*dx + dy*dy;

            if (dist_sq > radius_sq)
            {
                continue;
            } 

            visited.insert(neighbor);
            q.push(neighbor);
            
  
            if(dist_sq >= security_distance_squared)
            {
                valid_candidates.push_back(neighbor);
            }
            
        }
    }

   
    publish_reachability_cloud(valid_candidates);

   
    visualization_msgs::msg::MarkerArray marker_array;
    rclcpp::Time current_time = this->now();

    visualization_msgs::msg::Marker disk;
    disk.header.frame_id = "world"; disk.header.stamp = current_time;
    disk.ns = "reachability"; disk.id = 0; disk.type = visualization_msgs::msg::Marker::CYLINDER;
    disk.action = 0; disk.pose.orientation.w = 1.0;
    disk.pose.position = origin.position; disk.pose.position.z = 0.0;
    disk.scale.x = radius_2d * 2.0; disk.scale.y = radius_2d * 2.0; disk.scale.z = 0.01;
    disk.color.b = 1.0; disk.color.a = 0.2;
    marker_array.markers.push_back(disk);
    marker_publisher_->publish(marker_array);

   
    return valid_candidates;
}

void ProjectedReachabilityAnalysis::publish_reachability_cloud(const std::vector<std::pair<float, float>>& points)
{
    if (points.empty()) return;

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.frame_id = "world";
    cloud.header.stamp = this->now();
    cloud.height = 1;
    cloud.width = points.size();
    cloud.is_dense = true;
    
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(cloud.width);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

    for (const auto& p : points)
    {
        *iter_x = p.first;
        *iter_y = p.second;
        *iter_z = 0.02; 
        ++iter_x; ++iter_y; ++iter_z;
    }

    reachability_cloud_pub_->publish(cloud);
}

std::vector<std::array<float, 3>> ProjectedReachabilityAnalysis::get_offsets(float distanceToObstacle) 
{
    return {
        {-distanceToObstacle, -distanceToObstacle, 0.0}, {distanceToObstacle, -distanceToObstacle, 0.0},
        {distanceToObstacle, distanceToObstacle, 0.0}, {-distanceToObstacle, distanceToObstacle, 0.0}, 
        {-distanceToObstacle, 0.0, 0.0}, {distanceToObstacle, 0.0, 0.0},
        {0.0, distanceToObstacle, 0.0}, {0.0, -distanceToObstacle, 0.0},
    };
}

inline float ProjectedReachabilityAnalysis::round_to_multiple(float value, float multiple, int decimals) 
{
    if (multiple == 0.0) return value; 
    float result = std::round(value / multiple) * multiple;
    float factor = std::pow(10.0, decimals);
    result = std::round(result * factor) / factor;
    return result;
}

int ProjectedReachabilityAnalysis::count_decimals(float number) 
{
    float fractional = std::fabs(number - std::floor(number));
    int decimals = 0;
    const float epsilon = 1e-9; 
    while (fractional > epsilon && decimals < 20) {
        fractional *= 10;
        fractional -= std::floor(fractional);
        decimals++;
    }
    return decimals;
}

} // namespace manipulation

RCLCPP_COMPONENTS_REGISTER_NODE(manipulation::ProjectedReachabilityAnalysis)