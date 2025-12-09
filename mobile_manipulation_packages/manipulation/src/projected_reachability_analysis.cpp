#include <manipulation/ProjectedReachabilityAnalysis.hpp>
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <queue> // Necessário para priority_queue

using namespace std::chrono_literals;

namespace manipulation {

ProjectedReachabilityAnalysis::ProjectedReachabilityAnalysis(const rclcpp::NodeOptions & options)
: Node("projected_reachability_node", options) 
{
    RCLCPP_INFO(this->get_logger(), "Projected Reachability Analysis inicializado (Composable).");

    this->declare_parameter<double>("path_resolution", 0.05);

    distanceToObstacle_ =  static_cast<float>(this->get_parameter("path_resolution").get_parameter_value().get<double>());

    decimals = count_decimals(distanceToObstacle_);

    marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("reachability_visualization", 10);
}   

double ProjectedReachabilityAnalysis::calculate_max_2d_radius(
    const geometry_msgs::msg::Pose& pose, 
    const double& ROBOT_BASE_Z, 
    const double& MAX_REACH_3D, 
    const std::shared_ptr<const std::unordered_set<std::pair<float, float>, navigation::PairHash>>& obstaclesVertices)
{
    // Cateto vertical.
    double vertical_dist = std::abs(pose.position.z - ROBOT_BASE_Z);

    if (vertical_dist > MAX_REACH_3D) 
    {
        RCLCPP_WARN(this->get_logger(), "Objeto inalcançável! Altura relativa (%.2f) > Alcance Max (%.3f)", vertical_dist, MAX_REACH_3D);
        return 0.0; 
    }

 
    double radius_2d = std::sqrt(std::pow(MAX_REACH_3D, 2) - std::pow(vertical_dist, 2));

    RCLCPP_INFO(this->get_logger(), "Raio 2D no chão: %.4f m (Centro X: %.2f, Y: %.2f)", 
                radius_2d, pose.position.x, pose.position.y);


    auto bfs_result = bfs_to_calculate_possible_pick_points(pose, radius_2d, obstaclesVertices);
    
    if(bfs_result.second) 
    {
        RCLCPP_INFO(this->get_logger(), "Ponto válido encontrado em: (%.2f, %.2f)", bfs_result.first.first, bfs_result.first.second);
    } 
    else 
    {
        RCLCPP_WARN(this->get_logger(), "Nenhum ponto válido encontrado livre de colisão!");
    }

    visualization_msgs::msg::MarkerArray marker_array;
    rclcpp::Time current_time = this->now();

    auto create_base_marker = [&](int id, int type, std::string ns) 
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "world"; 
        m.header.stamp = current_time;
        m.ns = ns;
        m.id = id;
        m.type = type;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.orientation.w = 1.0; 
        m.color.a = 1.0; 
        return m;
    };

    // Marcador do Disco
    visualization_msgs::msg::Marker disk_marker = create_base_marker(0, visualization_msgs::msg::Marker::CYLINDER, "reach_zone");
    disk_marker.pose.position.x = pose.position.x;
    disk_marker.pose.position.y = pose.position.y;
    disk_marker.pose.position.z = 0.0; 
    disk_marker.scale.x = radius_2d * 2.0; 
    disk_marker.scale.y = radius_2d * 2.0; 
    disk_marker.scale.z = 0.015;            
    disk_marker.color.a = 0.3; 
    disk_marker.color.b = 1.0; 
    marker_array.markers.push_back(disk_marker);

    // Marcador do Cubo Alvo
    visualization_msgs::msg::Marker target_cube = create_base_marker(1, visualization_msgs::msg::Marker::CUBE, "target_obj");
    target_cube.pose = pose; 
    target_cube.scale.x = 0.05; 
    target_cube.scale.y = 0.05;
    target_cube.scale.z = 0.05;
    target_cube.color.r = 1.0; 
    marker_array.markers.push_back(target_cube);


    marker_publisher_->publish(marker_array);

    return radius_2d;
}

std::vector<std::array<float, 3>> ProjectedReachabilityAnalysis::get_offsets(float distanceToObstacle) 
{
    return {
        {-distanceToObstacle, -distanceToObstacle, 0.0},
        {distanceToObstacle, -distanceToObstacle, 0.0},
        {distanceToObstacle, distanceToObstacle, 0.0},
        {-distanceToObstacle, distanceToObstacle, 0.0}, 
        {-distanceToObstacle, 0.0, 0.0},
        {distanceToObstacle, 0.0, 0.0},
        {0.0, distanceToObstacle, 0.0},
        {0.0, -distanceToObstacle, 0.0},
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

std::pair<std::pair<float, float>, bool> ProjectedReachabilityAnalysis::bfs_to_calculate_possible_pick_points(
    geometry_msgs::msg::Pose origin, 
    const double& radius, 
    const std::shared_ptr<const std::unordered_set<std::pair<float, float>, navigation::PairHash>>& obstaclesVertices)
{

    if (!obstaclesVertices)
    {
        RCLCPP_ERROR(this->get_logger(), "O ponteiro do grafo de obstáculos é NULO!");
        return {{origin.position.x, origin.position.y}, false};
    }
    
    std::pair<float,float> nearest_rounded = std::make_pair(
        round_to_multiple(origin.position.x, distanceToObstacle_, decimals), 
        round_to_multiple(origin.position.y, distanceToObstacle_, decimals)
    );
    

    if (obstaclesVertices->find(nearest_rounded) == obstaclesVertices->end()) 
    {
        return {nearest_rounded, true};
    }

    struct SearchNode {
        float dist;
        std::pair<float, float> pos;
        
        bool operator>(const SearchNode& other) const {
            return dist > other.dist;
        }
    };

    std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<SearchNode>> pq;

    pq.push({0.0f, nearest_rounded});

    std::unordered_set<std::pair<float, float>, navigation::PairHash> visited;
    visited.insert(nearest_rounded);

    auto offsets = get_offsets(distanceToObstacle_);
    int steps = 0;
    int max_steps = 500; 

    std::pair<float, float> origin_pair = {origin.position.x, origin.position.y};

    while(!pq.empty())
    {
        if(steps++ > max_steps) break;

        auto current_node = pq.top();
        pq.pop();
        
        std::pair<float, float> current_pos = current_node.pos;

            if (obstaclesVertices->find(current_pos) == obstaclesVertices->end())
        {
            return {current_pos, true};
        }

        for(int i = 0; i < 8; i++)
        {
            float nx = round_to_multiple(current_pos.first + offsets[i][0], distanceToObstacle_, decimals);
            float ny = round_to_multiple(current_pos.second + offsets[i][1], distanceToObstacle_, decimals);
            std::pair<float, float> neighbor = {nx, ny};

            if(visited.find(neighbor) != visited.end()) continue;

            visited.insert(neighbor);

            float dist_from_origin = std::hypot(neighbor.first - origin_pair.first, neighbor.second - origin_pair.second);
            
            if (dist_from_origin <= radius) {
                pq.push({dist_from_origin, neighbor});
            }
        }
    }
    
    return {origin_pair, false}; 
}

} // namespace manipulation

RCLCPP_COMPONENTS_REGISTER_NODE(manipulation::ProjectedReachabilityAnalysis)