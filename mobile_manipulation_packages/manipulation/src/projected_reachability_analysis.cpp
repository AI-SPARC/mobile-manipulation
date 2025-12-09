#include <manipulation/ProjectedReachabilityAnalysis.hpp>
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <queue> 

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

std::pair<bool, double> ProjectedReachabilityAnalysis::calculate_max_2d_radius(
    const geometry_msgs::msg::Pose& pose, 
    const double& ROBOT_BASE_Z, 
    const double& MAX_REACH_3D, 
    const std::shared_ptr<navigation::SharedObstacleGraph>& graph_provider_node)
{
    // 1. Pega o Snapshot Inicial (Thread-Safe, muito rápido)
    auto initial_map_ptr = graph_provider_node->get_current_map();

    if (!initial_map_ptr) {
        RCLCPP_WARN(this->get_logger(), "Ponteiro do mapa é nulo. Abortando.");
        return {false, 0.0};
    }

    // 2. Validação Geométrica
    double vertical_dist = std::abs(pose.position.z - ROBOT_BASE_Z);
    if (vertical_dist > MAX_REACH_3D) {
        RCLCPP_WARN(this->get_logger(), "Objeto inalcançável verticalmente.");
        return {false, 0.0}; 
    }

    double radius_2d = std::sqrt(std::pow(MAX_REACH_3D, 2) - std::pow(vertical_dist, 2));
    RCLCPP_INFO(this->get_logger(), "Iniciando BFS com Raio 2D: %.4f m", radius_2d);

    // 3. Executa BFS com monitoramento de mudança de mapa
    auto bfs_result = bfs_to_calculate_possible_pick_points(pose, radius_2d, initial_map_ptr, graph_provider_node);
    
    // Se bfs_result.second for false, pode ser colisão OU mapa mudou.
    if(bfs_result.second) 
    {
        RCLCPP_INFO(this->get_logger(), "Ponto válido encontrado: (%.2f, %.2f)", bfs_result.first.first, bfs_result.first.second);
        
        // --- Visualização (Apenas sucesso) ---
        visualization_msgs::msg::MarkerArray marker_array;
        rclcpp::Time current_time = this->now();

        auto create_base_marker = [&](int id, int type) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "world"; m.header.stamp = current_time;
            m.ns = "reachability"; m.id = id; m.type = type;
            m.action = 0; m.pose.orientation.w = 1.0; m.color.a = 1.0;
            return m;
        };

        // Disco
        auto disk = create_base_marker(0, visualization_msgs::msg::Marker::CYLINDER);
        disk.pose.position = pose.position; disk.pose.position.z = 0.0;
        disk.scale.x = radius_2d * 2.0; disk.scale.y = radius_2d * 2.0; disk.scale.z = 0.01;
        disk.color.b = 1.0; disk.color.a = 0.3;
        marker_array.markers.push_back(disk);

        // Alvo
        auto target = create_base_marker(1, visualization_msgs::msg::Marker::CUBE);
        target.pose = pose;
        target.scale.x = 0.05; target.scale.y = 0.05; target.scale.z = 0.05;
        target.color.r = 1.0;
        marker_array.markers.push_back(target);

        marker_publisher_->publish(marker_array);

        return {true, radius_2d};
    } 
    else 
    {
        RCLCPP_WARN(this->get_logger(), "BFS Falhou: Mapa mudou ou sem espaço livre.");
        return {false, 0.0};
    }
}

// ... (get_offsets, round_to_multiple, count_decimals permanecem iguais) ...
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

std::pair<std::pair<float, float>, bool> ProjectedReachabilityAnalysis::bfs_to_calculate_possible_pick_points(
    geometry_msgs::msg::Pose origin, 
    const double& radius, 
    const std::shared_ptr<const std::unordered_set<std::pair<float, float>, navigation::PairHash>>& current_map_snapshot,
    const std::shared_ptr<navigation::SharedObstacleGraph>& graph_provider_node)
{
    if (!current_map_snapshot) return {{0,0}, false};
    
    std::pair<float,float> nearest_rounded = {
        round_to_multiple(origin.position.x, distanceToObstacle_, decimals), 
        round_to_multiple(origin.position.y, distanceToObstacle_, decimals)
    };
    
    // Check inicial no snapshot
    if (current_map_snapshot->find(nearest_rounded) == current_map_snapshot->end()) {
        return {nearest_rounded, true};
    }

    struct SearchNode {
        float dist;
        std::pair<float, float> pos;
        bool operator>(const SearchNode& other) const { return dist > other.dist; }
    };

    std::priority_queue<SearchNode, std::vector<SearchNode>, std::greater<SearchNode>> pq;
    pq.push({0.0f, nearest_rounded});

    std::unordered_set<std::pair<float, float>, navigation::PairHash> visited;
    visited.insert(nearest_rounded);

    auto offsets = get_offsets(distanceToObstacle_);
    int steps = 0;
    int max_steps = 2000; // Aumentei um pouco para cobrir raio 0.9m com res 0.05
    
    std::pair<float, float> origin_pair = {origin.position.x, origin.position.y};

    while(!pq.empty())
    {
        steps++;
        if(steps > max_steps) break;

        // --- CHECK DE PREEMPÇÃO EM TEMPO REAL ---
        // A cada 10 passos, verifica se o nó tem um mapa novo
        if (steps % 10 == 0) 
        {
            // Pega o ponteiro ATUAL do nó (nanossegundos)
            auto live_map_ptr = graph_provider_node->get_current_map();
            
            // Se o endereço mudou, significa que load/inflate rodou de novo
            if (live_map_ptr != current_map_snapshot) 
            {
                RCLCPP_WARN(this->get_logger(), "BFS PREEMPTADO: Mapa atualizado durante execução.");
                return {origin_pair, false};
            }
        }
        // ----------------------------------------

        auto current_node = pq.top();
        pq.pop();
        
        // Verifica colisão no SNAPSHOT (Seguro)
        if (current_map_snapshot->find(current_node.pos) == current_map_snapshot->end())
        {
            return {current_node.pos, true};
        }

        for(int i = 0; i < 8; i++)
        {
            float nx = round_to_multiple(current_node.pos.first + offsets[i][0], distanceToObstacle_, decimals);
            float ny = round_to_multiple(current_node.pos.second + offsets[i][1], distanceToObstacle_, decimals);
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