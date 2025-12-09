#include "navigation/SharedObstacleGraph.hpp"

// PCL includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "rclcpp_components/register_node_macro.hpp"

namespace navigation {

SharedObstacleGraph::SharedObstacleGraph(const rclcpp::NodeOptions & options)
: Node("obstacle_map_node", options)
{
    RCLCPP_INFO(this->get_logger(), "Obstacle Map Node inicializado (Raw Set Sharing).");

    this->declare_parameter<double>("path_resolution", 0.05);
    resolution_ = this->get_parameter("path_resolution").as_double();

    // Inicializa o mapa vazio para não retornar nullptr no início
    current_map_ = std::make_shared<std::unordered_set<std::pair<float, float>, PairHash>>();

    rclcpp::QoS qos(10);
    qos.best_effort(); // Melhor para sensores rápidos

    point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/obstacle_graph",
        qos,
        std::bind(&SharedObstacleGraph::point_cloud_callback, this, std::placeholders::_1)
    );

    decimals = count_decimals(resolution_);
}

// Método chamado pelo ServerNode/ReachabilityNode
std::shared_ptr<const std::unordered_set<std::pair<float, float>, PairHash>> SharedObstacleGraph::get_current_map() const
{
    std::lock_guard<std::mutex> lock(map_mutex_);

    return current_map_;
}

void SharedObstacleGraph::point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    if (cloud.empty()) 
    {
        return;
    }
    
    // 1. Cria o NOVO container (Back Buffer)
    auto new_map = std::make_shared<std::unordered_set<std::pair<float, float>, PairHash>>();
    
    // Otimização de memória
    new_map->reserve(cloud.size());

  
    // 2. Preenche o novo container
    for (const auto& pt : cloud.points)
    {

        std::pair<float, float> index = std::make_pair(
            round_to_multiple(pt.x, resolution_, decimals),
            round_to_multiple(pt.y, resolution_, decimals)
        );

        new_map->insert(index);
    }

    // 3. SWAP ATÔMICO (Troca o ponteiro antigo pelo novo)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        current_map_ = new_map;
    }
    
    // O mapa antigo é destruído automaticamente quando o último leitor soltá-lo.
}

inline float SharedObstacleGraph::round_to_multiple(float value, float multiple, int decimals) 
{
    if (multiple == 0.0) return value; 
    float result = std::round(value / multiple) * multiple;
    float factor = std::pow(10.0, decimals);
    result = std::round(result * factor) / factor;
    return result;
}

int SharedObstacleGraph::count_decimals(float number) 
{
    float fractional = std::fabs(number - std::floor(number));
    int decimals = 0;
    const float epsilon = 1e-9; 

    while (fractional > epsilon && decimals < 20) 
    {
        fractional *= 10;
        fractional -= std::floor(fractional);
        decimals++;
    }
    return decimals;
}

} // namespace navigation

RCLCPP_COMPONENTS_REGISTER_NODE(navigation::SharedObstacleGraph)