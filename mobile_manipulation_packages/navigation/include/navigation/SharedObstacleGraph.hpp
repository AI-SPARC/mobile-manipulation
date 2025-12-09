#ifndef SHARED_OBSTACLE_GRAPH_NODE_HPP_
#define SHARED_OBSTACLE_GRAPH_NODE_HPP_

#include <memory>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <functional> // Para std::hash

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace navigation {

// --- DEFINIÇÃO DE TIPOS ---

// Hash para o par de floats
struct PairHash {
    std::size_t operator()(const std::pair<float, float>& p) const {
        auto h1 = std::hash<float>{}(p.first);
        auto h2 = std::hash<float>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};


// --- CLASSE DO NÓ ---

class SharedObstacleGraph : public rclcpp::Node
{
public:
    explicit SharedObstacleGraph(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~SharedObstacleGraph() override = default;

    // --- MÉTODO LEITOR (Zero-Copy) ---
    // Retorna o ponteiro direto para o Set.
    // O ponteiro é const para garantir que quem lê não estrague os dados.
    std::shared_ptr<const std::unordered_set<std::pair<float, float>, PairHash>> get_current_map() const;

private:

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
    
    // O Ponteiro Ativo (Aponta para o Set atual)
    std::shared_ptr<std::unordered_set<std::pair<float, float>, PairHash>> current_map_;
    
    // Mutex para proteger APENAS a troca do ponteiro
    mutable std::mutex map_mutex_;

    double resolution_ = 0.05;
    int decimals = 0;

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    inline float round_to_multiple(float value, float multiple, int decimals);
    int count_decimals(float number);
};

} // namespace navigation

#endif // MANIPULATION__OBSTACLE_MAP_NODE_HPP_