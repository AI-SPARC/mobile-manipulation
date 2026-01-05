#include "navigation/FleetManagementAStar.hpp"
// PCL includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>

// Eigen & Math
#include <Eigen/Geometry>
#include <cmath>
#include <algorithm>
#include <limits>
#include <queue>

// Component Register
#include "rclcpp_components/register_node_macro.hpp"

namespace navigation {

using namespace std::chrono_literals;

AStar::AStar(const rclcpp::NodeOptions & options)
: Node("a_star", "navigation", options)
{
    // Parâmetros
    this->declare_parameter<double>("path_resolution", 0.05);
    this->declare_parameter<int>("iterations_before_verification", 10);
    this->declare_parameter<double>("max_security_distance", 0.30);
    this->declare_parameter("num_robots", 1);

    distanceToObstacle_ = static_cast<float>(this->get_parameter("path_resolution").as_double());
    iterations_before_verification = this->get_parameter("iterations_before_verification").as_int();
    maxSecurityDistance_ = static_cast<float>(this->get_parameter("max_security_distance").as_double());
    num_robots = this->get_parameter("num_robots").as_int();

    RCLCPP_INFO(this->get_logger(), "AStar initialized. Res: %.3f | SecDist: %.2f", distanceToObstacle_, maxSecurityDistance_);

    decimals = count_decimals(distanceToObstacle_);

    // Subscribers Dinâmicos (Odom e Dest)
    for (int i = 0; i < num_robots; ++i)
    {
        // Odometria
        std::string odom_topic = "/robot_" + std::to_string(i) + "/odom";
        auto odom_cb = [this, i](const nav_msgs::msg::Odometry::SharedPtr msg) {
            this->callback_odom(msg, i);
        };
        odom_subs_.push_back(this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, 10, odom_cb));
        
        RCLCPP_INFO(this->get_logger(), "Subscribed to Robot %d (Odom + Dest)", i);
    }

    // Subscriber do Grafo de Obstáculos (Estilo SharedObstacleGraph)
    // Usa QoS Best Effort para performance com PointCloud
    rclcpp::QoS qos(10);
    qos.best_effort();
    
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/obstacle_graph",
        qos,
        std::bind(&AStar::topic_callback, this, std::placeholders::_1)
    );
}


void AStar::callback_odom(const nav_msgs::msg::Odometry::SharedPtr msg, int robot_id)
{
    std::lock_guard<std::mutex> lock(odom_mutex);
    robot_states_[robot_id].current = msg->pose.pose;
    robot_states_[robot_id].has_current = true;
}


void AStar::topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(map_mutex_);

   
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    
    
    
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y) 
    {
        float x = *iter_x;
        float y = *iter_y;
        
        std::pair<float, float> index = std::make_pair(
            round_to_multiple(x, distanceToObstacle_, decimals),
            round_to_multiple(y, distanceToObstacle_, decimals)
        );
        obstaclesVertices.insert(index);
    }
}


std::vector<std::pair<float, float>> AStar::run_a_star(std::pair<float, float> start_tuple, std::pair<float, float> goal_tuple) 
{
    // Verifica Start
    auto start_search = find_nearest_free_point(start_tuple, 500);
    if (!start_search.second) {
        RCLCPP_WARN(this->get_logger(), "START BLOCKED.");
        return {};
    }
    std::pair<float, float> valid_start = start_search.first;

    // Verifica Goal
    auto goal_search = find_nearest_free_point(goal_tuple, 3000);
    if (!goal_search.second) {
        RCLCPP_WARN(this->get_logger(), "GOAL BLOCKED.");
        return {};
    }
    std::pair<float, float> valid_goal = goal_search.first;

    // Caso trivial
    if (valid_start == valid_goal) return {valid_goal};

    // Tenta Linha Reta primeiro (Greedy)
    auto initial_path = straight_line(valid_start, valid_goal);
    if(!initial_path.empty()) {
        initial_path.insert(initial_path.begin(), valid_start); // Insere início
        initial_path.push_back(valid_goal);
        return initial_path;
    }

    // A* Standard
    struct Node {
        std::pair<float, float> parent;
        float g_score = std::numeric_limits<float>::infinity();
        float f_score = std::numeric_limits<float>::infinity();
        bool closed = false;
    };

    std::unordered_map<std::pair<float, float>, Node, PairHash> nodes;
    auto offsets = get_offsets(distanceToObstacle_); // Cache offsets

    // Heurística Euclidiana
    auto heuristic = [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
        float dx = a.first - b.first;
        float dy = a.second - b.second;
        return std::sqrt(dx*dx + dy*dy);
    };

    // Priority Queue
    struct PairCompare {
        bool operator()(const std::pair<float, std::pair<float, float>>& a, 
                        const std::pair<float, std::pair<float, float>>& b) const {
            return a.first > b.first;
        }
    };
    std::priority_queue<std::pair<float, std::pair<float, float>>, 
                        std::vector<std::pair<float, std::pair<float, float>>>, 
                        PairCompare> open_set;

    // Setup Start
    nodes[valid_start].g_score = 0;
    nodes[valid_start].f_score = heuristic(valid_start, valid_goal);
    open_set.push({nodes[valid_start].f_score, valid_start});

    int iterations = 0;

    while (!open_set.empty()) 
    {
        auto current_pair = open_set.top();
        open_set.pop();
        auto current = current_pair.second;

        if (nodes[current].closed) continue;
        if (current_pair.first > nodes[current].f_score) continue; // Lazy deletion check

        nodes[current].closed = true;

        // Chegou?
        if (current == valid_goal) {
            std::vector<std::pair<float, float>> path;
            auto curr = current;
            while (curr != valid_start) {
                path.push_back(curr);
                curr = nodes[curr].parent;
            }
            path.push_back(valid_start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        // Otimização: Tenta linha reta a cada X iterações
        if(iterations >= iterations_before_verification) {
            iterations = 0;
            auto shortcut = straight_line(current, valid_goal);
            if(!shortcut.empty()) {
                // Reconstrói caminho até current + shortcut
                std::vector<std::pair<float, float>> path;
                auto curr = current;
                while (curr != valid_start) {
                    path.push_back(curr);
                    curr = nodes[curr].parent;
                }
                path.push_back(valid_start);
                std::reverse(path.begin(), path.end());
                path.insert(path.end(), shortcut.begin(), shortcut.end());
                path.push_back(valid_goal);
                return path;
            }
        }

        // Expansão de Vizinhos
        for (const auto& offset : offsets) {
            float nx = round_to_multiple(current.first + offset[0], distanceToObstacle_, decimals);
            float ny = round_to_multiple(current.second + offset[1], distanceToObstacle_, decimals);
            std::pair<float, float> neighbor = {nx, ny};

            // Colisão?
            if (obstaclesVertices.find(neighbor) != obstaclesVertices.end()) continue;

            // Já fechado?
            if (nodes[neighbor].closed) continue;

            float tentative_g = nodes[current].g_score + heuristic(current, neighbor);

            if (tentative_g < nodes[neighbor].g_score) {
                nodes[neighbor].parent = current;
                nodes[neighbor].g_score = tentative_g;
                nodes[neighbor].f_score = tentative_g + heuristic(neighbor, valid_goal);
                open_set.push({nodes[neighbor].f_score, neighbor});
            }
        }
        iterations++;
    }

    return {}; // Falha
}

std::pair<std::pair<float, float>, bool> AStar::find_nearest_free_point(
    std::pair<float, float> origin, int max_steps) 
{
    std::pair<float,float> rounded = {
        round_to_multiple(origin.first, distanceToObstacle_, decimals), 
        round_to_multiple(origin.second, distanceToObstacle_, decimals)
    };
    
    if (obstaclesVertices.find(rounded) == obstaclesVertices.end()) return {origin, true};

    // BFS para sair do obstáculo
    std::queue<std::pair<float, float>> q;
    q.push(rounded);
    std::unordered_set<std::pair<float, float>, PairHash> visited;
    visited.insert(rounded);
    
    auto offsets = get_offsets(distanceToObstacle_);
    int steps = 0;

    while(!q.empty() && steps++ < max_steps) {
        auto curr = q.front(); q.pop();

        if (obstaclesVertices.find(curr) == obstaclesVertices.end()) return {curr, true};

        for(const auto& off : offsets) {
            float nx = round_to_multiple(curr.first + off[0], distanceToObstacle_, decimals);
            float ny = round_to_multiple(curr.second + off[1], distanceToObstacle_, decimals);
            std::pair<float, float> neighbor = {nx, ny};

            if(visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    return {origin, false};
}

std::vector<std::pair<float, float>> AStar::straight_line(std::pair<float, float> start, std::pair<float, float> end)
{
    std::vector<std::pair<float, float>> path;
    float dx = end.first - start.first;
    float dy = end.second - start.second;
    float dist = std::hypot(dx, dy);
    
    if(dist < 1e-4) return {};

    float step = distanceToObstacle_;
    float ux = dx / dist;
    float uy = dy / dist;
    
    // Check collision along ray
    for(float t = step; t < dist; t += step) {
        float x = round_to_multiple(start.first + t*ux, distanceToObstacle_, decimals);
        float y = round_to_multiple(start.second + t*uy, distanceToObstacle_, decimals);
        
        if(obstaclesVertices.count({x,y})) return {}; // Blocked
        path.push_back({x,y});
    }
    return path;
}

std::pair<nav_msgs::msg::Path, nav_msgs::msg::Path> AStar::filter_path(
    std::vector<std::pair<float, float>>& path, 
    std::vector<std::pair<float, float>>& path_raw, 
    const geometry_msgs::msg::Pose& goal_pose)
{
    // Cria mensagens vazias
    nav_msgs::msg::Path p_filtered, p_raw;
    p_filtered.header.frame_id = "world"; p_filtered.header.stamp = this->now();
    p_raw.header.frame_id = "world"; p_raw.header.stamp = this->now();

    if(path.empty()) return {p_filtered, p_raw};

    // Orientação Final Desejada
    Eigen::Quaternionf final_q(
        goal_pose.orientation.w, goal_pose.orientation.x, 
        goal_pose.orientation.y, goal_pose.orientation.z);

    // --- FILTRAGEM (Simplificação de Caminho) ---
    // Raycasting do início para o fim, cortando nós redundantes
    int k = 0;
    while(k < (int)path.size() - 1) {
        bool shortcut = false;
        // Tenta conectar k diretamente com i (do mais longe para o mais perto)
        for(int i = (int)path.size() - 1; i > k + 1; i--) {
            auto segment = straight_line(path[k], path[i]);
            if(!segment.empty() || std::hypot(path[k].first-path[i].first, path[k].second-path[i].second) < distanceToObstacle_) {
                // Atalho válido encontrado! Remove nós intermediários
                path.erase(path.begin() + k + 1, path.begin() + i);
                shortcut = true;
                break;
            }
        }
        k++;
    }

    // --- Conversão para ROS Message (Filtered) ---
    for(size_t i = 0; i < path.size(); ++i) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = p_filtered.header;
        ps.pose.position.x = path[i].first;
        ps.pose.position.y = path[i].second;
        
        // Calcula orientação baseada no próximo ponto
        if(i < path.size() - 1) {
            float dx = path[i+1].first - path[i].first;
            float dy = path[i+1].second - path[i].second;
            if(std::hypot(dx,dy) > 1e-4) {
                Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), Eigen::Vector3f(dx,dy,0));
                ps.pose.orientation.w = q.w(); ps.pose.orientation.x = q.x();
                ps.pose.orientation.y = q.y(); ps.pose.orientation.z = q.z();
            } else ps.pose.orientation.w = 1.0;
        } else {
            ps.pose.orientation = goal_pose.orientation; // Último ponto pega orientação do goal
        }
        p_filtered.poses.push_back(ps);
    }

    // --- Conversão para ROS Message (Raw) ---
    // (Mesma lógica, mas sem o loop de filtragem)
    for(size_t i = 0; i < path_raw.size(); ++i) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = p_raw.header;
        ps.pose.position.x = path_raw[i].first;
        ps.pose.position.y = path_raw[i].second;
        
        if(i < path_raw.size() - 1) {
            float dx = path_raw[i+1].first - path_raw[i].first;
            float dy = path_raw[i+1].second - path_raw[i].second;
            Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), Eigen::Vector3f(dx,dy,0));
            ps.pose.orientation.w = q.w(); ps.pose.orientation.x = q.x();
            ps.pose.orientation.y = q.y(); ps.pose.orientation.z = q.z();
        } else ps.pose.orientation = goal_pose.orientation;
        p_raw.poses.push_back(ps);
    }

    return {p_filtered, p_raw};
}


std::vector<std::array<float, 3>> AStar::get_offsets(float d) {
    return {{-d,-d,0},{d,-d,0},{d,d,0},{-d,d,0},{-d,0,0},{d,0,0},{0,d,0},{0,-d,0}};
}

inline float AStar::round_to_multiple(float value, float multiple, int decimals) {
    if (multiple == 0.0) return value; 
    float result = std::round(value / multiple) * multiple;
    float factor = std::pow(10.0, decimals);
    return std::round(result * factor) / factor;
}

int AStar::count_decimals(float number) {
    float fractional = std::fabs(number - std::floor(number));
    int d = 0;
    while (fractional > 1e-9 && d < 20) { fractional *= 10; fractional -= std::floor(fractional); d++; }
    return d;
}

} // namespace navigation

// REGISTRO DE COMPONENTE
RCLCPP_COMPONENTS_REGISTER_NODE(navigation::AStar)