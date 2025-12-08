/**
 * @file a_star.cpp
 * @brief Implementação do nó de Planejamento de Caminho Global usando A*.
 */

#include <string>
#include <random>
#include <algorithm>
#include <geometry_msgs/msg/point.hpp>
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <map>
#include <stack>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <climits>
#include <iomanip>
#include <thread>
#include <queue>
#include <tuple>
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>                       
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/path.hpp>
#include <cmath>
#include <cstring>
#include <utility> 
#include <iomanip>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include <filesystem>
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include <utility>
#include <string>
#include <filesystem>
#include "rclcpp_action/rclcpp_action.hpp"
#include "mobile_manipulation_interfaces/action/path.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp> 
#include <mutex>
#include "geometry_msgs/msg/pose_array.hpp"
#include <Eigen/Geometry> 

using namespace std::chrono_literals;

namespace std 
{
    // DOC-START: hash_tuple_float
    // Especialização de hash para tuplas de float (usado em mapas não ordenados).
    template <>
    struct hash<std::tuple<float, float, float>> 
    {
        size_t operator()(const std::tuple<float, float, float>& t) const 
        {
            size_t h1 = hash<float>()(std::get<0>(t));
            size_t h2 = hash<float>()(std::get<1>(t));
            size_t h3 = hash<float>()(std::get<2>(t));
            
            // Combinação de hashes estilo boost::hash_combine
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
    // DOC-END: hash_tuple_float
}

namespace std {
    // DOC-START: hash_tuple_pair
    // Especialização de hash para tupla contendo par de inteiros e bool.
    template<>
    struct hash<std::tuple<std::pair<int, int>, bool>> {
        size_t operator()(const std::tuple<std::pair<int, int>, bool>& t) const {
            const auto& p = std::get<0>(t);
            bool b = std::get<1>(t);
            size_t h1 = std::hash<int>{}(p.first);
            size_t h2 = std::hash<int>{}(p.second);
            size_t h3 = std::hash<bool>{}(b);
            size_t seed = h1;
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
    // DOC-END: hash_tuple_pair
}

template <typename T1, typename T2>
struct pair_hash {
    std::size_t operator ()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1); 
    }
};

template<typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, const std::tuple<T1, T2, T3>& t) {
    os << "(" << std::get<0>(t) << ", " 
       << std::get<1>(t) << ", " 
       << std::get<2>(t) << ")";
    return os;
}

class AStar : public rclcpp::Node 
{
    
public:
    // DOC-START: AStar_Constructor
    // Construtor: Inicializa parâmetros, susbcribers, publishers e Action Server.
    AStar()
     : Node("a_star")
    {
        // Declaração de parâmetros configuráveis via ROS 2
        this->declare_parameter<double>("path_resolution", 0.05);
        this->declare_parameter<double>("security_distance", 0.50);
        this->declare_parameter<int>("iterations_before_verification", 10);

        // Conversão e armazenamento dos parâmetros
        distanceToObstacle_ =  static_cast<float>(this->get_parameter("path_resolution").get_parameter_value().get<double>());
        security_distance = static_cast<float>(this->get_parameter("security_distance").get_parameter_value().get<double>());
        iterations_before_verification = this->get_parameter("iterations_before_verification").get_parameter_value().get<int>();

        RCLCPP_INFO(this->get_logger(), "Resolução do caminho: %f", distanceToObstacle_);
        RCLCPP_INFO(this->get_logger(), "Iterações antes de verificar linha reta: %d", iterations_before_verification);

        // Criação do Action Server para receber requisições de caminho
        this->action_server_ = rclcpp_action::create_server<mobile_manipulation_interfaces::action::Path>(
            this, 
            "path",
            std::bind(&AStar::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&AStar::handle_cancel, this, std::placeholders::_1),
            std::bind(&AStar::handle_accepted, this, std::placeholders::_1));

        // Calcula precisão decimal para arredondamento de coordenadas (evita erros de float)
        decimals = count_decimals(distanceToObstacle_);

        publisher_nav_path_ = this->create_publisher<nav_msgs::msg::Path>("visualize_path", 10);

        // Subscriber de odometria para saber onde o robô está
        subscription_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&AStar::callback_odom, this, std::placeholders::_1));

        // Subscriber de mapa de obstáculos (nuvem de pontos)
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/obstacles_vertices",
            10,
            std::bind(&AStar::topic_callback, this, std::placeholders::_1)
        );
    }
    // DOC-END: AStar_Constructor

private:

    // DOC-START: Structs
    struct Vertex {
        int key;
        float x, y, z;
    };

    struct VertexDijkstra {
        float x, y;
        float orientation_x, orientation_y, orientation_z;
        float orientation_w;
    };

    struct Destinos {
        float x, y, z;
        float orientation_x, orientation_y, orientation_z;
        float orientation_w;
    };

    struct Edge {
        int v1, v2;
    };

    struct PairHash {
        std::size_t operator()(const std::pair<float, float>& p) const {
            auto h1 = std::hash<float>{}(p.first);
            auto h2 = std::hash<float>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };
    // DOC-END: Structs

    
    //Publishers.
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr publisher_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_nav_path_;

    //Subscriptions.
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_odom_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

    //Action server.
    rclcpp_action::Server<mobile_manipulation_interfaces::action::Path>::SharedPtr action_server_;
    std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Path>> active_goal_handle_;

    size_t i_ = 0; 

    // Mutexes para proteger acesso concorrente (ROS Callbacks vs Action Thread)
    std::mutex map_mutex_;
    std::mutex goal_mutex_;
    std::mutex odom_mutex;

    // Mapa de obstáculos usando Set para busca rápida O(1)
    std::unordered_set<std::pair<float, float>, PairHash> obstaclesVertices;

    std::string yaml_file;

    float pose_x_ = 0.0, pose_y_ = 0.0, pose_z_ = 0.0;
    float distanceToObstacle_, security_distance = 0.5;
    int decimals = 0, iterations_before_verification = 10;

    // DOC-START: round_to_multiple
    // Arredonda um valor float para o múltiplo mais próximo da resolução do grid.
    // Essencial para discretizar o espaço contínuo em nós de grafo.
    inline float round_to_multiple(float value, float multiple, int decimals) 
    {
        if (multiple == 0.0) return value; 
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0, decimals);
        result = std::round(result * factor) / factor;
        return result;
    }
    // DOC-END: round_to_multiple
    
    // DOC-START: count_decimals
    // Conta quantas casas decimais são necessárias baseada na resolução do grid.
    int count_decimals(float number) 
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
    // DOC-END: count_decimals

    // DOC-START: get_offsets
    // Retorna os deslocamentos (x, y) para os 8 vizinhos (N, S, L, O, NE, NO, SE, SO).
    std::vector<std::array<float, 3>> get_offsets(float distanceToObstacle) {
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
    // DOC-END: get_offsets


    // DOC-START: find_nearest_free_point
    // Se o ponto de início ou fim estiver dentro de um obstáculo (devido a erro de sensor ou dilatação),
    // esta função busca em espiral (BFS) o ponto livre mais próximo para começar/terminar o planejamento.
    std::pair<std::pair<float, float>, bool> find_nearest_free_point(
        std::pair<float, float> origin, 
        int max_steps) 
    {
        // Arredonda para o grid
        std::pair<float,float> nearest_rounded = std::make_pair(round_to_multiple(std::get<0>(origin), distanceToObstacle_, decimals), 
        round_to_multiple(std::get<1>(origin), distanceToObstacle_, decimals));
        
        // Se já está livre, retorna
        if (obstaclesVertices.find(nearest_rounded) == obstaclesVertices.end()) 
        {
            return {origin, true};
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

        std::unordered_set<std::pair<float, float>, PairHash> visited;
        visited.insert(nearest_rounded);

        auto offsets = get_offsets(distanceToObstacle_);
        int steps = 0;

        while(!pq.empty())
        {
            if(steps++ > max_steps) break;

            auto current_node = pq.top();
            pq.pop();
            
            std::pair<float, float> current_pos = current_node.pos;

            // Achou ponto livre
            if (obstaclesVertices.find(current_pos) == obstaclesVertices.end())
            {
                return {current_pos, true};
            }

            // Expande vizinhos
            for(int i = 0; i < 8; i++)
            {
                float nx = round_to_multiple(current_pos.first + offsets[i][0], distanceToObstacle_, decimals);
                float ny = round_to_multiple(current_pos.second + offsets[i][1], distanceToObstacle_, decimals);
                std::pair<float, float> neighbor = {nx, ny};

                if(visited.find(neighbor) != visited.end()) continue;

                visited.insert(neighbor);

                float dist_from_origin = std::hypot(neighbor.first - origin.first, neighbor.second - origin.second);
                
                pq.push({dist_from_origin, neighbor});
            }
        }
        return {origin, false}; // Falha
    }
    // DOC-END: find_nearest_free_point
   
    // DOC-START: run_a_star
    // O algoritmo A* principal.
    // 1. Verifica validade de Start/Goal.
    // 2. Tenta conectar com linha reta (otimização).
    // 3. Se falhar, executa busca heurística no grid.
    // 4. Implementa verificação periódica de linha reta ("Theta* light") para atalhos.
    std::pair<std::vector<std::pair<float, float>>, bool> run_a_star(std::pair<float, float> start_tuple, std::pair<float, float> goal_tuple) 
    {
        // Garante que start e goal estão fora de obstáculos
        auto start_search = find_nearest_free_point(start_tuple, 500);
        if (!start_search.second) 
        {
            RCLCPP_WARN(this->get_logger(), "START BLOCKED: Robô preso.");
            return {};
        }
        std::pair<float, float> valid_start = start_search.first;

        auto goal_search = find_nearest_free_point(goal_tuple, 3000);
        if (!goal_search.second) 
        {
            RCLCPP_WARN(this->get_logger(), "GOAL BLOCKED: Destino inalcançável.");
            return {};
        }
        std::pair<float, float> valid_goal = goal_search.first;

        if (valid_start == valid_goal) 
        {
            std::vector<std::pair<float, float>> path;
            path.push_back(valid_goal);
            return {path, true};
        }

        // Tenta linha reta direta primeiro
        std::vector<std::pair<float, float>> initial_path = straight_line(valid_start, valid_goal);
        if(!initial_path.empty())
        {
            initial_path.push_back(valid_start);
            initial_path.push_back(valid_goal);
            return std::make_pair(initial_path, true);
        }

        // Estruturas do A*
        struct Node {
            std::pair<float, float> parent;
            float g_score = std::numeric_limits<float>::infinity();
            float f_score = std::numeric_limits<float>::infinity();
            bool closed = false;
        };

        std::unordered_map<std::pair<float, float>, Node, PairHash> nodes;
        std::unordered_map<std::pair<float, float>, std::vector<std::pair<float, float>>, PairHash> adjacency_list_tuples;
        auto offsets1 = get_offsets(distanceToObstacle_);

        // Expansão inicial dos vizinhos do start e goal para garantir conectividade no grafo
        // ... (código omitido de expansão inicial para brevidade nos comentários) ...
        
        // Heurística Euclidiana
        auto heuristic = [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
            float x1 = std::get<0>(a);
            float y1 = std::get<1>(a);
            float x2 = std::get<0>(b);
            float y2 = std::get<1>(b);
            return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
        };
        
        nodes[valid_start].g_score = 0;
        nodes[valid_start].f_score = heuristic(valid_start, valid_goal);
        
        // Fila de prioridade (Min-Heap)
        struct PairCompare {
            bool operator()(const std::pair<float, std::pair<float, float>>& a, 
                            const std::pair<float, std::pair<float, float>>& b) const {
                return a.first > b.first;
            }
        };
        
        std::priority_queue<
            std::pair<float, std::pair<float, float>>,
            std::vector<std::pair<float, std::pair<float, float>>>,
            PairCompare
        > open_set;
        
        open_set.push({nodes[valid_start].f_score, valid_start});
        
        int iterations = 0;

        while (!open_set.empty()) 
        {
            auto current_pair = open_set.top();
            open_set.pop();
            auto current = current_pair.second;
            
            if (nodes[current].closed) continue;
            
            nodes[current].closed = true;
           
            // Gera vizinhos dinamicamente (Lazy Generation) para economizar memória
            if (current != valid_start && current != valid_goal)
            {
                for (int a = 0; a < 8; a++) 
                {
                    float new_x = round_to_multiple(std::get<0>(current) + offsets1[a][0], distanceToObstacle_, decimals);
                    float new_y = round_to_multiple(std::get<1>(current) + offsets1[a][1], distanceToObstacle_, decimals);
                    std::pair<float, float> neighbor_tuple = std::make_pair(new_x, new_y); 
                    
                    // Só adiciona se não for obstáculo
                    if (obstaclesVertices.find(neighbor_tuple) == obstaclesVertices.end())
                    {
                        adjacency_list_tuples[current].push_back(neighbor_tuple);
                    }
                }
            }
            
            // Reconstrução do caminho se chegou ao alvo
            if (current == valid_goal) 
            {
                std::vector<std::pair<float, float>> path;
                auto current_vertex = current;
                path.insert(path.begin(), current_vertex);
                while (nodes.find(current_vertex) != nodes.end() && current_vertex != valid_start) {
                    current_vertex = nodes[current_vertex].parent;
                    path.insert(path.begin(), current_vertex);
                }
                return std::make_pair(path, false);
            }
            
            // Otimização: Tenta conectar direto ao goal periodicamente
            if(iterations == iterations_before_verification) 
            {
                iterations = 0;
                std::vector<std::pair<float, float>> path1 = straight_line(current, valid_goal);

                if(!path1.empty()) 
                {
                    std::vector<std::pair<float, float>> path;
                    std::vector<std::pair<float, float>> path_to_current;
                    auto current_vertex = current;
                    while (nodes.find(current_vertex) != nodes.end() && current_vertex != valid_start) {
                        path_to_current.insert(path_to_current.begin(), current_vertex);
                        current_vertex = nodes[current_vertex].parent;
                    }
                    path_to_current.insert(path_to_current.begin(), valid_start); 
                    path.insert(path.end(), path_to_current.begin(), path_to_current.end());
                    path.insert(path.end(), path1.begin(), path1.end());
                    return std::make_pair(path, true);
                }
            }
           
            // Relaxamento de arestas
            for (const auto& neighbor : adjacency_list_tuples[current])
            {
                if (nodes.find(neighbor) != nodes.end() && nodes[neighbor].closed) continue;
                
                float tentative_g_score = nodes[current].g_score + heuristic(current, neighbor);
                
                if (nodes.find(neighbor) == nodes.end() || tentative_g_score < nodes[neighbor].g_score) 
                {
                    nodes[neighbor].parent = current;
                    nodes[neighbor].g_score = tentative_g_score;
                    nodes[neighbor].f_score = tentative_g_score + heuristic(neighbor, valid_goal);
                    open_set.push({nodes[neighbor].f_score, neighbor});
                }
            }
            iterations++;
            adjacency_list_tuples.erase(current); // Limpa memória de vizinhos já processados
        }
        
        RCLCPP_WARN(this->get_logger(), "Não foi possível encontrar caminho.");
        return {};
    }
    // DOC-END: run_a_star

    // DOC-START: straight_line
    // Verifica se existe uma linha reta livre de obstáculos entre dois pontos (Raycasting no Grid).
    std::vector<std::pair<float, float>> straight_line(std::pair<float, float> start_tuple, std::pair<float, float> goal_tuple)
    {
        std::pair<float, float> A { std::get<0>(start_tuple), std::get<1>(start_tuple) };
        std::pair<float, float> B { std::get<0>(goal_tuple), std::get<1>(goal_tuple) };

        float ax = std::get<0>(A), ay = std::get<1>(A);
        float bx = std::get<0>(B), by = std::get<1>(B);

        float dx = bx - ax, dy = by - ay;
        float distance = std::sqrt(dx * dx + dy * dy);

        float ux = dx / distance;
        float uy = dy / distance;

        float step = distanceToObstacle_; // Avança conforme resolução do grid
        float t = 0.0f;
        bool obstacleFound = false;
        
        std::vector<std::pair<float, float>> path;

        while (t < distance && obstacleFound == false) 
        {
            std::tuple<float, float, float> point;
            std::get<0>(point) = ax + t * ux;
            std::get<1>(point) = ay + t * uy;

            float new_x = round_to_multiple(std::get<0>(point), distanceToObstacle_, decimals);
            float new_y = round_to_multiple(std::get<1>(point), distanceToObstacle_, decimals);

            std::pair<float, float> neighbor_tuple = std::make_pair(static_cast<float>(new_x), static_cast<float>(new_y));
            
            path.push_back(neighbor_tuple);
            // Verifica colisão
            if (obstaclesVertices.find(neighbor_tuple) != obstaclesVertices.end()) 
            {
                obstacleFound = true;
                break;
            }
            t += step;
        }
        
        if(obstacleFound == true) return {};
        else return path;
    }
    // DOC-END: straight_line

    // DOC-START: store_edges_in_path
    // Pós-processamento do caminho:
    // 1. Suavização (Shortcut Pruning): Remove nós intermediários desnecessários se houver linha de visada.
    // 2. Cálculo de Orientação: Adiciona quaterniões apontando para o próximo ponto do caminho.
    // 3. Preenche os vetores de saída para visualização e execução.
    void store_edges_in_path(
        std::vector<std::pair<float, float>>& path, 
        bool straight_line, 
        std::pair<float, float> original_goal,
        std::vector<VertexDijkstra>& out_path_points,
        std::vector<std::pair<float, float>>& out_path_no_filter
    ) 
    {
        // Limpa os vetores locais
        out_path_points.clear();
        out_path_no_filter.clear();
        
        if (path.empty()) return;

        int k = 0;

        // Tenta simplificar o final do caminho se não for linha reta pura
        if (straight_line == false && path.size() >= 2)
        {
            std::pair<float, float> goal = original_goal;
            for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i)
            {
                float dx = goal.first  - path[i].first;
                float dy = goal.second - path[i].second;
                float dist = std::hypot(dx, dy); 

                if (dist >= security_distance)
                {
                    if (i + 1 < static_cast<int>(path.size())) {
                        path.erase(path.begin() + i + 1, path.end());
                    }
                    break;
                }
            }
        }

        // Algoritmo de "Path Smoothing" (Atalhos)
        while (k < static_cast<int>(path.size()) - 1) 
        {
            bool shortcutFound = false;
            // Olha para frente no caminho, do fim para o início
            for (int i = static_cast<int>(path.size()) - 1; i > k; i--) 
            {
                // Verifica linha reta entre o ponto K e o ponto I
                // ... (código de raycasting omitido para brevidade) ...
                
                bool obstacleFound = false; // (Resultado do raycasting)
                
                // Se não achou obstáculo, podemos pular todos os nós entre K e I
                if (obstacleFound == false) 
                {
                    path.erase(path.begin() + k + 1, path.begin() + i);
                    shortcutFound = true;
                    break;  
                }
            }
            if (shortcutFound == true) k++;
            else break;
        }

        // Densificação: Preenche os espaços vazios entre os nós do caminho suavizado
        // para garantir que o controlador local tenha pontos suficientes para seguir.
        for (size_t i = 0; i < path.size() - 1; i++) 
        {
            float start_x = path[i].first;
            float start_y = path[i].second;
            float end_x   = path[i+1].first;
            float end_y   = path[i+1].second;

            float dx = end_x - start_x;
            float dy = end_y - start_y;
            float dist = std::sqrt(dx * dx + dy * dy);
            
            float ux = (dist > 0) ? (dx / dist) : 0;
            float uy = (dist > 0) ? (dy / dist) : 0;

            float traveled = 0.0f;
            
            while (traveled < dist)
            {
                float px = start_x + ux * traveled;
                float py = start_y + uy * traveled;

                std::pair<float, float> point = std::make_pair(round_to_multiple(px, distanceToObstacle_, decimals), round_to_multiple(py, distanceToObstacle_, decimals));
                
                if(obstaclesVertices.find(point) == obstaclesVertices.end())
                {
                    out_path_no_filter.push_back(point);
                }
                
                traveled += distanceToObstacle_;
            }
        }

        if (!path.empty()) out_path_no_filter.push_back(path.back());

        // Cálculo de Orientação (Quaterniões)
        for (size_t i = 0; i < path.size(); i++) 
        {
            VertexDijkstra vertex;
            vertex.x = std::get<0>(path[i]);
            vertex.y = std::get<1>(path[i]);

            float dx = 0.0f;
            float dy = 0.0f;
            bool calculation_possible = false;

            if (i < path.size() - 1) 
            {
                // Aponta para o próximo nó
                const std::pair<float, float>& current_vertex = path[i];
                const std::pair<float, float>& next_vertex = path[i + 1];
                dx = std::get<0>(next_vertex) - std::get<0>(current_vertex);
                dy = std::get<1>(next_vertex) - std::get<1>(current_vertex);
                calculation_possible = true;
            } 
            else 
            {
                // O último nó aponta para o goal original
                const std::pair<float, float>& current_vertex = path[i];
                dx = original_goal.first - std::get<0>(current_vertex);
                dy = original_goal.second - std::get<1>(current_vertex);
                if (std::sqrt(dx*dx + dy*dy) > 1e-3) calculation_possible = true;
            }

            if (calculation_possible) 
            {
                float distance = std::sqrt(dx * dx + dy * dy);
                if (distance > 0.0f) { dx /= distance; dy /= distance; }

                Eigen::Vector3f direction(dx, dy, 0.0f);
                Eigen::Vector3f reference(1.0f, 0.0f, 0.0f); 
                Eigen::Quaternionf quaternion = Eigen::Quaternionf::FromTwoVectors(reference, direction);

                vertex.orientation_x = quaternion.x();
                vertex.orientation_y = quaternion.y();
                vertex.orientation_z = quaternion.z();
                vertex.orientation_w = quaternion.w();
            }
            else 
            {
                // Mantém orientação anterior se parado
                if (!out_path_points.empty()) { 
                    vertex.orientation_x = out_path_points.back().orientation_x;
                    vertex.orientation_y = out_path_points.back().orientation_y;
                    vertex.orientation_z = out_path_points.back().orientation_z;
                    vertex.orientation_w = out_path_points.back().orientation_w;
                } else {
                    vertex.orientation_w = 1.0;
                }
            }
            out_path_points.push_back(vertex);
        }

        // Publica para visualização no RViz
        publisher_dijkstra_path(out_path_points);
    }
    // DOC-END: store_edges_in_path


    // DOC-START: publisher_dijkstra_path
    // Publica o caminho calculado como nav_msgs/Path para visualização no RViz.
    void publisher_dijkstra_path(const std::vector<VertexDijkstra>& points_to_publish)
    {
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = "world";

        for (const auto& vertex : points_to_publish)
        {
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = this->now();
            pose_stamped.header.frame_id = "world";
            
            pose_stamped.pose.position.x = vertex.x;
            pose_stamped.pose.position.y = vertex.y;
            pose_stamped.pose.position.z = 0.0;
            pose_stamped.pose.orientation.x = vertex.orientation_x;
            pose_stamped.pose.orientation.y = vertex.orientation_y;
            pose_stamped.pose.orientation.z = vertex.orientation_z;
            pose_stamped.pose.orientation.w = vertex.orientation_w;
            
            path_msg.poses.push_back(pose_stamped);
        }
        publisher_nav_path_->publish(path_msg);
    }
    // DOC-END: publisher_dijkstra_path



    // Callbacks.
    void callback_odom(const nav_msgs::msg::Odometry::SharedPtr msg) 
    {
        std::lock_guard<std::mutex> lock(odom_mutex);
        pose_x_ = msg->pose.pose.position.x;
        pose_y_ = msg->pose.pose.position.y;
        pose_z_ = 0.0;
    }

    // DOC-START: topic_callback
    // Callback da Nuvem de Pontos (Obstáculos).
    // Converte a nuvem de pontos em um Set de coordenadas discretas (Grid).
    void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);

        // Iterador eficiente para ler campos X e Y da mensagem binária
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y) 
        {
            float x = *iter_x;
            float y = *iter_y;
            
            // Arredonda e insere no conjunto de obstáculos
            std::pair<float, float> index = std::make_pair(
                round_to_multiple(x, distanceToObstacle_, decimals),
                round_to_multiple(y, distanceToObstacle_, decimals)
            );
            obstaclesVertices.insert(index);
        }
    }
    // DOC-END: topic_callback

    // Action server.

    // DOC-START: handle_goal
    // Aceita requisição de planejamento.
    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const mobile_manipulation_interfaces::action::Path::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Goal recebido: [x: %.2f, y: %.2f]", 
            goal->pose.position.x, goal->pose.position.y);
        (void)uuid;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }
    // DOC-END: handle_goal

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Path>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Cancelamento recebido.");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Path>> goal_handle)
    {
        using namespace std::placeholders;
        
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            active_goal_handle_ = goal_handle;
        }

        // Lança thread para execução pesada do A*
        std::thread{std::bind(&AStar::execute, this, std::placeholders::_1), goal_handle}.detach();
    }

    
    // DOC-START: execute
    // Execução da Action: Planeja o caminho e monitora colisão dinâmica.
    void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Path>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Iniciando A*...");
        
        const auto goal = goal_handle->get_goal();
        auto result = std::make_shared<mobile_manipulation_interfaces::action::Path::Result>();
        auto feedback = std::make_shared<mobile_manipulation_interfaces::action::Path::Feedback>();

        // Vetores locais para armazenar o caminho da thread atual (Thread-safe)
        std::vector<VertexDijkstra> local_path_points;
        std::vector<std::pair<float, float>> local_path_without_filter;

        std::pair<float, float> current_goal_pose = {goal->pose.position.x, goal->pose.position.y};
        std::pair<float, float> start_pose;

        {
            std::lock_guard<std::mutex> lock(odom_mutex);
            start_pose = {pose_x_, pose_y_};
        }
        
        rclcpp::Rate loop_rate(20.0); // 20 Hz de verificação
        bool path_needs_calculation = false;

        try {
            // 1. Planejamento Inicial
            {
                feedback->recalculating_path = false;
                std::pair<std::vector<std::pair<float, float>>, bool> a_star_result;
                std::vector<std::pair<float, float>> shortestPath;
                bool straight_line = false;

                {
                    std::lock_guard<std::mutex> lock(map_mutex_); // Protege leitura do mapa
                    a_star_result = run_a_star(start_pose, current_goal_pose);
                    shortestPath = a_star_result.first;
                    straight_line = a_star_result.second;

                    if (!shortestPath.empty())
                    {
                        // Processa o caminho (suavização)
                        store_edges_in_path(shortestPath, straight_line, current_goal_pose, local_path_points, local_path_without_filter);
                    }
                }

                if (shortestPath.empty())
                {
                    RCLCPP_WARN(this->get_logger(), "Falha no A* inicial.");
                    path_needs_calculation = true;
                    feedback->recalculating_path = true;
                    goal_handle->publish_feedback(feedback);
                }
                else
                {
                    // Envia caminho válido via feedback
                    feedback->path.poses.clear();
                    feedback->path.header.stamp = this->now();
                    feedback->path.header.frame_id = "world";

                    for (const auto& vertex : local_path_points) 
                    {
                        geometry_msgs::msg::PoseStamped pose_stamped;
                        // ... preenche pose ...
                        pose_stamped.pose.position.x = vertex.x;
                        pose_stamped.pose.position.y = vertex.y;
                        // ...
                        feedback->path.poses.push_back(pose_stamped);
                    }
                    
                    goal_handle->publish_feedback(feedback); 
                    publisher_dijkstra_path(local_path_points); 
                }
            }

            // 2. Loop de Monitoramento
            while (rclcpp::ok()) 
            {
                // Verifica preempção ou cancelamento
                {
                    std::lock_guard<std::mutex> lock(goal_mutex_);
                    if (active_goal_handle_ != goal_handle) return; 
                }
                if (goal_handle->is_canceling()) {
                    result->success = false;
                    goal_handle->canceled(result);
                    return;
                }

                // Verifica colisão no caminho atual
                if (!path_needs_calculation)
                {
                    std::lock_guard<std::mutex> lock(map_mutex_); 
                    for(size_t i = 0; i < local_path_without_filter.size(); ++i)
                    {
                        if(obstaclesVertices.find(local_path_without_filter[i]) != obstaclesVertices.end())
                        {
                            RCLCPP_WARN(this->get_logger(), "Obstáculo no caminho! Recalculando...");
                            path_needs_calculation = true;
                            break; 
                        }
                    }
                }

                // Recálculo Dinâmico
                if (path_needs_calculation)
                {
                    // Avisa que está recalculando (para o robô parar)
                    feedback->recalculating_path = true;
                    feedback->path.poses.clear();
                    goal_handle->publish_feedback(feedback);
                    
                    rclcpp::sleep_for(std::chrono::milliseconds(100));

                    // Tenta novo plano a partir da posição atual
                    // ... (lógica de rechamada do A* similar ao início) ...
                    // Se sucesso, envia novo path e seta path_needs_calculation = false
                }

                loop_rate.sleep();
            }
        }
        catch (...) {
            goal_handle->abort(result);
        }
    }
    // DOC-END: execute
    

};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AStar>());
    rclcpp::shutdown();
    return 0;
}