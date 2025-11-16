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
#include "object_manipulation_interfaces/srv/goal_pose.hpp"

using namespace std::chrono_literals;

namespace std 
{
    template <>
    struct hash<std::tuple<float, float, float>> 
    {
        size_t operator()(const std::tuple<float, float, float>& t) const 
        {
            size_t h1 = hash<float>()(std::get<0>(t));
            size_t h2 = hash<float>()(std::get<1>(t));
            size_t h3 = hash<float>()(std::get<2>(t));
            
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

namespace std {
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



class AStar : public rclcpp::Node {

private:


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

    struct CompareWithTieBreaker {
        bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const {
            if (std::abs(a.first - b.first) < 1e-6) {
                return a.second > b.second;
            }
            return a.first > b.first;
        }
    };
    struct PairHash {
        std::size_t operator()(const std::pair<float, float>& p) const {
            auto h1 = std::hash<float>{}(p.first);
            auto h2 = std::hash<float>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    struct TupleCompare {
        bool operator()(const std::pair<float, std::tuple<float, float, float>>& a, 
                        const std::pair<float, std::tuple<float, float, float>>& b) const {
            return a.first > b.first;
        }
    };

    struct PositionProb {
        float x;
        float y;
        float prob;
    };
    
    //Service.
    rclcpp::Service<object_manipulation_interfaces::srv::GoalPose>::SharedPtr service_;
    
    //Publishers.
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr publisher_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_nav_path_;

    //Subscriptions.
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_navigable_removed_vertices;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_odom_;

    //Timers.
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr timer_path_;
    rclcpp::TimerBase::SharedPtr timer_visualize_path_;
    rclcpp::TimerBase::SharedPtr parameterTimer;    
    

    size_t i_ = 0; 
    float pose_x_ = 3.24, pose_y_ = 8.22, pose_z_ = 0.0;
    float distanceToObstacle_, security_distance = 0.5;
    int decimals = 0, iterations_before_verification = 10;

    std::vector<VertexDijkstra> verticesDestino_;
    std::vector<VertexDijkstra> verticesDijkstra;

    std::unordered_map<int, std::vector<int>> adjacency_list;
    std::unordered_set<std::pair<float, float>, PairHash> obstaclesVertices;
    std::unordered_map<int, Vertex> navigableVerticesMapInteger;

    std::string yaml_file;

    inline float round_to_multiple(float value, float multiple, int decimals) 
    {
        if (multiple == 0.0) return value; 
        
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0, decimals);
        result = std::round(result * factor) / factor;
        
        return result;
    }
    

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
    
    std::pair<std::vector<std::pair<float, float>>, bool> run_a_star(std::pair<float, float> start_tuple, std::pair<float, float> goal_tuple) 
    {
        std::vector<std::pair<float, float>> initial_path = straight_line(start_tuple, goal_tuple);

        if(!initial_path.empty())
        {
            initial_path.push_back(start_tuple);
            initial_path.push_back(goal_tuple);

            return std::make_pair(initial_path, true);
        }

    
        
        struct Node {
            std::pair<float, float> parent;
            float g_score = std::numeric_limits<float>::infinity();
            float f_score = std::numeric_limits<float>::infinity();
            bool closed = false;
        };
        
        
        std::unordered_map<std::pair<float, float>, Node, PairHash> nodes;
        
        std::unordered_map<std::pair<float, float>, std::vector<std::pair<float, float>>, PairHash> adjacency_list_tuples;
        
        auto offsets1 = get_offsets(distanceToObstacle_);
        
   
        float new_x = 0.0, new_y = 0.0;
        bool findNavigableVertice = false;
        
        for(int i = 1; i <= 2; i++)
        {
            for (int a = 0; a < 8; a++) 
            {
                new_x = round_to_multiple(std::get<0>(start_tuple) + (offsets1[a][0] * i), distanceToObstacle_, decimals);
                new_y = round_to_multiple(std::get<1>(start_tuple) + (offsets1[a][1] * i), distanceToObstacle_, decimals);

                std::pair<float, float> neighbor_tuple = std::make_pair(static_cast<float>(new_x), static_cast<float>(new_y));
                
                if (obstaclesVertices.find(neighbor_tuple) == obstaclesVertices.end())
                { 
                    adjacency_list_tuples[start_tuple].push_back(neighbor_tuple);
                    findNavigableVertice = true;
                }
            }

            if(findNavigableVertice == true)
            {
                break;
            }
        }
        
        if(findNavigableVertice == false) 
        {
            RCLCPP_WARN(this->get_logger(), "The robot is too far of the navigable area.");
            return {};
        }
        
        bool findNavigableGoalVertice = false;
        
        for(int i = 1; i <= 2; i++)
        {
            for (int a = 0; a < 8; a++) 
            {
                new_x = round_to_multiple(std::get<0>(goal_tuple) + (offsets1[a][0] * i), distanceToObstacle_, decimals);
                new_y = round_to_multiple(std::get<1>(goal_tuple) + (offsets1[a][1] * i), distanceToObstacle_, decimals);

                std::pair<float, float> neighbor_tuple = std::make_pair(static_cast<float>(new_x), static_cast<float>(new_y));
                
                if (obstaclesVertices.find(neighbor_tuple) == obstaclesVertices.end())
                { 
                    adjacency_list_tuples[neighbor_tuple].push_back(goal_tuple);
                    findNavigableGoalVertice = true;
                }
            }


            if(findNavigableGoalVertice == true)
            {
                break;
            }
        }
        
        if(findNavigableGoalVertice == false)
        {
            RCLCPP_WARN(this->get_logger(), "Destination is too far of the navigable area. Increase navigable area.");
            return {};
        }
        
        auto heuristic = [](const std::pair<float, float>& a, const std::pair<float, float>& b) {
            float x1 = std::get<0>(a);
            float y1 = std::get<1>(a);
            
            float x2 = std::get<0>(b);
            float y2 = std::get<1>(b);
            
            return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
        };
        
        nodes[start_tuple].g_score = 0;
        nodes[start_tuple].f_score = heuristic(start_tuple, goal_tuple);
        
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
        
        open_set.push({nodes[start_tuple].f_score, start_tuple});
        
        int iterations = 0;

        while (!open_set.empty()) 
        {
            auto current_pair = open_set.top();
            open_set.pop();
            auto current = current_pair.second;
            
            if (nodes[current].closed)
                continue;
                
            if (current_pair.first > nodes[current].f_score)
                continue;
                
            nodes[current].closed = true;
           
            if (current != start_tuple && current != goal_tuple)
            {
                for (int a = 0; a < 8; a++) 
                {
                    new_x = round_to_multiple(std::get<0>(current) + offsets1[a][0], distanceToObstacle_, decimals);
                    new_y = round_to_multiple(std::get<1>(current) + offsets1[a][1], distanceToObstacle_, decimals);

                    std::pair<float, float> neighbor_tuple = std::make_pair(static_cast<float>(new_x), static_cast<float>(new_y)); 
                    
                    if (obstaclesVertices.find(neighbor_tuple) == obstaclesVertices.end())
                    {
                        adjacency_list_tuples[current].push_back(neighbor_tuple);
                    }
                }
    
            }
            
            if (current == goal_tuple) 
            {
                std::vector<std::pair<float, float>> path;
                auto current_vertex = current;
                
                path.insert(path.begin(), current_vertex);
                
                while (nodes.find(current_vertex) != nodes.end() && 
                    current_vertex != start_tuple) {
                    current_vertex = nodes[current_vertex].parent;
                    path.insert(path.begin(), current_vertex);
                }
                
                return std::make_pair(path, false);
            }
            
            if(iterations == iterations_before_verification) 
            {
                iterations = 0;
                
                std::vector<std::pair<float, float>> path1 = straight_line(current, goal_tuple);

                if(!path1.empty()) 
                {
                    std::vector<std::pair<float, float>> path;
                    
                    std::vector<std::pair<float, float>> path_to_current;
                    auto current_vertex = current;
                    
                    while (nodes.find(current_vertex) != nodes.end() && 
                        current_vertex != start_tuple) {
                        path_to_current.insert(path_to_current.begin(), current_vertex);
                        current_vertex = nodes[current_vertex].parent;
                    }
                    path_to_current.insert(path_to_current.begin(), start_tuple); 
                    
                    path.insert(path.end(), path_to_current.begin(), path_to_current.end());
                    
                    path.insert(path.end(), path1.begin(), path1.end());
                    
                    return std::make_pair(path, true);
                }
            }
           
            
            
           
            for (const auto& neighbor : adjacency_list_tuples[current])
            {
                if (nodes.find(neighbor) != nodes.end() && nodes[neighbor].closed)
                    continue;
                
                float tentative_g_score = nodes[current].g_score + heuristic(current, neighbor);
                
                if (nodes.find(neighbor) == nodes.end() || tentative_g_score < nodes[neighbor].g_score) 
                {
                    nodes[neighbor].parent = current;
                    nodes[neighbor].g_score = tentative_g_score;
                    nodes[neighbor].f_score = tentative_g_score + heuristic(neighbor, goal_tuple);
                    open_set.push({nodes[neighbor].f_score, neighbor});
                }
            }
            
            iterations++;
            adjacency_list_tuples.erase(current);
        }
        
        RCLCPP_WARN(this->get_logger(), "It is not possible to reach the destination.");
        return {};
    }

    std::vector<std::pair<float, float>> straight_line(std::pair<float, float> start_tuple, std::pair<float, float> goal_tuple)
    {
        
        
        std::pair<float, float> A {
            std::get<0>(start_tuple),
            std::get<1>(start_tuple),
        };
        std::pair<float, float> B {
            std::get<0>(goal_tuple),
            std::get<1>(goal_tuple),
        };

        float ax = std::get<0>(A), ay = std::get<1>(A);
        float bx = std::get<0>(B), by = std::get<1>(B);

        float dx = bx - ax, dy = by - ay;
        float distance = std::sqrt(dx * dx + dy * dy);

        float ux = dx / distance;
        float uy = dy / distance;

        float step = distanceToObstacle_;
        float t = 0.0f;
        bool obstacleFound = false;
        auto offsets1 = get_offsets(distanceToObstacle_);
        
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
            if (obstaclesVertices.find(neighbor_tuple) != obstaclesVertices.end()) 
            {
                
                obstacleFound = true;
                break;
            }

            t += step;
        }

        
        if(obstacleFound == true)
        {
            return {};
        }
        else
        {
            return path;
        }
    }

    void store_edges_in_path(std::vector<std::pair<float, float>>& path, bool straight_line) 
    {
        verticesDijkstra.clear();
        
        if (path.empty()) {
            return;
        }

        auto start_time1_ = std::chrono::high_resolution_clock::now();
        int k = 0;

        if (straight_line == false && path.size() >= 2)
        {
            auto goal = path.back();

            for (int i = static_cast<int>(path.size()) - 2; i >= 0; --i)
            {
                float dx = goal.first  - path[i].first;
                float dy = goal.second - path[i].second;
                float dist = std::sqrt(dx * dx + dy * dy);

                if (dist >= security_distance)
                {
                    float ux = dx / dist;
                    float uy = dy / dist;

                    std::pair<float, float> new_goal = {
                        goal.first  - ux * security_distance,
                        goal.second - uy * security_distance
                    };

                    path.erase(path.begin() + i + 1, path.end());
                    path.push_back(new_goal);
                    break;
                }
            }
        }


     
        while (k < static_cast<int>(path.size()) - 1) 
        {
            bool shortcutFound = false;
            for (int i = static_cast<int>(path.size()) - 1; i > k; i--) 
            {
                std::pair<float, float> A {
                    std::get<0>(path[k]),
                    std::get<1>(path[k]),
                };
                std::pair<float, float> B {
                    std::get<0>(path[i]),
                    std::get<1>(path[i]),
                };
        
                float ax = std::get<0>(A), ay = std::get<1>(A);
                float bx = std::get<0>(B), by = std::get<1>(B);
        
                float dx = bx - ax, dy = by - ay;
                float distance = std::sqrt(dx * dx + dy * dy);
        
                if (distance == 0) 
                {
                    continue;
                }
        
                float ux = dx / distance;
                float uy = dy / distance;
        
                float step = distanceToObstacle_;
                float t = 0.0f;
                bool obstacleFound = false;
                auto offsets1 = get_offsets(distanceToObstacle_);
        
                while (t < distance && obstacleFound == false) 
                {
                    std::pair<float, float> point;
                    std::get<0>(point) = ax + t * ux;
                    std::get<1>(point) = ay + t * uy;
        
                    double new_x = round_to_multiple(std::get<0>(point), distanceToObstacle_, decimals);
                    double new_y = round_to_multiple(std::get<1>(point), distanceToObstacle_, decimals);
        
                    std::pair<float, float> neighbor_tuple = std::make_pair(static_cast<float>(new_x), static_cast<float>(new_y));
                    
                    
                    if (obstaclesVertices.find(neighbor_tuple) != obstaclesVertices.end()) 
                    {
                        obstacleFound = true;
                        break;
                    }
                

                    t += step;
                }
        
                if (obstacleFound == false) 
                {
                    path.erase(path.begin() + k + 1, path.begin() + i);
                    shortcutFound = true;
                    break;  
                }
            }
        
       

            if (shortcutFound == true)
            {
                k++;
            } 
            else if(shortcutFound == false)
            {
                break;
            }
        }

      
        auto end_time1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration1 = end_time1 - start_time1_; 

        RCLCPP_INFO(this->get_logger(), "A* filter execution time: %.10f", duration1.count());


        if (straight_line == true) 
        {
            auto& last = path.back();
            auto& second_last = path[path.size() - 2];

            float dx = last.first - second_last.first;
            float dy = last.second - second_last.second;
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist > security_distance)  
            {
                float ux = dx / dist;
                float uy = dy / dist;

                last.first  -= ux * security_distance;
                last.second -= uy * security_distance;
            }
        }
        

        for (size_t i = 0; i < path.size(); i++) 
        {
            VertexDijkstra vertex;
            
            vertex.x = std::get<0>(path[i]);
            vertex.y = std::get<1>(path[i]);

            if (i < path.size() - 1) 
            {
                const std::pair<float, float>& current_vertex = path[i];
                const std::pair<float, float>& next_vertex = path[i + 1];

                float dx = std::get<0>(next_vertex) - std::get<0>(current_vertex);
                float dy = std::get<1>(next_vertex) - std::get<1>(current_vertex);
                float distance = std::sqrt(dx * dx + dy * dy);

                if (distance > 0.0f) {
                    dx /= distance;
                    dy /= distance;
                }

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
                vertex.orientation_x = 0.0;
                vertex.orientation_y = 0.0;
                vertex.orientation_z = 0.0;
                vertex.orientation_w = 1.0;
            }

            verticesDijkstra.push_back(vertex);
        }


        publisher_dijkstra();
        publisher_dijkstra_path();
    }


    /*

        PUBLISHERS.

    */


    
    void publisher_dijkstra()
    {   
        geometry_msgs::msg::PoseArray message;
        message.header.stamp = this->now();
        message.header.frame_id = "world";

        for (const auto& vertex : verticesDijkstra) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = vertex.x;
            pose.position.y = vertex.y;
            pose.position.z = 0.0;
            pose.orientation.x = vertex.orientation_x;
            pose.orientation.y = vertex.orientation_y;
            pose.orientation.z = vertex.orientation_z;
            pose.orientation.w = vertex.orientation_w; 
            message.poses.push_back(pose);
        }

        publisher_path_->publish(message);
    }

    void publisher_dijkstra_path()
    {
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = "world";

        for (const auto& vertex : verticesDijkstra)
        {
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = this->now();
            pose_stamped.header.frame_id = "map";
            
            pose_stamped.pose.position.x = vertex.x;
            pose_stamped.pose.position.y = vertex.y;
            pose_stamped.pose.position.z = 0.0  ;
            pose_stamped.pose.orientation.x = vertex.orientation_x;
            pose_stamped.pose.orientation.y = vertex.orientation_y;
            pose_stamped.pose.orientation.z = vertex.orientation_z;
            pose_stamped.pose.orientation.w = vertex.orientation_w;
            
            path_msg.poses.push_back(pose_stamped);
        }

        publisher_nav_path_->publish(path_msg);
    }



    /*
    
        CALLBACKS.

    */
    

    

    void callback_odom(const nav_msgs::msg::Odometry::SharedPtr msg) 
    {
        pose_x_ = msg->pose.pose.position.x;
        pose_y_ = msg->pose.pose.position.y;
        pose_z_ = 0.0;
    }

        
    void load_black_points(const std::string &yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);
        std::filesystem::path yaml_path(yaml_file);
        std::filesystem::path image_path = yaml_path.parent_path() / config["image"].as<std::string>();

        double resolution = config["resolution"].as<double>();
        std::vector<double> origin = config["origin"].as<std::vector<double>>();
        int negate = config["negate"] ? config["negate"].as<int>() : 0;
        double occ_th = config["occupied_thresh"] ? config["occupied_thresh"].as<double>() : 0.65;

        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_UNCHANGED);
        if (image.empty()) 
        {
            throw std::runtime_error("Falha ao carregar imagem: " + image_path.string());
            return;
        }

        obstaclesVertices.clear();

        for (int y = 0; y < image.rows; ++y) 
        {
            for (int x = 0; x < image.cols; ++x) {
            unsigned char pixel = image.at<unsigned char>(image.rows - y - 1, x);
            if (negate) pixel = 255 - pixel;
            double occ = (255 - pixel) / 255.0;

            if (occ > occ_th) 
            {  
                float wx = origin[0] + (x + 0.5f) * resolution;
                float wy = origin[1] + (y + 0.5f) * resolution;
                obstaclesVertices.insert({wx, wy});
            }
            }
        }
    }
    

    // Service

    void handle_request(const std::shared_ptr<object_manipulation_interfaces::srv::GoalPose::Request> request, std::shared_ptr<object_manipulation_interfaces::srv::GoalPose::Response> response)
    {
        std::pair<float, float> initial_pose, goal_pose;

        initial_pose = std::make_pair(pose_x_, pose_y_);

        geometry_msgs::msg::Pose pose = request->pose;

        goal_pose = std::make_pair(pose.position.x, pose.position.y);

        std::pair<std::vector<std::pair<float, float>>, bool> a_star_result = run_a_star(initial_pose, goal_pose);
        std::vector<std::pair<float, float>> shortestPath = std::get<0>(a_star_result);
        bool straight_line = std::get<1>(a_star_result);
        store_edges_in_path(shortestPath, straight_line);

        bool success = true;  

        response->success = success;

        if (success)
        {
            RCLCPP_INFO(this->get_logger(), "Processamento concluído com sucesso!");
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Falha ao processar o pedido!");
        }
    }

public:
    AStar()
     : Node("a_star")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<double>("path_resolution", 0.05);
        this->declare_parameter<double>("security_distance", 0.50);
        this->declare_parameter<int>("iterations_before_verification", 10);

        yaml_file = this->get_parameter("yaml_file").as_string();
        distanceToObstacle_ =  static_cast<float>(this->get_parameter("path_resolution").get_parameter_value().get<double>());
        security_distance = static_cast<float>(this->get_parameter("security_distance").get_parameter_value().get<double>());
        iterations_before_verification = this->get_parameter("iterations_before_verification").get_parameter_value().get<int>();

        RCLCPP_INFO(this->get_logger(), "path_resolution is set to: %f", distanceToObstacle_);
        RCLCPP_INFO(this->get_logger(), "iterations_before_verification is set to: %d", iterations_before_verification);

        service_ = this->create_service<object_manipulation_interfaces::srv::GoalPose>("/goal_pose",std::bind(&AStar::handle_request, this, std::placeholders::_1, std::placeholders::_2));

        decimals = count_decimals(distanceToObstacle_);

     
        publisher_nav_path_ = this->create_publisher<nav_msgs::msg::Path>("visualize_path", 10);

        publisher_path_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/path", 10);
        

        subscription_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&AStar::callback_odom, this, std::placeholders::_1));


     
        // load_black_points(yaml_file);
    }
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    
    rclcpp::spin(std::make_shared<AStar>());
    rclcpp::shutdown();
    return 0;
}