#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include <limits>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp" 

#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include "geometry_msgs/msg/vector3.hpp"
#include <yaml-cpp/yaml.h>

namespace storage_manager 
{

struct StorageResult 
{
    bool success = false;
    std::string storage_name;
    geometry_msgs::msg::Pose pose;      
    std::vector<double> limits; 
    std::vector<int> indexes;
    geometry_msgs::msg::Vector3 size;          
    int current_count = 0;
    int max_count = -1;
};

class StorageNode : public rclcpp::Node 
{
public:
    explicit StorageNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~StorageNode() override = default;


    StorageResult getBestStorage(const std::string& label, const geometry_msgs::msg::Pose& robot_pose);


    void incrementStorageCount(const std::string& storage_name, int amount = 1);
    void addNewIndexes(const std::string& storage_name, const std::vector<int>& new_indexes);

private:
    // --- ESTRUTURAS INTERNAS ---
    struct StorageInfo {
        geometry_msgs::msg::Pose pose;
        double size_x;
        double size_y;
        std::vector<int> indexes;
        int max_objects = -1;   
        int actual_objects = 0; 
    };

    struct StorageLimits {
        double min_x, max_x, min_y, max_y;
    };

    std::mutex mutex_;
    
    // Dá pra otimizar fortemente isso aqui. Quem sabe um dia.
    std::unordered_map<std::string, std::vector<std::string>> labels_to_storage_;
    std::unordered_map<std::string, std::vector<StorageInfo>> storage_map_;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr update_sub_;

    void loadLabelToStorage(const std::string &yaml_file);
    void loadStoragePoses(const std::string &yaml_file);
    
    StorageLimits calculateLimits(const geometry_msgs::msg::Pose& pose, double sx, double sy);
    
};

} // namespace storage_manager