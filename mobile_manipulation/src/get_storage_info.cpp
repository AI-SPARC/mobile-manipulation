#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <limits>

#include <yaml-cpp/yaml.h>
#include "rclcpp/rclcpp.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "std_msgs/msg/bool.hpp"

#include <tf2/utils.h> 
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "mobile_manipulation_interfaces/srv/get_storage_info.hpp"

class GetStorageInfo : public rclcpp::Node 
{

public:
    GetStorageInfo()
     : Node("simple_manipulation_storage")
    {
        this->declare_parameter<std::string>("label_to_storage_yaml_file", "");
        this->declare_parameter<std::string>("storage_poses_yaml_file", "");

        std::string label_to_storage_yaml_file = this->get_parameter("label_to_storage_yaml_file").as_string();
        std::string storage_poses_yaml_file = this->get_parameter("storage_poses_yaml_file").as_string();
    
        if(!label_to_storage_yaml_file.empty())
        {
            loadLabelToStorage(label_to_storage_yaml_file);
        }

        if(!storage_poses_yaml_file.empty())
        {
            loadStoragePoses(storage_poses_yaml_file);
        }

        service_ = this->create_service<mobile_manipulation_interfaces::srv::GetStorageInfo>(
            "get_storage_info",
            std::bind(&GetStorageInfo::handle_service_request, this, std::placeholders::_1, std::placeholders::_2)
        );

    }   

    ~GetStorageInfo() {}

private:

    struct StorageInfo 
    {
        geometry_msgs::msg::Pose pose;
        double size_x;
        double size_y;
    };

    struct StorageLimits 
    {
        double min_x;
        double max_x;
        double min_y;
        double max_y;
    };

    rclcpp::Service<mobile_manipulation_interfaces::srv::GetStorageInfo>::SharedPtr service_;

    std::unordered_map<std::string, std::vector<StorageInfo>> storage;
    std::unordered_map<std::string, std::vector<std::string>> labels_to_storage;

    void loadLabelToStorage(const std::string &yaml_file)
    {
        try {
            YAML::Node config = YAML::LoadFile(yaml_file);

            for (auto it = config.begin(); it != config.end(); ++it)
            {

                std::string group_name = it->first.as<std::string>();  
                const YAML::Node &entries = it->second;                
                std::vector<std::string> storages;

                for (const auto &entry : entries)
                {
                    const YAML::Node &value = entry["storage"];
                    if (value)
                    {
                        storages.push_back(value.as<std::string>());
                    }
                }

                labels_to_storage[group_name] = storages;
            }
        } catch (const YAML::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error loading Label YAML: %s", e.what());
        }
    }

    void loadStoragePoses(const std::string &yaml_file)
    {
        try {
            YAML::Node config = YAML::LoadFile(yaml_file);
            
            for (auto it = config.begin(); it != config.end(); ++it)
            {
                std::string storage_name = it->first.as<std::string>();
                const YAML::Node &locations = it->second;
                
                std::vector<StorageInfo> infos;

                for (const auto &loc_node : locations)
                {
                    StorageInfo info;

                    if (loc_node["position"]) 
                    {
                        const auto& pos = loc_node["position"];
                        info.pose.position.x = pos[0].as<double>();
                        info.pose.position.y = pos[1].as<double>();
                        info.pose.position.z = pos[2].as<double>();
                    }

                    if (loc_node["orientation"]) 
                    {
                        const auto& ori = loc_node["orientation"];
                        info.pose.orientation.x = ori[0].as<double>();
                        info.pose.orientation.y = ori[1].as<double>();
                        info.pose.orientation.z = ori[2].as<double>();
                        info.pose.orientation.w = ori[3].as<double>();
                    } 
                    else 
                    {
                        info.pose.orientation.w = 1.0;
                    }
                    
                    if (loc_node["size"]) {
                        const auto& s = loc_node["size"];
                        info.size_x = s[0].as<double>();
                        info.size_y = s[1].as<double>();
                    } 
                    else 
                    {
                        info.size_x = 0.5;
                        info.size_y = 0.5;
                    }

                    infos.push_back(info);
                }
                storage[storage_name] = infos;
            }
        } 
        catch (const YAML::Exception &e) 
        {
            RCLCPP_ERROR(this->get_logger(), "Error loading Storage YAML: %s", e.what());
        }
    }

    std::pair<bool, std::pair<std::string, StorageInfo>> getClosestStorage(const std::string& label, double px, double py, double pz)
    {
        double best_dist = std::numeric_limits<double>::max();
        std::string best_storage_name;
        StorageInfo best_info; 

        if (!labels_to_storage.count(label)) {
            return {false, {"", StorageInfo()}};
        }

        const auto& storage_list = labels_to_storage[label];

        for (const auto& storage_name : storage_list)
        {
            if (!storage.count(storage_name)) continue;
            
            const auto& infos = storage.at(storage_name);

            for (const auto& info : infos)
            {
                double dx = info.pose.position.x - px;
                double dy = info.pose.position.y - py;
                double dz = info.pose.position.z - pz;
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_storage_name = storage_name;
                    best_info = info; 
                }
            }
        }

        if (best_storage_name.empty()) 
        {
            return {false, {"", StorageInfo()}};
        }
        
        return {true, {best_storage_name, best_info}};
    }

    StorageLimits calculateStorageLimits(const geometry_msgs::msg::Pose& pose, double size_x, double size_y)
    {
        double yaw = tf2::getYaw(pose.orientation);
        
        double cos_theta = std::cos(yaw);
        double sin_theta = std::sin(yaw);

        double dx = size_x / 2.0;
        double dy = size_y / 2.0;

        double local_corners_x[4] = { dx,  dx, -dx, -dx};
        double local_corners_y[4] = { dy, -dy,  dy, -dy};

        StorageLimits limits;
        limits.min_x = std::numeric_limits<double>::max();
        limits.max_x = std::numeric_limits<double>::lowest();
        limits.min_y = std::numeric_limits<double>::max();
        limits.max_y = std::numeric_limits<double>::lowest();

        for(int i = 0; i < 4; i++)
        {
            double global_x = pose.position.x + (local_corners_x[i] * cos_theta - local_corners_y[i] * sin_theta);
            double global_y = pose.position.y + (local_corners_x[i] * sin_theta + local_corners_y[i] * cos_theta);

            if (global_x < limits.min_x) limits.min_x = global_x;
            if (global_x > limits.max_x) limits.max_x = global_x;
            if (global_y < limits.min_y) limits.min_y = global_y;
            if (global_y > limits.max_y) limits.max_y = global_y;
        }

        return limits;
    }

    // Service Callback
    void handle_service_request(const std::shared_ptr<mobile_manipulation_interfaces::srv::GetStorageInfo::Request> request,
        std::shared_ptr<mobile_manipulation_interfaces::srv::GetStorageInfo::Response> response)
    {
        std::string id = request->object_id;
        
        double px = request->pose.position.x;
        double py = request->pose.position.y;
        double pz = request->pose.position.z;

        std::pair<bool, std::pair<std::string, StorageInfo>> result_data = getClosestStorage(id, px, py, pz);
        
        bool found = result_data.first;

        if (found == true)
        {
            std::string storage_name = result_data.second.first;
            StorageInfo info = result_data.second.second;

            StorageLimits limites = calculateStorageLimits(info.pose, info.size_x, info.size_y);

            response->success = true;
            response->pose = info.pose;
            
            response->limits = {
                static_cast<float>(limites.min_x), 
                static_cast<float>(limites.max_x), 
                static_cast<float>(limites.min_y), 
                static_cast<float>(limites.max_y)
            };

            RCLCPP_INFO(this->get_logger(), "Enviando dados do storage '%s' para ID: %s", storage_name.c_str(), id.c_str());
        }
        else
        {
            response->success = false;
            response->pose = geometry_msgs::msg::Pose(); 
            response->limits = {}; 

            RCLCPP_WARN(this->get_logger(), "Nenhum storage encontrado para label/ID: %s na posição (%.2f, %.2f)", id.c_str(), px, py);
        }
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GetStorageInfo>());
  rclcpp::shutdown();
  return 0;
}