#include <storage_manager/GetStorageInfo.hpp>

namespace storage_manager 
{

StorageNode::StorageNode(const rclcpp::NodeOptions & options)
: Node("storage_manager_node", options)
{
    this->declare_parameter<std::string>("label_to_storage_yaml_file", "");
    this->declare_parameter<std::string>("storage_poses_yaml_file", "");

    std::string label_file = this->get_parameter("label_to_storage_yaml_file").as_string();
    std::string poses_file = this->get_parameter("storage_poses_yaml_file").as_string();

    if(!label_file.empty()) {
        loadLabelToStorage(label_file);
    } else {
        RCLCPP_WARN(this->get_logger(), "Param 'label_to_storage_yaml_file' is empty.");
    }

    if(!poses_file.empty()) {
        loadStoragePoses(poses_file);
    } else {
        RCLCPP_WARN(this->get_logger(), "Param 'storage_poses_yaml_file' is empty.");
    }

    RCLCPP_INFO(this->get_logger(), "StorageNode Initialized (Ready for Direct Access).");
}

StorageResult StorageNode::getBestStorage(const std::string& label, const geometry_msgs::msg::Pose& robot_pose)
{
    std::lock_guard<std::mutex> lock(mutex_);

    StorageResult result;
    result.success = false;

    if (labels_to_storage_.find(label) == labels_to_storage_.end()) {
        RCLCPP_WARN(this->get_logger(), "Label '%s' not found in rules.", label.c_str());
        return result;
    }

    double best_dist = std::numeric_limits<double>::max();
    bool found = false;
    std::string selected_name;
    StorageInfo selected_info;

    const auto& candidates = labels_to_storage_.at(label);

    for (const auto& storage_name : candidates)
    {
        if (storage_map_.find(storage_name) == storage_map_.end()) continue;

        const auto& infos = storage_map_.at(storage_name);
        
        for (const auto& info : infos)
        {
            bool is_unlimited = (info.max_objects == -1);
            bool has_space = (info.actual_objects < info.max_objects);

            if (!is_unlimited && !has_space) {
                continue; 
            }

            double dx = info.pose.position.x - robot_pose.position.x;
            double dy = info.pose.position.y - robot_pose.position.y;
            double dz = info.pose.position.z - robot_pose.position.z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            if (dist < best_dist) {
                best_dist = dist;
                selected_name = storage_name;
                selected_info = info;
                found = true;
            }
        }
    }

    if (found) {
        result.success = true;
        result.storage_name = selected_name;
        result.pose = selected_info.pose;
        result.current_count = selected_info.actual_objects;
        result.max_count = selected_info.max_objects;
        
        auto lims = calculateLimits(selected_info.pose, selected_info.size_x, selected_info.size_y);
        result.limits = {lims.min_x, lims.max_x, lims.min_y, lims.max_y};
        
        RCLCPP_INFO(this->get_logger(), "Selected '%s' for item '%s'. Count: %d/%d", 
            selected_name.c_str(), label.c_str(), result.current_count, result.max_count);
    } else {
        RCLCPP_WARN(this->get_logger(), "No valid storage found for '%s'", label.c_str());
    }

    return result;
}

void StorageNode::incrementStorageCount(const std::string& storage_name, int amount)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (storage_map_.find(storage_name) != storage_map_.end()) {
        for (auto& info : storage_map_[storage_name]) {
            info.actual_objects += amount;
            if (info.actual_objects < 0) info.actual_objects = 0;
        }
        RCLCPP_INFO(this->get_logger(), "Updated count for '%s' by %d.", storage_name.c_str(), amount);
    }
}

void StorageNode::loadLabelToStorage(const std::string &yaml_file)
{
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        YAML::Node config = YAML::LoadFile(yaml_file);
        for (auto it = config.begin(); it != config.end(); ++it) {
            std::string group = it->first.as<std::string>();
            std::vector<std::string> targets;
            for (const auto &entry : it->second) {
                if (entry["storage"]) targets.push_back(entry["storage"].as<std::string>());
            }
            labels_to_storage_[group] = targets;
        }
    } catch (const YAML::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "YAML Label Error: %s", e.what());
    }
}

void StorageNode::loadStoragePoses(const std::string &yaml_file)
{
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        YAML::Node config = YAML::LoadFile(yaml_file);
        for (auto it = config.begin(); it != config.end(); ++it) {
            std::string name = it->first.as<std::string>();
            std::vector<StorageInfo> info_list;

            for (const auto &node : it->second) {
                StorageInfo info;
                
                if (node["position"]) {
                    info.pose.position.x = node["position"][0].as<double>();
                    info.pose.position.y = node["position"][1].as<double>();
                    info.pose.position.z = node["position"][2].as<double>();
                }
                
                if (node["orientation"]) {
                    tf2::Quaternion q;
                    q.setRPY(node["orientation"][0].as<double>(), 
                             node["orientation"][1].as<double>(), 
                             node["orientation"][2].as<double>());
                    info.pose.orientation = tf2::toMsg(q);
                } else {
                    info.pose.orientation.w = 1.0;
                }

                if (node["size"]) {
                    info.size_x = node["size"][0].as<double>();
                    info.size_y = node["size"][1].as<double>();
                } else {
                    info.size_x = 0.5; info.size_y = 0.5;
                }

                if (node["max_objects"]) {
                    info.max_objects = node["max_objects"].as<int>();
                }

                info_list.push_back(info);
            }
            storage_map_[name] = info_list;
        }
    } catch (const YAML::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "YAML Storage Error: %s", e.what());
    }
}

StorageNode::StorageLimits StorageNode::calculateLimits(const geometry_msgs::msg::Pose& pose, double sx, double sy)
{
    double yaw = tf2::getYaw(pose.orientation);
    double c = std::cos(yaw);
    double s = std::sin(yaw);
    double dx = sx/2.0; 
    double dy = sy/2.0;

    double lx[4] = {dx, dx, -dx, -dx};
    double ly[4] = {dy, -dy, dy, -dy};

    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();

    for(int i=0; i<4; i++) 
    {
        double gx = pose.position.x + (lx[i]*c - ly[i]*s);
        double gy = pose.position.y + (lx[i]*s + ly[i]*c);
        
        if (gx < min_x) { min_x = gx; }
        if (gx > max_x) { max_x = gx; }
        
        if (gy < min_y) { min_y = gy; }
        if (gy > max_y) { max_y = gy; }
    }
    return {min_x, max_x, min_y, max_y};
}


} // namespace storage_manager