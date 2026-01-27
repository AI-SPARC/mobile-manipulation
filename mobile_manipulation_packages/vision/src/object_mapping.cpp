#include "vision/ObjectMapping.hpp"
#include <cmath>
#include <algorithm>
#include <rclcpp_components/register_node_macro.hpp>

namespace vision
{

ObjectMapping::ObjectMapping(const rclcpp::NodeOptions & options)
 : Node("object_mapping", options),
   is_robot_stopped_(false)
{
    
    this->declare_parameter<double>("velocity_threshold", 0.25); 
    this->declare_parameter<double>("settlement_duration", 0.25); 
    this->declare_parameter<double>("voxel_leaf_size", 0.0025); 
    this->declare_parameter<double>("octomap_resolution", 0.0075);
    this->declare_parameter<bool>("publish_octomap_to_moveit", true);
    this->declare_parameter<double>("surrounding_distance_threshold", 0.3);

    
    velocity_threshold_ = this->get_parameter("velocity_threshold").as_double();
    settlement_duration_ = this->get_parameter("settlement_duration").as_double();
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
    octomap_resolution_ = this->get_parameter("octomap_resolution").as_double();
    publish_octomap_to_moveit_ = this->get_parameter("publish_octomap_to_moveit").as_bool();
    surrounding_distance_threshold_ = this->get_parameter("surrounding_distance_threshold").as_double();
    
    last_motion_time_ = this->now();

    
    sub_joint_states_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/isaac_joint_states", 
        rclcpp::SensorDataQoS(), 
        std::bind(&ObjectMapping::jointStatesCallback, this, std::placeholders::_1)
    );

    sub_semantic_pcl_ = this->create_subscription<mobile_manipulation_interfaces::msg::SemanticPcl>(
        "/semantic_pcl_array", 
        10, 
        std::bind(&ObjectMapping::semanticPclCallback, this, std::placeholders::_1)
    );

    pub_accumulated_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/mapped_object", 
       10
    );

    pub_planning_scene_ = this->create_publisher<moveit_msgs::msg::PlanningScene>(
        "/planning_scene", 
        10
    );

    pub_semantic_environment_ = this->create_publisher<mobile_manipulation_interfaces::msg::SemanticPcl>(
        "/mapped_environment_semantic", 
        10
    );

    pub_environment_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/mapped_environment_cloud", 
        10
    );

    RCLCPP_INFO(
        this->get_logger(), 
        "ObjectMapping iniciado. Voxel: %.3fm | Raio Ambiente: %.2fm", 
        voxel_leaf_size_, 
        surrounding_distance_threshold_
    );
}

void ObjectMapping::ObjectToMap(std::string id)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    object_to_map_ = id;
    RCLCPP_INFO(this->get_logger(), "Alvo definido para mapeamento: %s", id.c_str());
}

std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> 
ObjectMapping::getObjectAndEnvironment(const std::string& object_id)
{
    
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr env_copy(new pcl::PointCloud<pcl::PointXYZ>);

    if (object_map_.find(object_id) != object_map_.end())
    {
        
        *cloud_copy = *object_map_[object_id].cloud;
        *env_copy = *object_map_[object_id].environment;
    }
    

    return std::make_pair(cloud_copy, env_copy);
}

void ObjectMapping::jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    bool currently_moving = false;

    
    if (msg->velocity.size() == msg->name.size()) 
    {
        for (double vel : msg->velocity) 
        {
            if (std::abs(vel) > velocity_threshold_) 
            {
                currently_moving = true;
                break;
            }
        }
    }
    else 
    { 
        return; 
    }

    if (currently_moving) 
    {
        last_motion_time_ = this->now();
        is_robot_stopped_ = false;
    } 
    else 
    {
        auto time_since_stop = this->now() - last_motion_time_;
        if (time_since_stop.seconds() >= settlement_duration_) 
        {
            is_robot_stopped_ = true;
        }
    }
}

void ObjectMapping::semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg) 
{
    
    publishAccumulatedCloud();

    if (!is_robot_stopped_)
    {
        return;
    }

    
    std::string current_target;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (object_to_map_.empty())
        {
            return; 
        }
        current_target = object_to_map_;
    }
    
   
    if (msg->labels.size() != msg->clouds.size() || msg->labels.size() != msg->poses.size()) 
    {
        RCLCPP_WARN(this->get_logger(), "ERRO: Tamanhos incompatíveis no SemanticPcl recebido!");
        return;
    }

    bool map_updated = false;

    
    for (size_t i = 0; i < msg->labels.size(); ++i) 
    {
        if (msg->labels[i] == current_target)
        {
            bool updated = processTargetCloud(msg->labels[i], msg->clouds[i], msg->poses[i]);
            if (updated)
            {
                map_updated = true;
            }
        }
    }

    
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (object_map_.find(current_target) != object_map_.end())
        {
            if (map_updated && publish_octomap_to_moveit_) 
            {
                publishToPlanningScene();
            }
        }
    }
    
    
    processEnvironmentClouds(msg, current_target);
}

bool ObjectMapping::processTargetCloud(
    const std::string& label,
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    const geometry_msgs::msg::Pose& pose_msg)
{
    pcl::PointCloud<pcl::PointXYZ> incoming_cloud;
    pcl::fromROSMsg(cloud_msg, incoming_cloud);

    if (incoming_cloud.empty())
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (object_map_.find(label) == object_map_.end()) 
    {
        MappingObjectData new_data;
        *new_data.cloud = incoming_cloud;
        new_data.pose = pose_msg;
        
        object_map_[label] = new_data;
        return true;
    } 
    else 
    {
        MappingObjectData & stored_data = object_map_[label];

        Eigen::Affine3d tf_old;
        Eigen::Affine3d tf_new;
        tf2::fromMsg(stored_data.pose, tf_old);
        tf2::fromMsg(pose_msg, tf_new);

        
        Eigen::Affine3d tf_correction = tf_new * tf_old.inverse();
       
        pcl::transformPointCloud(*stored_data.cloud, *stored_data.cloud, tf_correction);
        
        
        *stored_data.cloud += incoming_cloud;
        stored_data.pose = pose_msg;

        
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(stored_data.cloud);
        sor.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        sor.filter(*cloud_filtered);
        
        stored_data.cloud = cloud_filtered;
        return true;
    }
}

void ObjectMapping::processEnvironmentClouds(
    const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr & input_msg, 
    const std::string& target_label)
{
    std::lock_guard<std::mutex> lock(data_mutex_);


    if (object_map_.find(target_label) == object_map_.end())
    {
        return;
    }
    
    MappingObjectData& target_data = object_map_.at(target_label);
    
    if (target_data.cloud->empty())
    {
        return;
    }

    
    target_data.environment->clear();

    
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_target;
    kdtree_target.setInputCloud(target_data.cloud);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

   
    mobile_manipulation_interfaces::msg::SemanticPcl output_semantic_msg;
    output_semantic_msg.header.frame_id = "world";
    output_semantic_msg.header.stamp = this->now();

    
    sensor_msgs::msg::PointCloud2 target_cloud_msg;
    pcl::toROSMsg(*target_data.cloud, target_cloud_msg);
    target_cloud_msg.header.frame_id = "world";
    target_cloud_msg.header.stamp = output_semantic_msg.header.stamp;

    output_semantic_msg.labels.push_back(target_label);
    output_semantic_msg.poses.push_back(target_data.pose);
    output_semantic_msg.clouds.push_back(target_cloud_msg);

    
    for (size_t i = 0; i < input_msg->labels.size(); ++i) 
    {
        std::string current_env_label = input_msg->labels[i];
        
        if (current_env_label == target_label)
        {
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr env_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(input_msg->clouds[i], *env_cloud_ptr);

        if (env_cloud_ptr->empty())
        {
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_env_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        
        Eigen::Vector4f min_t;
        Eigen::Vector4f max_t;
        Eigen::Vector4f min_e;
        Eigen::Vector4f max_e;
        
        pcl::getMinMax3D(*target_data.cloud, min_t, max_t);
        pcl::getMinMax3D(*env_cloud_ptr, min_e, max_e);
        
        double dx = std::max({0.0f, min_t[0] - max_e[0], min_e[0] - max_t[0]});
        double dy = std::max({0.0f, min_t[1] - max_e[1], min_e[1] - max_t[1]});
        double dz = std::max({0.0f, min_t[2] - max_e[2], min_e[2] - max_t[2]});

        double dist_sq = dx*dx + dy*dy + dz*dz;
        double thresh_sq = surrounding_distance_threshold_ * surrounding_distance_threshold_;

        if (dist_sq > thresh_sq) 
        {
            continue; 
        }

        
        for (const auto& pt : *env_cloud_ptr)
        {
            if (kdtree_target.radiusSearch(pt, surrounding_distance_threshold_, pointIdxRadiusSearch, pointRadiusSquaredDistance, 1) > 0)
            {
                filtered_env_cloud->points.push_back(pt);
            }
        }

        if (!filtered_env_cloud->empty())
        {
            filtered_env_cloud->width = filtered_env_cloud->points.size();
            filtered_env_cloud->height = 1;
            filtered_env_cloud->is_dense = true;

           
            *target_data.environment += *filtered_env_cloud;

           
            sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
            pcl::toROSMsg(*filtered_env_cloud, filtered_cloud_msg);
            filtered_cloud_msg.header = input_msg->clouds[i].header; 
            filtered_cloud_msg.header.frame_id = "world";

            output_semantic_msg.labels.push_back(current_env_label);
            output_semantic_msg.poses.push_back(input_msg->poses[i]);
            output_semantic_msg.clouds.push_back(filtered_cloud_msg);
        }
    }

    
    pcl::PointCloud<pcl::PointXYZRGB> combined_visual_cloud;

    
    for (const auto& pt : *target_data.cloud) 
    {
        pcl::PointXYZRGB p;
        p.x = pt.x; 
        p.y = pt.y; 
        p.z = pt.z;
        p.r = 0; 
        p.g = 255; 
        p.b = 0; 
        combined_visual_cloud.push_back(p);
    }

   
    if (!target_data.environment->empty())
    {
        for (const auto& pt : *target_data.environment)
        {
            pcl::PointXYZRGB p_vis;
            p_vis.x = pt.x; 
            p_vis.y = pt.y; 
            p_vis.z = pt.z;
            p_vis.r = 255; 
            p_vis.g = 0; 
            p_vis.b = 0; 
            combined_visual_cloud.push_back(p_vis);
        }
    }
    
    
    publishEnvironmentVisualization(output_semantic_msg, combined_visual_cloud);
}

void ObjectMapping::publishEnvironmentVisualization(
    const mobile_manipulation_interfaces::msg::SemanticPcl& semantic_msg,
    const pcl::PointCloud<pcl::PointXYZRGB>& visual_cloud)
{
    
    if (!semantic_msg.labels.empty()) 
    {
        pub_semantic_environment_->publish(semantic_msg);
    }

    
    if (!visual_cloud.empty()) 
    {
        sensor_msgs::msg::PointCloud2 visual_msg;
        pcl::toROSMsg(visual_cloud, visual_msg);
        visual_msg.header.frame_id = "world";
        visual_msg.header.stamp = this->now(); 

        pub_environment_cloud_->publish(visual_msg);
    }
}

void ObjectMapping::publishAccumulatedCloud()
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    pcl::PointCloud<pcl::PointXYZRGB> display_cloud;

    for (const auto& [label, data] : object_map_) 
    {
        if (data.cloud->empty())
        {
            continue;
        }
        
    
        std::size_t hash = std::hash<std::string>{}(label);
        uint8_t r = (hash >> 16) & 0xFF;
        uint8_t g = (hash >> 8)  & 0xFF;
        uint8_t b = (hash)       & 0xFF;

        for (const auto& pt : *data.cloud) 
        {
            pcl::PointXYZRGB pt_rgb;
            pt_rgb.x = pt.x; 
            pt_rgb.y = pt.y; 
            pt_rgb.z = pt.z;
            pt_rgb.r = r; 
            pt_rgb.g = g; 
            pt_rgb.b = b;
            display_cloud.push_back(pt_rgb);
        }
    }

    if (display_cloud.empty())
    {
        return;
    }

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(display_cloud, output_msg);
    output_msg.header.frame_id = "world"; 
    output_msg.header.stamp = this->now();
    pub_accumulated_cloud_->publish(output_msg);
}

void ObjectMapping::publishToPlanningScene()
{
    bool has_points = false;
    for (const auto& [label, data] : object_map_) 
    {
        if (!data.cloud->empty()) 
        { 
            has_points = true; 
            break; 
        }
    }

    if (!has_points)
    {
        return;
    }

    std::shared_ptr<octomap::OcTree> tree(new octomap::OcTree(octomap_resolution_));

    for (const auto& [label, data] : object_map_) 
    {
        if (data.cloud->empty())
        {
            continue;
        }
        for (const auto& pt : *data.cloud) 
        {
            tree->updateNode(octomap::point3d(pt.x, pt.y, pt.z), true);
        }
    }

    moveit_msgs::msg::PlanningScene scene_msg;
    scene_msg.is_diff = true; 

    octomap_msgs::msg::Octomap octomap_msg;

    {
        
        std::streambuf* orig_buf = std::cout.rdbuf();
        std::stringstream dummy_stream;
        std::cout.rdbuf(dummy_stream.rdbuf());
        
        octomap_msgs::binaryMapToMsg(*tree, octomap_msg);
        
        std::cout.rdbuf(orig_buf);
    }
    
    octomap_msg.header.frame_id = "world";
    octomap_msg.header.stamp = this->now();

    scene_msg.world.octomap.header = octomap_msg.header;
    scene_msg.world.octomap.octomap = octomap_msg;
    
    pub_planning_scene_->publish(scene_msg);
}

} // namespace vision

RCLCPP_COMPONENTS_REGISTER_NODE(vision::ObjectMapping)