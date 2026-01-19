#include "vision/ObjectMapping.hpp"
#include <cmath>
#include <rclcpp_components/register_node_macro.hpp>


namespace vision {

ObjectMapping::ObjectMapping(const rclcpp::NodeOptions & options)
 : Node("object_mapping", options),
   is_robot_stopped_(false)
{
    
    this->declare_parameter<double>("velocity_threshold", 0.3); 
    this->declare_parameter<double>("settlement_duration", 0.35); 
    
    velocity_threshold_ = this->get_parameter("velocity_threshold").as_double();
    settlement_duration_ = this->get_parameter("settlement_duration").as_double();
    
    last_motion_time_ = this->now();

    
    sub_joint_states_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/isaac_joint_states", 
        rclcpp::SensorDataQoS(), 
        std::bind(&ObjectMapping::jointStatesCallback, this, std::placeholders::_1)
    );

    sub_semantic_pcl_ = this->create_subscription<mobile_manipulation_interfaces::msg::SemanticPcl>(
        "/semantic_pcl_array", 10, std::bind(&ObjectMapping::semanticPclCallback, this, std::placeholders::_1));

    
    pub_accumulated_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/mapped_object", 10);

    RCLCPP_INFO(this->get_logger(), "ObjectMapping iniciado. Aguardando estabilização (< %.2f rad/s por %.2fs)", 
        velocity_threshold_, settlement_duration_);
}

void ObjectMapping::jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    bool currently_moving = false;

    if (msg->velocity.size() == msg->name.size()) 
    {
        for (double vel : msg->velocity) {
            if (std::abs(vel) > velocity_threshold_) {
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

void ObjectMapping::ObjectToMap(std::string id)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    object_to_map_ = id;
}

void ObjectMapping::semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg) 
{
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
        RCLCPP_WARN(this->get_logger(), "Tamanhos dos arrays na mensagem SemanticPcl não batem!");
        return;
    }

    bool map_updated = false;

    for (size_t i = 0; i < msg->labels.size(); ++i) 
    {
        std::string label = msg->labels[i];

        
        if (label != current_target) 
        {
            continue;
        }    

        
        const auto & pose = msg->poses[i];
        RCLCPP_INFO(this->get_logger(), 
            "Objeto '%s' encontrado. Pose -> Pos: [x: %.3f, y: %.3f, z: %.3f] | Ori: [w: %.3f, x: %.3f, y: %.3f, z: %.3f]",
            label.c_str(),
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
        );
        
        pcl::PointCloud<pcl::PointXYZ> incoming_cloud;
        pcl::fromROSMsg(msg->clouds[i], incoming_cloud);

        if (incoming_cloud.empty()) continue;
        
        
        if (object_points_.find(label) == object_points_.end()) 
        {
            object_points_[label] = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(incoming_cloud);
        } 
        else 
        {
            *object_points_[label] += incoming_cloud;
        }
    
        map_updated = true;
    }

    if (map_updated) 
    {
        publishAccumulatedCloud();
    }
}

void ObjectMapping::publishAccumulatedCloud()
{
    pcl::PointCloud<pcl::PointXYZRGB> display_cloud;

    for (const auto& [label, cloud_ptr] : object_points_) 
    {
        if (cloud_ptr->empty()) continue;

        
        std::size_t hash = std::hash<std::string>{}(label);
        uint8_t r = (hash >> 16) & 0xFF;
        uint8_t g = (hash >> 8)  & 0xFF;
        uint8_t b = (hash)       & 0xFF;

        for (const auto& pt : *cloud_ptr) 
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

    if (display_cloud.empty()) return;

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(display_cloud, output_msg);

    output_msg.header.frame_id = "world"; 
    output_msg.header.stamp = this->now();

    pub_accumulated_cloud_->publish(output_msg);
}

} // namespace vision

RCLCPP_COMPONENTS_REGISTER_NODE(vision::ObjectMapping)