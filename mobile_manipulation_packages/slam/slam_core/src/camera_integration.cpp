#include "slam_core/CameraIntegration.hpp"
#include <cmath>

namespace slam_core
{

CameraIntegration::CameraIntegration(const rclcpp::NodeOptions & options)
: Node("camera_integration", options)
{
    num_cameras_ = this->declare_parameter("num_cameras", 1);
    sync_timeout_sec_ = this->declare_parameter("sync_timeout_sec", 0.05); 
    
    std::string robot_ns = this->declare_parameter<std::string>("robot_namespace", "robot_0");

    camera_buffers_.resize(num_cameras_);

    RCLCPP_INFO(this->get_logger(), "[%s] Iniciando agregador otimizado para %d camera(s).", robot_ns.c_str(), num_cameras_);

    for (int i = 0; i < num_cameras_; i++) 
    {
        CameraSubs cam;
        cam.cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        
        rclcpp::SubscriptionOptions sub_opt;
        sub_opt.callback_group = cam.cb_group;

        std::string ns = "/" + robot_ns + "/camera_" + std::to_string(i);

        cam.rgb_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, ns + "/rgb/image_raw", rmw_qos_profile_sensor_data, sub_opt);
            
        cam.depth_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, ns + "/depth/image_rect_raw", rmw_qos_profile_sensor_data, sub_opt);
            
        cam.info_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>>(
            this, ns + "/depth/camera_info", rmw_qos_profile_sensor_data, sub_opt);

        cam.sync = std::make_shared<Synchronizer>(
            SyncPolicy(10), *cam.rgb_sub, *cam.depth_sub, *cam.info_sub);
        
        cam.sync->registerCallback(
            std::bind(&CameraIntegration::camera_callback, this, 
                      std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, i));

        cameras_.push_back(cam);
        RCLCPP_INFO(this->get_logger(), "Pronto para sincronizar: %s", ns.c_str());
    }
}

void CameraIntegration::camera_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg,
    int cam_id)
{
    std::lock_guard<std::mutex> lock(sync_mutex_);

    auto& buffer = camera_buffers_[cam_id];

    buffer.push_back({rgb_msg, depth_msg, info_msg, rgb_msg->header.stamp});

    if (buffer.size() > MAX_BUFFER_SIZE) 
    {
        buffer.pop_front();
    }
}

bool CameraIntegration::get_latest_frame(int cam_id, CameraData& out_data)
{
    std::lock_guard<std::mutex> lock(sync_mutex_);
    
    auto& buffer = camera_buffers_[cam_id];
    if (buffer.empty()) 
    {

        return false;
    }

    out_data = buffer.back();
    
    buffer.clear(); 
    
    return true;
}

bool CameraIntegration::retrieve_frame_by_timestamp(
    int cam_id, 
    const rclcpp::Time& target_time, 
    CameraData& out_data, 
    double tolerance_sec)
{
    std::lock_guard<std::mutex> lock(sync_mutex_);

    auto& buffer = camera_buffers_[cam_id];
    if (buffer.empty()) return false;

    int64_t target_ns = target_time.nanoseconds();
    int64_t tolerance_ns = static_cast<int64_t>(tolerance_sec * 1e9);

    int64_t best_diff = INT64_MAX;
    auto best_it = buffer.end();

  
    for (auto it = buffer.begin(); it != buffer.end(); ++it) 
    {
        int64_t diff = std::abs(it->stamp.nanoseconds() - target_ns);
        if (diff < best_diff) 
        {
            best_diff = diff;
            best_it = it;
        }
    }

    if (best_it != buffer.end() && best_diff <= tolerance_ns) 
    {
        out_data = *best_it;
        
       
        buffer.erase(buffer.begin(), best_it + 1); 
        
        return true;
    }

    return false;
}

} // namespace slam_core

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(slam_core::CameraIntegration)