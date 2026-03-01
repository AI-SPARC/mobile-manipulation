#ifndef SLAM_CORE_CAMERA_INTEGRATION_HPP_
#define SLAM_CORE_CAMERA_INTEGRATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <deque>
#include <vector>
#include <mutex>
#include <memory>
#include <string>

namespace slam_core
{

// Estrutura leve para guardar os ponteiros das mensagens sincronizadas
struct CameraData 
{
    sensor_msgs::msg::Image::ConstSharedPtr rgb;
    sensor_msgs::msg::Image::ConstSharedPtr depth;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info;
    rclcpp::Time stamp;
};

class CameraIntegration : public rclcpp::Node
{
public:
    explicit CameraIntegration(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

   
    bool retrieve_frame_by_timestamp(
        int cam_id, 
        const rclcpp::Time& target_time, 
        CameraData& out_data, 
        double tolerance_sec = 0.05);

    bool get_latest_frame(int cam_id, CameraData& out_data);

private:

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, 
        sensor_msgs::msg::Image, 
        sensor_msgs::msg::CameraInfo> SyncPolicy;
        
    typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;

    struct CameraSubs 
    {
        rclcpp::CallbackGroup::SharedPtr cb_group;
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub;
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub;
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CameraInfo>> info_sub;
        std::shared_ptr<Synchronizer> sync;
    };

    int num_cameras_;
    double sync_timeout_sec_;

    std::vector<CameraSubs> cameras_;

    
    std::vector<std::deque<CameraData>> camera_buffers_;
    
    
    std::mutex sync_mutex_;
    
    const size_t MAX_BUFFER_SIZE = 15; 

    void camera_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg,
        int cam_id);
};

} // namespace slam_core

#endif // SLAM_CORE_CAMERA_INTEGRATION_HPP_