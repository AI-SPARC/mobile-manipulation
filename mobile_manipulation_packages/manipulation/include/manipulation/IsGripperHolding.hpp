#ifndef IS_GRIPPER_HOLDING_HPP
#define IS_GRIPPER_HOLDING_HPP

#include <memory>
#include <cmath>
#include <string>
#include <chrono>
#include <deque>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

namespace manipulation {

class IsGripperHolding : public rclcpp::Node 
{
public:
    IsGripperHolding();
    bool checkIsHolding(); 

private:
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription_;
    
    std::deque<float> contact_sensor_data_;
    std::mutex contact_sensor_mutex_;
    
    const size_t MAX_SAMPLES = 10;
    const float PRESSURE_THRESHOLD = 0.1; 

    void topic_callback(const std_msgs::msg::Float32 & msg);
};

} // namespace manipulation

#endif // IS_GRIPPER_HOLDING_HPP