#ifndef MANIPULATION__IS_GRIPPER_HOLDING_HPP_
#define MANIPULATION__IS_GRIPPER_HOLDING_HPP_

#include <memory>
#include <vector>
#include <deque>
#include <mutex>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

namespace manipulation {

class IsGripperHolding : public rclcpp::Node
{
public:
    explicit IsGripperHolding(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    
    ~IsGripperHolding() override = default;

    bool checkIsHolding();

private:
    void topic_callback(const std_msgs::msg::Float32 & msg);

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription_;
    std::deque<float> contact_sensor_data_;
    std::mutex contact_sensor_mutex_;
};

} // namespace manipulation

#endif // MANIPULATION__IS_GRIPPER_HOLDING_HPP_