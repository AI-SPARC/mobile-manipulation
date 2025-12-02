#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <deque>
#include <mutex>

#include <manipulation/IsGripperHolding.hpp>

using namespace std::chrono_literals;

namespace manipulation {


IsGripperHolding::IsGripperHolding()
: Node("gripper_monitor_node") 
{
    RCLCPP_INFO(this->get_logger(), "Gripper Monitor Node inicializado. Monitorando 'contact_sensor'.");

    subscription_ = this->create_subscription<std_msgs::msg::Float32>(
        "contact_sensor", 10, 
        std::bind(&IsGripperHolding::topic_callback, this, std::placeholders::_1)
    );
}   


bool IsGripperHolding::checkIsHolding() 
{
    std::lock_guard<std::mutex> lock(contact_sensor_mutex_);
    int contador = 0;

    for(size_t i = 0; i < contact_sensor_data_.size(); i++)
    {
        if(contact_sensor_data_[i] > 0.5)
        {
            contador++;
        }
    }

    if(contador >= 9)
    {
        return true;
    }

    
    return false;
}


void IsGripperHolding::topic_callback(const std_msgs::msg::Float32 & msg)
{
    std::lock_guard<std::mutex> lock(contact_sensor_mutex_); 

    if (contact_sensor_data_.size() > 10) 
    {
        contact_sensor_data_.pop_front();
    }


    contact_sensor_data_.push_back(msg.data);
}

} // namespace manipulation