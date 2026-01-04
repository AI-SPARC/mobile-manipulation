#include <manipulation/IsGripperHolding.hpp>
#include "rclcpp_components/register_node_macro.hpp"

using namespace std::chrono_literals;

namespace manipulation {

IsGripperHolding::IsGripperHolding(const rclcpp::NodeOptions & options)
: Node("gripper_monitor_node", options) 
{
    RCLCPP_INFO(this->get_logger(), "Gripper Monitor Node inicializado (Composable).");

    subscription_ = this->create_subscription<std_msgs::msg::Float32>(
        "contact_sensor", 10, 
        std::bind(&IsGripperHolding::topic_callback, this, std::placeholders::_1)
    );
}   

bool IsGripperHolding::checkIsHolding() 
{
    std::lock_guard<std::mutex> lock(contact_sensor_mutex_);
    int contador = 0;

    if (contact_sensor_data_.empty()) 
    {
        return false;
    }

    for(size_t i = 0; i < contact_sensor_data_.size(); i++)
    {
        if(contact_sensor_data_[i] > 0.1)
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

    if (contact_sensor_data_.size() >= 10) 
    {
        contact_sensor_data_.pop_front();
    }

    contact_sensor_data_.push_back(msg.data);
}

} // namespace manipulation

RCLCPP_COMPONENTS_REGISTER_NODE(manipulation::IsGripperHolding)