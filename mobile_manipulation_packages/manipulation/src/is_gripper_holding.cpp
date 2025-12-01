#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <deque>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

using namespace std::chrono_literals;

class IsGripperHolding : public rclcpp::Node 
{
public:
    IsGripperHolding()
     : Node("gripper_monitor_node") 
    {
        RCLCPP_INFO(this->get_logger(), "Gripper Monitor Node inicializado. Monitorando 'contact_sensor'.");

        subscription_ = this->create_subscription<std_msgs::msg::Float32>(
            "contact_sensor", 10, 
            std::bind(&IsGripperHolding::topic_callback, this, std::placeholders::_1)
        );
    }   


    bool checkIsHolding() 
    {
        std::lock_guard<std::mutex> lock(contact_sensor_mutex_);

        if (contact_sensor_data_.empty()) 
        {
            return false; 
        }


        float latest_pressure = contact_sensor_data_.back();

        bool holding = latest_pressure >= PRESSURE_THRESHOLD;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, "Check Holding Status: %.3f. Result: %s", latest_pressure, holding ? "TRUE" : "FALSE");
        
        return holding;
    }

private:
    // Subscribers.
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription_;
    
    std::deque<float> contact_sensor_data_;
    std::mutex contact_sensor_mutex_;
    
    const size_t MAX_SAMPLES = 10;
    const float PRESSURE_THRESHOLD = 0.1; 


    void topic_callback(const std_msgs::msg::Float32 & msg)
    {
        std::lock_guard<std::mutex> lock(contact_sensor_mutex_); 

        if (contact_sensor_data_.size() >= MAX_SAMPLES) 
        {
            contact_sensor_data_.pop_front();
        }

        contact_sensor_data_.push_back(msg.data);
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  // Cria o nó e o executa em um loop síncrono.
  rclcpp::spin(std::make_shared<IsGripperHolding>()); 
  rclcpp::shutdown();
  return 0;
}