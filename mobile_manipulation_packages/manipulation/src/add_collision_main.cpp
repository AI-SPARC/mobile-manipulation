#include "rclcpp/rclcpp.hpp"
#include <manipulation/AddCollision.hpp>

int main(int argc, char **argv) 
{
    rclcpp::init(argc, argv);
    
    rclcpp::spin(std::make_shared<manipulation::AddCollision>());
    
    rclcpp::shutdown();
    return 0;
}