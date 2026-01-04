#include <memory>

#include "rclcpp/rclcpp.hpp"
#include <manipulation/IsGripperHolding.hpp>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  
  rclcpp::spin(std::make_shared<manipulation::IsGripperHolding>()); 
  
  rclcpp::shutdown();
  return 0;
}