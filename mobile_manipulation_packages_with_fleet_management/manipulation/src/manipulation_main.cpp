#include <memory>

#include "rclcpp/rclcpp.hpp"
#include <manipulation/SimpleManipulation.hpp>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<manipulation::SimpleManipulation>());
  rclcpp::shutdown();
  return 0;
}