#include <rclcpp/rclcpp.hpp>
#include "vision/GenerateGraspPoses.hpp" 

int main(int argc, char * argv[])
{
    
    rclcpp::init(argc, argv);

    
    rclcpp::NodeOptions options;

    
    auto node = std::make_shared<vision::GenerateGraspPoses>(options);

    
    rclcpp::spin(node);

    
    rclcpp::shutdown();

    return 0;
}