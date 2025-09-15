#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

class CameraInfoFilter : public rclcpp::Node
{
public:
    CameraInfoFilter()
    : Node("camera_info_filter")
    {
        RCLCPP_INFO(this->get_logger(), "Nó iniciado, filtrando depth_optical_frame_2 256x256");

        // Subscrição ao tópico original
        subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info",  // Substitua pelo tópico correto
            10,
            std::bind(&CameraInfoFilter::cameraInfoCallback, this, std::placeholders::_1)
        );

        // Publicador para o novo tópico
        publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            "/depth_camera_1/camera_info",
            10
        );
    }

private:
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        // Filtra por frame_id e resolução
        if (msg->header.frame_id == "depth_optical_frame_2" &&
            msg->width == 256 &&
            msg->height == 256)
        {
            RCLCPP_INFO(this->get_logger(), "Mensagem válida recebida. Publicando em /depth_camera_1/camera_info");

            // Publica a mensagem filtrada
            publisher_->publish(*msg);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraInfoFilter>());
    rclcpp::shutdown();
    return 0;
}
