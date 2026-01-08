#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // ESSENCIAL para transformar poses

using std::placeholders::_1;

class OdomWorldTransformer : public rclcpp::Node
{
public:
    OdomWorldTransformer() : Node("odom_world_transformer")
    {
        // Configurações
        target_frame_ = "world";  // Frame de destino desejado
        std::string input_topic = "robot_04/odom"; // Seu tópico de odometria

        // Inicializa o Buffer e Listener do TF2
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Inscreve no tópico de odometria
        subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            input_topic, 10, std::bind(&OdomWorldTransformer::topic_callback, this, _1));

        RCLCPP_INFO(this->get_logger(), "Nó iniciado. Transformando '%s' -> '%s'", input_topic.c_str(), target_frame_.c_str());
    }

private:
    void topic_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // 1. Converter a Odometria recebida para um PoseStamped
        // O TF2 precisa de um cabeçalho (header) com timestamp e frame_id para saber ONDE e QUANDO transformar
        geometry_msgs::msg::PoseStamped pose_in;
        pose_in.header = msg->header; // Mantém o frame original ("odom") e o tempo (sec: 81...)
        pose_in.pose = msg->pose.pose;

        geometry_msgs::msg::PoseStamped pose_out;

        try {
            // 2. Tentar transformar para o frame 'world'
            // O timeout de 0.1s ajuda a aguardar caso o TF chegue um pouco atrasado
            if (tf_buffer_->canTransform(target_frame_, msg->header.frame_id, tf2::TimePointZero)) {
                
                pose_out = tf_buffer_->transform(pose_in, target_frame_);

                // 3. Printar o resultado transformado
                RCLCPP_INFO(this->get_logger(), 
                    "\n--- POSE NO FRAME WORLD ---\n"
                    "Position: [x: %.5f, y: %.5f, z: %.5f]\n"
                    "Orientation: [x: %.5f, y: %.5f, z: %.5f, w: %.5f]",
                    pose_out.pose.position.x,
                    pose_out.pose.position.y,
                    pose_out.pose.position.z,
                    pose_out.pose.orientation.x,
                    pose_out.pose.orientation.y,
                    pose_out.pose.orientation.z,
                    pose_out.pose.orientation.w
                );
            } else {
                // Throttle para não spammar o terminal se o TF não existir
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                    "Aguardando Transformação: %s -> %s", msg->header.frame_id.c_str(), target_frame_.c_str());
            }

        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "Erro na transformação: %s", ex.what());
        }
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::string target_frame_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdomWorldTransformer>());
    rclcpp::shutdown();
    return 0;
}