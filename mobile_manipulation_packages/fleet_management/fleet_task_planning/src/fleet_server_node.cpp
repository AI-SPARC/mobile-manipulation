#include <memory>
#include <chrono>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // Header CRÍTICO para a mágica do .transform()

using std::placeholders::_1;

class OdomWorldTransformer : public rclcpp::Node
{
public:
    OdomWorldTransformer() : Node("odom_world_transformer")
    {
        // Parâmetros para flexibilidade
        this->declare_parameter<std::string>("target_frame", "world");
        this->declare_parameter<std::string>("odom_topic", "robot_04/odom");

        target_frame_ = this->get_parameter("target_frame").as_string();
        std::string odom_topic = this->get_parameter("odom_topic").as_string();

        // Configuração do TF2
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Subscriber
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, 10, std::bind(&OdomWorldTransformer::odom_callback, this, _1));

        RCLCPP_INFO(this->get_logger(), "Odom Transformer iniciado. Convertendo %s -> %s", 
            odom_topic.c_str(), target_frame_.c_str());
    }

private:
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // 1. Criar um PoseStamped a partir da Odometria recebida
        // O TF2 trabalha melhor com Stamped types para alinhar o tempo
        geometry_msgs::msg::PoseStamped pose_in;
        pose_in.header = msg->header; // Copia frame_id ("odom") e stamp
        pose_in.pose = msg->pose.pose;

        geometry_msgs::msg::PoseStamped pose_out;

        try {
            // 2. Verificar se a transformação está disponível
            if (tf_buffer_->canTransform(target_frame_, msg->header.frame_id, tf2::TimePointZero)) {
                
                // 3. Realizar a transformação
                // A função 'transform' lida com a matemática de quaterniões e matrizes automaticamente
                pose_out = tf_buffer_->transform(pose_in, target_frame_);

                // 4. Print solicitado
                RCLCPP_INFO(this->get_logger(), 
                    "\n>>> [WORLD FRAME] Position: [x: %.6f, y: %.6f, z: %.6f] | Orientation (quat): [x: %.6f, y: %.6f, z: %.6f, w: %.6f]",
                    pose_out.pose.position.x,
                    pose_out.pose.position.y,
                    pose_out.pose.position.z,
                    pose_out.pose.orientation.x,
                    pose_out.pose.orientation.y,
                    pose_out.pose.orientation.z,
                    pose_out.pose.orientation.w
                );
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                    "Aguardando transformacao de '%s' para '%s'...", 
                    msg->header.frame_id.c_str(), target_frame_.c_str());
            }

        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "Falha na transformacao TF: %s", ex.what());
        }
    }

    std::string target_frame_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdomWorldTransformer>());
    rclcpp::shutdown();
    return 0;
}