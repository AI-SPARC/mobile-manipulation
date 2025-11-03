#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include <cmath>
#include <mutex>
#include <vector>
#include <algorithm> 
#include "object_manipulation_interfaces/srv/goal_reached.hpp"

class DijkstraController3D : public rclcpp::Node
{
public:
    DijkstraController3D() : Node("dijkstra_controller_3d"), 
                              current_waypoint_idx_(0),
                              path_received_(false),
                              executing_path_(false),
                              pose_initialized_(false)
    {
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&DijkstraController3D::odom_callback, this, std::placeholders::_1));
            
        vertex_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/path", 10, std::bind(&DijkstraController3D::vertex_callback, this, std::placeholders::_1));

        client_1 = this->create_client<object_manipulation_interfaces::srv::GoalReached>(
            "/goal_reached");

        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&DijkstraController3D::control_loop, this));

        linear_speed_ = 0.8;
        angular_speed_ = 2.0;
        waypoint_tolerance_ = 0.15; 
        angle_tolerance_ = 0.1;

        RCLCPP_INFO(this->get_logger(), "DijkstraController3D iniciado!");
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr vertex_sub_;
    rclcpp::Client<object_manipulation_interfaces::srv::GoalReached>::SharedPtr client_1;
    rclcpp::TimerBase::SharedPtr control_timer_;

    geometry_msgs::msg::Pose current_pose_;
    bool pose_initialized_;
    
    std::vector<geometry_msgs::msg::Pose> current_path_;
    std::mutex path_mutex_;
    size_t current_waypoint_idx_;
    bool path_received_;
    bool executing_path_;
    
    double linear_speed_;
    double angular_speed_;
    double waypoint_tolerance_;
    double angle_tolerance_;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        current_pose_ = msg->pose.pose;
        pose_initialized_ = true;
    }

    void vertex_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(path_mutex_);

        if (executing_path_) {
            RCLCPP_WARN(this->get_logger(), "Novo caminho ignorado. Execução em andamento.");
            return;
        }

        current_path_.clear();
        for (const auto& pose : msg->poses) {
            current_path_.push_back(pose);
        }

        if (!current_path_.empty()) {
            current_waypoint_idx_ = 0;
            path_received_ = true;
            executing_path_ = true;
            RCLCPP_INFO(this->get_logger(), "Novo caminho recebido com %zu pontos. Iniciando trajetória.", current_path_.size());
        }
    }

    void control_loop()
    {
        if (!pose_initialized_ || !path_received_ || !executing_path_) {
            publish_zero_velocity();
            return;
        }

        std::lock_guard<std::mutex> lock(path_mutex_);

        if (current_path_.empty() || current_waypoint_idx_ >= current_path_.size()) {
            publish_zero_velocity();
            executing_path_ = false;
            return;
        }

        const auto& target = current_path_[current_waypoint_idx_];
        double dx = target.position.x - current_pose_.position.x;
        double dy = target.position.y - current_pose_.position.y;
        double distance = std::sqrt(dx*dx + dy*dy);

        tf2::Quaternion q(current_pose_.orientation.x,
                          current_pose_.orientation.y,
                          current_pose_.orientation.z,
                          current_pose_.orientation.w);
        double yaw = get_yaw_from_quaternion(q);

        double target_yaw = std::atan2(dy, dx);
        double angle_error = normalize_angle(target_yaw - yaw);

        geometry_msgs::msg::Twist cmd;
        
        if (std::fabs(angle_error) > angle_tolerance_) 
        {
            cmd.linear.x = 0.0;
            cmd.angular.z = std::clamp(angle_error * 2.0, -angular_speed_, angular_speed_);
        } 
        else 
        {
            cmd.linear.x = std::min(linear_speed_, distance * 1.5); 
            cmd.linear.x = std::clamp(cmd.linear.x, 0.0, linear_speed_);
            cmd.angular.z = std::clamp(angle_error * 2.0, -angular_speed_, angular_speed_);
        }


        cmd.angular.z = -cmd.angular.z;

        cmd_vel_pub_->publish(cmd);

        if (distance < waypoint_tolerance_) {
            RCLCPP_INFO(this->get_logger(), "Reached waypoint %zu.", current_waypoint_idx_);
            current_waypoint_idx_++; 

            if (current_waypoint_idx_ >= current_path_.size()) {
                publish_zero_velocity();
                executing_path_ = false;
                path_received_ = false; 
                current_path_.clear();
                goal_reached();
                RCLCPP_INFO(this->get_logger(), "Goal Reached.");
            }
        }
    }

    void goal_reached()
    {
        auto request = std::make_shared<object_manipulation_interfaces::srv::GoalReached::Request>();
        request->reached = true;
        
      
        client_1->async_send_request(request,
            [this](rclcpp::Client<object_manipulation_interfaces::srv::GoalReached>::SharedFuture future_response) 
            {
                auto response = future_response.get();  

                if (response->success) 
                {
                   
                    RCLCPP_INFO(this->get_logger(), "Service executado com sucesso!");
                } 
                else 
                {
                    RCLCPP_WARN(this->get_logger(), "Falha ao executar service");
                }
            }
        );
    }

    void publish_zero_velocity()
    {
        geometry_msgs::msg::Twist stop;
        stop.linear.x = 0.0;
        stop.angular.z = 0.0;
        cmd_vel_pub_->publish(stop);
    }

    double get_yaw_from_quaternion(const tf2::Quaternion& q)
    {
        double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1.0 - 2.0 * (q.y()*q.y() + q.z()*q.z());
        return std::atan2(siny_cosp, cosy_cosp);
    }

    double normalize_angle(double a)
    {
        while(a > M_PI) a -= 2*M_PI;
        while(a < -M_PI) a += 2*M_PI;
        return a;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DijkstraController3D>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}