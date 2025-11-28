#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include <cmath>
#include <mutex>
#include <vector>
#include <algorithm> 
#include <thread>
#include <chrono>
#include "rclcpp_action/rclcpp_action.hpp"
#include "mobile_manipulation_interfaces/action/controller.hpp"

class RobotController : public rclcpp::Node
{
public:
    RobotController() : Node("controller"), 
                              pose_initialized_(false)
    {
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&RobotController::odom_callback, this, std::placeholders::_1));
            
        this->action_server_ = rclcpp_action::create_server<mobile_manipulation_interfaces::action::Controller>(
            this, 
            "controller",
            std::bind(&RobotController::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&RobotController::handle_cancel, this, std::placeholders::_1),
            std::bind(&RobotController::handle_accepted, this, std::placeholders::_1));

 
        linear_speed_ = 0.5;
        angular_speed_ = 2.0;
        
        waypoint_tolerance_ = 0.03; 
        
        angle_tolerance_ = 0.06;

        RCLCPP_INFO(this->get_logger(), "RobotController iniciado com suporte a Preempção!");
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    rclcpp_action::Server<mobile_manipulation_interfaces::action::Controller>::SharedPtr action_server_;

    geometry_msgs::msg::Pose current_pose_;
    
    std::mutex pose_mutex_; 
    std::mutex goal_mutex_;
    std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> active_goal_;

    bool pose_initialized_;
    double linear_speed_;
    double angular_speed_;
    double waypoint_tolerance_;
    double angle_tolerance_;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        current_pose_ = msg->pose.pose;
        pose_initialized_ = true;
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

    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const mobile_manipulation_interfaces::action::Controller::Goal> goal)
    {
        (void)uuid;
        if (goal->path.poses.empty()) 
        {
            RCLCPP_WARN(this->get_logger(), "Caminho vazio recebido. Rejeitando goal.");
            return rclcpp_action::GoalResponse::REJECT;
        }
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Recebido pedido de cancelamento da Action.");
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> goal_handle)
    {
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            
            // Se já existe um goal ativo rodando, aborte-o para matar a thread antiga
            if (active_goal_ && active_goal_->is_active()) 
            {
                RCLCPP_WARN(this->get_logger(), "PREEMPÇÃO: Abortando goal antigo para iniciar novo caminho.");
                auto result = std::make_shared<mobile_manipulation_interfaces::action::Controller::Result>();
                result->success = false;
                try {
                    active_goal_->abort(result);
                } catch (...) { }
            }
            active_goal_ = goal_handle;
        }

        std::thread{std::bind(&RobotController::execute, this, std::placeholders::_1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Iniciando execução do caminho...");

        const auto goal = goal_handle->get_goal();
        auto result = std::make_shared<mobile_manipulation_interfaces::action::Controller::Result>();
        
        std::vector<geometry_msgs::msg::Pose> path;
        for (const auto& p : goal->path.poses) 
        {
            path.push_back(p.pose);
        }

        size_t current_idx = 0;
        rclcpp::Rate rate(100);

        // Espera odometria
        while (rclcpp::ok() && !pose_initialized_) 
        {
             if (goal_handle->is_canceling()) {
                result->success = false;
                goal_handle->canceled(result);
                return;
             }
             rate.sleep();
        }

        while (rclcpp::ok() && current_idx < path.size())
        {
            // --- VERIFICAÇÃO DE SAÍDA DE THREAD ---
            if (goal_handle->is_canceling() || !goal_handle->is_active()) 
            {
                publish_zero_velocity();
                if (goal_handle->is_canceling()) {
                    result->success = false;
                    goal_handle->canceled(result);
                    RCLCPP_INFO(this->get_logger(), "Execução cancelada pelo usuário.");
                } else {
                    RCLCPP_INFO(this->get_logger(), "Thread antiga encerrada (Preempção).");
                }
                return; 
            }
            // --------------------------------------

            geometry_msgs::msg::Pose local_pose;
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                local_pose = current_pose_;
            }

            // --- Lógica de Controle ---
            const auto& target = path[current_idx];
            
            double dx = target.position.x - local_pose.position.x;
            double dy = target.position.y - local_pose.position.y;
            double distance = std::sqrt(dx*dx + dy*dy);

            tf2::Quaternion q(local_pose.orientation.x, local_pose.orientation.y, local_pose.orientation.z, local_pose.orientation.w);
            double yaw = get_yaw_from_quaternion(q);
            double target_yaw = std::atan2(dy, dx);
            double angle_error = normalize_angle(target_yaw - yaw);

            geometry_msgs::msg::Twist cmd;
            double k_p_angular = 2.5; 

            // Turn in place se o erro for grande (> 20 graus)
            if (std::fabs(angle_error) > 0.35) 
            {
                cmd.linear.x = 0.0;
                cmd.angular.z = std::clamp(angle_error * k_p_angular, -angular_speed_, angular_speed_);
            } 
            else 
            {
                // Movimento suave
                double approach_speed = std::min(linear_speed_, distance * 2.0);
                cmd.linear.x = std::clamp(approach_speed, 0.0, linear_speed_);
                cmd.angular.z = std::clamp(angle_error * k_p_angular, -angular_speed_, angular_speed_);
            }

            // --- CORREÇÃO DE ZONA MORTA DO MOTOR ---
            double min_angular_cmd = 0.3; 
            if (std::abs(cmd.angular.z) > 0.001 && std::abs(cmd.angular.z) < min_angular_cmd) 
            {
                cmd.angular.z = (cmd.angular.z > 0) ? min_angular_cmd : -min_angular_cmd;
            }

            // --- SUA INVERSÃO DE HARDWARE ---
            cmd.angular.z = -cmd.angular.z;
            // -------------------------------

            cmd_vel_pub_->publish(cmd);

            // Troca de Waypoint
            if (distance < waypoint_tolerance_) 
            {
                current_idx++;
            }

            rate.sleep();
        }

        publish_zero_velocity();

        if (rclcpp::ok()) 
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            if (active_goal_ == goal_handle && goal_handle->is_active()) 
            {
                result->success = true;
                goal_handle->succeed(result);
                RCLCPP_INFO(this->get_logger(), "Caminho finalizado com sucesso.");
                active_goal_.reset();
            }
        }
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}