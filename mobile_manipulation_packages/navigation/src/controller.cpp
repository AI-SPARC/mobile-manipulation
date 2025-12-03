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

        linear_speed_ = 1.75;
        angular_speed_ = 3.0;
        
        waypoint_tolerance_ = 0.075; 
        lookahead_distance_ = 0.15; 

        RCLCPP_INFO(this->get_logger(), "RobotController iniciado (Logica de Parada Forçada no Final)");
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
    double lookahead_distance_;

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
        if (goal->path.poses.empty()) return rclcpp_action::GoalResponse::REJECT;
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> goal_handle)
    {
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> goal_handle)
    {
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            if (active_goal_ && active_goal_->is_active()) 
            {
                auto result = std::make_shared<mobile_manipulation_interfaces::action::Controller::Result>();
                result->success = false;
                try { active_goal_->abort(result); } catch (...) { }
                publish_zero_velocity(); 
            }
            active_goal_ = goal_handle;
        }
        std::thread{std::bind(&RobotController::execute, this, std::placeholders::_1), goal_handle}.detach();
    }


    void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Controller recebeu");
        publish_zero_velocity();
        const auto goal = goal_handle->get_goal();
        auto result = std::make_shared<mobile_manipulation_interfaces::action::Controller::Result>();
        
        std::vector<geometry_msgs::msg::Pose> path;
        for (const auto& p : goal->path.poses) path.push_back(p.pose);

        if (path.empty()) {
            result->success = true;
            goal_handle->succeed(result);
            return;
        }

        rclcpp::Time zone_1_entry_time; 
        bool in_zone_1 = false;
        
        rclcpp::Time zone_2_entry_time; 
        bool in_zone_2 = false;

        size_t current_idx = 0;
        rclcpp::Rate rate(50); 

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
            if (goal_handle->is_canceling() || !goal_handle->is_active()) 
            {
                publish_zero_velocity();
                if (goal_handle->is_canceling()) {
                    result->success = false;
                    goal_handle->canceled(result);
                }
                return; 
            }

            geometry_msgs::msg::Pose local_pose;
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                local_pose = current_pose_;
            }

            double dist_to_final_goal = std::hypot(
                path.back().position.x - local_pose.position.x,
                path.back().position.y - local_pose.position.y
            );


            if (dist_to_final_goal < waypoint_tolerance_) 
            {
                RCLCPP_INFO(this->get_logger(), "Chegou na tolerância fina (%.3f). Indo para rotação.", dist_to_final_goal);
                break; 
            }

            if (dist_to_final_goal >= waypoint_tolerance_ && dist_to_final_goal < 0.15)
            {
                if (!in_zone_1) {
                    zone_1_entry_time = this->now();
                    in_zone_1 = true;
                }
                
                if ((this->now() - zone_1_entry_time).seconds() > 1.5) {
                    RCLCPP_WARN(this->get_logger(), "Aceitando Zona 1 (%.3fm) após 1.5s. Indo para rotação.", dist_to_final_goal);
                    break;
                }
            }
            else {
                in_zone_1 = false;
            }

            if (dist_to_final_goal >= 0.15 && dist_to_final_goal < 0.25)
            {
                if (!in_zone_2) {
                    zone_2_entry_time = this->now();
                    in_zone_2 = true;
                }
                
                if ((this->now() - zone_2_entry_time).seconds() > 4.0) {
                    RCLCPP_WARN(this->get_logger(), "Aceitando Zona 2 (%.3fm) após 4.0s. Indo para rotação.", dist_to_final_goal);
                    break;
                }
            }
            else {
                in_zone_2 = false;
            }

            size_t target_idx = current_idx;
            double dx, dy, dist_to_target;
            
            for (size_t i = current_idx; i < path.size(); i++)
            {
                double d_x = path[i].position.x - local_pose.position.x;
                double d_y = path[i].position.y - local_pose.position.y;
                double dist = std::hypot(d_x, d_y);
                
                target_idx = i;
                dx = d_x;
                dy = d_y;
                dist_to_target = dist;

                if (dist >= lookahead_distance_) break; 
            }
            
            tf2::Quaternion q(local_pose.orientation.x, local_pose.orientation.y, local_pose.orientation.z, local_pose.orientation.w);
            double yaw = get_yaw_from_quaternion(q);
            double target_yaw = std::atan2(dy, dx);
            double angle_error = normalize_angle(target_yaw - yaw);

            geometry_msgs::msg::Twist cmd;
            double k_p_angular = 2.0; 

            double dist_to_current_wp = std::hypot(
                path[current_idx].position.x - local_pose.position.x,
                path[current_idx].position.y - local_pose.position.y
            );
            if (dist_to_current_wp < waypoint_tolerance_) current_idx++;
            
            if (std::fabs(angle_error) > 0.8) 
            {
                cmd.linear.x = 0.0;
                cmd.angular.z = std::clamp(angle_error * k_p_angular, -angular_speed_, angular_speed_);
                if (std::abs(cmd.angular.z) < 0.3) cmd.angular.z = (cmd.angular.z > 0) ? 0.3 : -0.3;
            } 
            else 
            {
                double curvature_slowdown = std::max(0.2, 1.0 - (std::fabs(angle_error) * 1.5));
                double distance_slowdown = std::min(linear_speed_, dist_to_target);
                cmd.linear.x = std::clamp(distance_slowdown * curvature_slowdown, 0.0, linear_speed_);
                cmd.angular.z = std::clamp(angle_error * k_p_angular, -angular_speed_, angular_speed_);
            }
            cmd.angular.z = -cmd.angular.z; 

            cmd_vel_pub_->publish(cmd);

     
            if (current_idx >= path.size()) 
            {
                current_idx = path.size() - 1; 
            }

            rate.sleep();
        }



        publish_zero_velocity();
        rclcpp::sleep_for(std::chrono::milliseconds(400)); 

        geometry_msgs::msg::Pose final_pose = path.back();
        tf2::Quaternion q_final(
            final_pose.orientation.x, final_pose.orientation.y,
            final_pose.orientation.z, final_pose.orientation.w);
        double final_target_yaw = get_yaw_from_quaternion(q_final);
        
        bool aligned = false;
        double alignment_tolerance = 0.05; 
        
        auto rotation_start_time = this->now();
        rclcpp::Duration rotation_timeout = rclcpp::Duration::from_seconds(4.0); 

        RCLCPP_INFO(this->get_logger(), "Iniciando rotação final (Max 5s)...");

        while (rclcpp::ok() && !aligned)
        {
             if (goal_handle->is_canceling()) {
                publish_zero_velocity();
                result->success = false;
                goal_handle->canceled(result);
                return;
             }

             if ((this->now() - rotation_start_time) > rotation_timeout) {
                 RCLCPP_WARN(this->get_logger(), "Timeout de Rotação! Parando mesmo sem alinhar perfeitamente.");
                 aligned = true;
                 break;
             }

             geometry_msgs::msg::Pose local_pose;
             {
                 std::lock_guard<std::mutex> lock(pose_mutex_);
                 local_pose = current_pose_;
             }

             tf2::Quaternion q_curr(local_pose.orientation.x, local_pose.orientation.y, local_pose.orientation.z, local_pose.orientation.w);
             double current_yaw = get_yaw_from_quaternion(q_curr);
             double angle_error = normalize_angle(final_target_yaw - current_yaw);

             if (std::abs(angle_error) <= alignment_tolerance) {
                 aligned = true;
                 break;
             }

             geometry_msgs::msg::Twist cmd;
             cmd.linear.x = 0.0;
             double k_p_final = 2.5; 
             
             cmd.angular.z = std::clamp(angle_error * k_p_final, -angular_speed_, angular_speed_);
             
             // Torque mínimo
             if (std::abs(cmd.angular.z) < 0.2) cmd.angular.z = (cmd.angular.z > 0) ? 0.2 : -0.2;

             cmd.angular.z = -cmd.angular.z; 

             cmd_vel_pub_->publish(cmd);
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
                active_goal_.reset();
                RCLCPP_INFO(this->get_logger(), "Objetivo finalizado.");
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