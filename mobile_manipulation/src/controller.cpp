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
#include <map>   // Necessário para std::map
#include <tuple> // Necessário para std::tuple
#include "rclcpp_action/rclcpp_action.hpp"
#include "mobile_manipulation_interfaces/action/controller.hpp"
#include "mobile_manipulation_interfaces/srv/stop_pose.hpp"

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

        service_ = this->create_service<mobile_manipulation_interfaces::srv::StopPose>("/stop_pose",
            std::bind(&RobotController::handle_request, this, std::placeholders::_1, std::placeholders::_2));

        linear_speed_ = 0.5;
        angular_speed_ = 2.0;
        waypoint_tolerance_ = 0.1; 
        angle_tolerance_ = 0.1;

        // Só iniciando com valores impossíveis para não parar quando não deveria.
        stop_pose_.position.x = -99999.256; 

        RCLCPP_INFO(this->get_logger(), "RobotController iniciado (Modo Action Loop)!");
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    // Action server.
    rclcpp_action::Server<mobile_manipulation_interfaces::action::Controller>::SharedPtr action_server_;

    // Service.
    rclcpp::Service<mobile_manipulation_interfaces::srv::StopPose>::SharedPtr service_;

    geometry_msgs::msg::Pose current_pose_;
    geometry_msgs::msg::Pose stop_pose_;
    
    std::mutex pose_mutex_; 
    std::mutex stop_pose_mutex;

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

    // Service
    void handle_request(const std::shared_ptr<mobile_manipulation_interfaces::srv::StopPose::Request> request, 
                        std::shared_ptr<mobile_manipulation_interfaces::srv::StopPose::Response> response)
    {   
        {
            std::lock_guard<std::mutex> lock(stop_pose_mutex);
            stop_pose_ = request->stop_pose;
        }
        
        response->success = true;
        RCLCPP_INFO(this->get_logger(), "Stop Pose atualizado via serviço.");
    }

    // Action server (controller).

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
        std::thread{std::bind(&RobotController::execute, this, std::placeholders::_1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Iniciando execução do caminho (Action Thread)...");

        const auto goal = goal_handle->get_goal();
        auto result = std::make_shared<mobile_manipulation_interfaces::action::Controller::Result>();
        
        std::vector<geometry_msgs::msg::Pose> path;
        std::map<std::tuple<float, float, float>, int> index_map;
        int i = 0;

        for (const auto& p : goal->path.poses) 
        {
            path.push_back(p.pose);
            index_map[std::make_tuple(static_cast<float>(p.pose.position.x), 
                                      static_cast<float>(p.pose.position.y), 
                                      static_cast<float>(p.pose.position.z))] = i;
            i++;
        }

        size_t current_idx = 0;
        rclcpp::Rate rate(100);

        while (rclcpp::ok() && !pose_initialized_) 
        {
             if (goal_handle->is_canceling()) 
             {
                result->success = false;
                goal_handle->canceled(result);
                return;
             }
             RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Aguardando odometria...");
             rate.sleep();
        }

        while (rclcpp::ok() && current_idx < path.size())
        {
            if (goal_handle->is_canceling()) 
            {
                publish_zero_velocity();
                result->success = false;
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Execução cancelada.");
                return;
            }

            geometry_msgs::msg::Pose local_pose;
            int stop_index = -1; 

            {
                std::lock_guard<std::mutex> lock(stop_pose_mutex);
                auto search_tuple = std::make_tuple(static_cast<float>(stop_pose_.position.x), 
                                                    static_cast<float>(stop_pose_.position.y), 
                                                    static_cast<float>(stop_pose_.position.z));
                
                if(index_map.find(search_tuple) != index_map.end())
                {
                    stop_index = index_map[search_tuple];
                }
            }

            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                local_pose = current_pose_;
            }

            if(stop_index != -1 && static_cast<int>(current_idx) >= stop_index)
            {
                RCLCPP_INFO(this->get_logger(), "Parada solicitada no waypoint %d.", stop_index);
                break;
            }

            // 3. Lógica de Controle
            const auto& target = path[current_idx];
            
            double dx = target.position.x - local_pose.position.x;
            double dy = target.position.y - local_pose.position.y;
            double distance = std::sqrt(dx*dx + dy*dy);

            tf2::Quaternion q(local_pose.orientation.x,
                              local_pose.orientation.y,
                              local_pose.orientation.z,
                              local_pose.orientation.w);
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
                double approach_speed = std::min(linear_speed_, distance * 1.5);
                cmd.linear.x = std::clamp(approach_speed, 0.0, linear_speed_);
                cmd.angular.z = std::clamp(angle_error * 2.0, -angular_speed_, angular_speed_);
            }

            cmd.angular.z = -cmd.angular.z;
            cmd_vel_pub_->publish(cmd);

            if (distance < waypoint_tolerance_) 
            {
                RCLCPP_INFO(this->get_logger(), "Reached waypoint %zu/%zu.", current_idx + 1, path.size());
                current_idx++;
            }

            rate.sleep();
        }

        publish_zero_velocity();

        if (rclcpp::ok()) 
        {
            result->success = true;
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Action finalizada com sucesso.");
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