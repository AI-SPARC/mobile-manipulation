/**
 * @file controller.cpp
 * @brief Controlador de Trajetória Local (Pure Pursuit Modificado).
 * Implementa seguimento de caminho com zonas de tolerância adaptativas e rotação final precisa.
 */

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
    // DOC-START: Controller_Constructor
    // Construtor: Configura publishers, subscribers e o Action Server do controlador.
    RobotController() : Node("controller"), 
                              pose_initialized_(false)
    {
        // Publica comandos de velocidade para a base móvel
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        
        // Assina odometria para feedback de malha fechada
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&RobotController::odom_callback, this, std::placeholders::_1));
            
        // Action Server que recebe o caminho completo (Path) do planejador global
        this->action_server_ = rclcpp_action::create_server<mobile_manipulation_interfaces::action::Controller>(
            this, 
            "controller",
            std::bind(&RobotController::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&RobotController::handle_cancel, this, std::placeholders::_1),
            std::bind(&RobotController::handle_accepted, this, std::placeholders::_1));

        // Parâmetros de Controle (PID e Geometria)
        linear_speed_ = 4.75;  // Velocidade linear máxima (m/s)
        angular_speed_ = 8.0;  // Velocidade angular máxima (rad/s)
        
        waypoint_tolerance_ = 0.075; // Distância para considerar um waypoint "atingido"
        lookahead_distance_ = 0.15;  // Horizonte de previsão do Pure Pursuit

        RCLCPP_INFO(this->get_logger(), "RobotController iniciado.");
    }
    // DOC-END: Controller_Constructor

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

    // DOC-START: odom_callback
    // Callback de odometria. Atualiza a pose global do robô de forma thread-safe.
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        current_pose_ = msg->pose.pose;
        pose_initialized_ = true; // Libera o controlador para começar
    }
    // DOC-END: odom_callback

    // DOC-START: Helpers
    // Envia comando de parada (velocidade zero) para segurança.
    void publish_zero_velocity()
    {
        geometry_msgs::msg::Twist stop;
        stop.linear.x = 0.0;
        stop.angular.z = 0.0;
        cmd_vel_pub_->publish(stop);
    }

    // Converte Quaternião (x,y,z,w) para ângulo Yaw (radianos).
    double get_yaw_from_quaternion(const tf2::Quaternion& q)
    {
        double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1.0 - 2.0 * (q.y()*q.y() + q.z()*q.z());
        return std::atan2(siny_cosp, cosy_cosp);
    }

    // Normaliza ângulo para o intervalo [-PI, PI].
    double normalize_angle(double a)
    {
        while(a > M_PI) a -= 2*M_PI;
        while(a < -M_PI) a += 2*M_PI;
        return a;
    }
    // DOC-END: Helpers

    // Action Server Callbacks (Padrão ROS 2)
    // ...

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
            // Preempção: Se já existe um goal ativo, aborta o anterior e aceita o novo.
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


    // DOC-START: execute
    // Loop Principal de Controle:
    // 1. Segue o caminho usando Pure Pursuit (Lookahead).
    // 2. Monitora "Zonas de Estagnação" para aceitar chegada mesmo com erro.
    // 3. Executa alinhamento final de rotação (in-place rotation).
    void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::Controller>> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Controller: Executando trajetória...");
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

        // Variáveis para lógica de "Timeout de Zona" (evita oscilação infinita perto do alvo)
        rclcpp::Time zone_1_entry_time; 
        bool in_zone_1 = false;
        rclcpp::Time zone_2_entry_time; 
        bool in_zone_2 = false;

        size_t current_idx = 0;
        rclcpp::Rate rate(50); // 50 Hz

        // Aguarda primeira odometria válida
        while (rclcpp::ok() && !pose_initialized_) 
        {
             if (goal_handle->is_canceling()) {
                result->success = false;
                goal_handle->canceled(result);
                return;
             }
             rate.sleep();
        }

        // Loop de Seguimento de Caminho (Translação + Curva)
        while (rclcpp::ok() && current_idx < path.size())
        {
            // Verificação de cancelamento/preempção
            if (goal_handle->is_canceling() || !goal_handle->is_active()) 
            {
                publish_zero_velocity();
                if (goal_handle->is_canceling()) {
                    result->success = false;
                    goal_handle->canceled(result);
                }
                return; 
            }

            // Atualiza pose local
            geometry_msgs::msg::Pose local_pose;
            {
                std::lock_guard<std::mutex> lock(pose_mutex_);
                local_pose = current_pose_;
            }

            // Cálculo da distância até o objetivo final
            double dist_to_final_goal = std::hypot(
                path.back().position.x - local_pose.position.x,
                path.back().position.y - local_pose.position.y
            );

            // Condição de Parada 1: Chegou na tolerância fina
            if (dist_to_final_goal < waypoint_tolerance_) 
            {
                RCLCPP_INFO(this->get_logger(), "Alvo atingido (precisão).");
                break; 
            }

            // Condição de Parada 2 (Watchdog de Zona 1): Perto, mas não perfeito.
            if (dist_to_final_goal >= waypoint_tolerance_ && dist_to_final_goal < 0.15)
            {
                if (!in_zone_1) {
                    zone_1_entry_time = this->now();
                    in_zone_1 = true;
                }
                // Se ficar 1s tentando ajustar finamente e não conseguir, aceita e para.
                if ((this->now() - zone_1_entry_time).seconds() > 1.0) {
                    RCLCPP_WARN(this->get_logger(), "Watchdog Zona 1: Aceitando posição.");
                    break;
                }
            }
            else in_zone_1 = false;

            // Condição de Parada 3 (Watchdog de Zona 2): Um pouco mais longe.
            if (dist_to_final_goal >= 0.15 && dist_to_final_goal < 0.25)
            {
                if (!in_zone_2) {
                    zone_2_entry_time = this->now();
                    in_zone_2 = true;
                }
                if ((this->now() - zone_2_entry_time).seconds() > 3.0) {
                    RCLCPP_WARN(this->get_logger(), "Watchdog Zona 2: Aceitando posição.");
                    break;
                }
            }
            else in_zone_2 = false;

            // Pure Pursuit: Encontra o ponto alvo no horizonte (Lookahead)
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
            
            // Cálculo do erro angular
            tf2::Quaternion q(local_pose.orientation.x, local_pose.orientation.y, local_pose.orientation.z, local_pose.orientation.w);
            double yaw = get_yaw_from_quaternion(q);
            double target_yaw = std::atan2(dy, dx);
            double angle_error = normalize_angle(target_yaw - yaw);

            geometry_msgs::msg::Twist cmd;
            double k_p_angular = 2.0; 

            // Avanço do índice do caminho (impede que o robô volte para trás)
            double dist_to_current_wp = std::hypot(
                path[current_idx].position.x - local_pose.position.x,
                path[current_idx].position.y - local_pose.position.y
            );
            if (dist_to_current_wp < waypoint_tolerance_) current_idx++;
            
            // Lógica de Velocidade Adaptativa
            if (std::fabs(angle_error) > 0.8) 
            {
                // Erro angular grande: Gira no próprio eixo (Turn in place)
                cmd.linear.x = 0.0;
                cmd.angular.z = std::clamp(angle_error * k_p_angular, -angular_speed_, angular_speed_);
                // Garante torque mínimo para vencer atrito estático
                if (std::abs(cmd.angular.z) < 0.3) cmd.angular.z = (cmd.angular.z > 0) ? 0.3 : -0.3;
            } 
            else 
            {
                // Erro pequeno: Anda e gira (Curvatura suave)
                double curvature_slowdown = std::max(0.2, 1.0 - (std::fabs(angle_error) * 1.5));
                double distance_slowdown = std::min(linear_speed_, dist_to_target); // Desacelera ao chegar
                
                cmd.linear.x = std::clamp(distance_slowdown * curvature_slowdown, 0.0, linear_speed_);
                cmd.angular.z = std::clamp(angle_error * k_p_angular, -angular_speed_, angular_speed_);
            }
            cmd.angular.z = -cmd.angular.z; // Inversão se necessário (depende da cinemática)

            cmd_vel_pub_->publish(cmd);

            if (current_idx >= path.size()) current_idx = path.size() - 1; 

            rate.sleep();
        }

        // Fim do movimento linear. Para e prepara rotação final.
        publish_zero_velocity();
        rclcpp::sleep_for(std::chrono::milliseconds(200)); 

        // Rotação Final: Alinha com a orientação do último waypoint
        geometry_msgs::msg::Pose final_pose = path.back();
        tf2::Quaternion q_final(
            final_pose.orientation.x, final_pose.orientation.y,
            final_pose.orientation.z, final_pose.orientation.w);
        double final_target_yaw = get_yaw_from_quaternion(q_final);
        
        bool aligned = false;
        double alignment_tolerance = 0.05; // ~2.8 graus
        
        auto rotation_start_time = this->now();
        rclcpp::Duration rotation_timeout = rclcpp::Duration::from_seconds(2.0); 

        RCLCPP_INFO(this->get_logger(), "Iniciando alinhamento final...");

        while (rclcpp::ok() && !aligned)
        {
             if (goal_handle->is_canceling()) {
                publish_zero_velocity();
                result->success = false;
                goal_handle->canceled(result);
                return;
             }

             // Timeout de rotação para não travar o fluxo
             if ((this->now() - rotation_start_time) > rotation_timeout) {
                 RCLCPP_WARN(this->get_logger(), "Timeout de Rotação. Encerrando.");
                 aligned = true;
                 break;
             }

             // Atualiza pose
             geometry_msgs::msg::Pose local_pose;
             {
                 std::lock_guard<std::mutex> lock(pose_mutex_);
                 local_pose = current_pose_;
             }

             // Calcula erro angular
             tf2::Quaternion q_curr(local_pose.orientation.x, local_pose.orientation.y, local_pose.orientation.z, local_pose.orientation.w);
             double current_yaw = get_yaw_from_quaternion(q_curr);
             double angle_error = normalize_angle(final_target_yaw - current_yaw);

             if (std::abs(angle_error) <= alignment_tolerance) {
                 aligned = true;
                 break;
             }

             // Controle P simples para rotação
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

        // Finaliza Action com sucesso
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
    // DOC-END: execute

};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}