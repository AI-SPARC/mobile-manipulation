#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "mobile_manipulation_interfaces/msg/fleet_paths.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace std::chrono_literals;

class FleetScenarioGenerator : public rclcpp::Node {
public:
    FleetScenarioGenerator() : Node("fleet_scenario_generator"), gen_(rd_()) {
        // --- PARÂMETROS DE CONTAGEM ---
        this->declare_parameter<int>("min_robot_count", 10);
        this->declare_parameter<int>("max_robot_count", 30);
        
        // --- PARÂMETROS DA ÁREA (MAPA) ---
        // O mapa vai de -X a +X e -Y a +Y
        this->declare_parameter<double>("map_limit_x", 25.0); 
        this->declare_parameter<double>("map_limit_y", 25.0);
        
        // --- PARÂMETRO DE QUALIDADE ---
        // Distância mínima que um robô deve viajar. 
        // Evita que um robô nasça e morra no mesmo lugar.
        this->declare_parameter<double>("min_travel_dist", 10.0);

        publisher_ = this->create_publisher<mobile_manipulation_interfaces::msg::FleetPaths>(
            "/fleet/all_robot_plans", 10);

        // Timer para publicação automática (opcional)
        timer_ = this->create_wall_timer(
            5000ms, std::bind(&FleetScenarioGenerator::timer_callback, this));

        // Serviço para gerar sob demanda (Manual)
        service_ = this->create_service<std_srvs::srv::Trigger>(
            "/fleet/generate_scenario",
            std::bind(&FleetScenarioGenerator::handle_service_request, this, std::placeholders::_1, std::placeholders::_2));
            
        RCLCPP_INFO(this->get_logger(), "Gerador CAÓTICO Iniciado.");
        RCLCPP_INFO(this->get_logger(), "Robos irao de pontos aleatorios A para B.");
    }

private:
    void timer_callback() {
        // Se quiser usar APENAS o serviço (manual), comente a linha abaixo:
        // publish_scenario(); 
    }

    void handle_service_request(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;
        RCLCPP_INFO(this->get_logger(), "Sinal recebido. Gerando caos aleatorio...");
        publish_scenario();
        response->success = true;
        response->message = "Cenario caotico publicado";
    }

    void publish_scenario() {
        // 1. Ler parâmetros
        int min_r = this->get_parameter("min_robot_count").as_int();
        int max_r = this->get_parameter("max_robot_count").as_int();
        double limit_x = this->get_parameter("map_limit_x").as_double();
        double limit_y = this->get_parameter("map_limit_y").as_double();
        double min_dist = this->get_parameter("min_travel_dist").as_double();

        // Proteção de input
        if (min_r > max_r) std::swap(min_r, max_r);
        if (min_r < 1) min_r = 1;

        // 2. Sortear número de robôs
        std::uniform_int_distribution<> count_dist(min_r, max_r);
        int total_robots = count_dist(gen_);

        // Distributions para Posição (X e Y)
        std::uniform_real_distribution<double> pos_x_dist(-limit_x, limit_x);
        std::uniform_real_distribution<double> pos_y_dist(-limit_y, limit_y);

        mobile_manipulation_interfaces::msg::FleetPaths msg;
        msg.header.frame_id = "world";
        msg.header.stamp = this->now();

        for (int i = 0; i < total_robots; ++i) {
            double start_x, start_y, goal_x, goal_y;
            double dist_sq = 0.0;
            double min_dist_sq = min_dist * min_dist;

            // Loop para garantir que o destino não é muito perto da origem
            int attempts = 0;
            do {
                start_x = pos_x_dist(gen_);
                start_y = pos_y_dist(gen_);
                
                goal_x = pos_x_dist(gen_);
                goal_y = pos_y_dist(gen_);

                double dx = goal_x - start_x;
                double dy = goal_y - start_y;
                dist_sq = (dx * dx) + (dy * dy);
                attempts++;
            } while (dist_sq < min_dist_sq && attempts < 100);

            nav_msgs::msg::Path path;
            path.header.frame_id = "world";
            path.header.stamp = this->now();

            // Adiciona Start e Goal
            // Se quiser caminhos curvos aleatórios, pode adicionar waypoints intermediários aqui
            path.poses.push_back(create_pose(start_x, start_y));
            path.poses.push_back(create_pose(goal_x, goal_y));

            msg.robot_ids.push_back(i);
            msg.robot_speeds.push_back(2.0); // Velocidade fixa ou também poderia ser aleatória
            msg.paths.push_back(path);
        }

        publisher_->publish(msg);
        RCLCPP_INFO(this->get_logger(), "Publicado: %d Robos em trajetorias aleatorias.", total_robots);
    }

    geometry_msgs::msg::PoseStamped create_pose(double x, double y) {
        geometry_msgs::msg::PoseStamped p;
        p.header.frame_id = "world";
        p.pose.position.x = x;
        p.pose.position.y = y;
        p.pose.position.z = 0.0;
        p.pose.orientation.w = 1.0;
        return p;
    }

    rclcpp::Publisher<mobile_manipulation_interfaces::msg::FleetPaths>::SharedPtr publisher_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr service_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::random_device rd_;
    std::mt19937 gen_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FleetScenarioGenerator>());
    rclcpp::shutdown();
    return 0;
}