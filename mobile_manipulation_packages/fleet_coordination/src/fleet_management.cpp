#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "mobile_manipulation_interfaces/msg/fleet_paths.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using std::placeholders::_1;
using namespace std::chrono_literals;

// --- CONFIGURAÇÕES GLOBAIS ---
const double ROBOT_WIDTH = 0.60;      // Robô quadrado de 60x60cm
const double LONGITUDINAL_PAD = 0.30; // Padding para retângulos estáticos
const double TIME_STEP = 0.02;        // 20ms (Precisão da verificação física)
const double ANIMATION_FREQ = 0.05;   // 20Hz (Taxa de atualização visual no RViz)

// --- MATEMÁTICA VETORIAL ---
struct Vec2 { 
    double x, y; 
    Vec2 operator-(const Vec2& other) const { return {x - other.x, y - other.y}; }
    double dot(const Vec2& other) const { return x * other.x + y * other.y; }
};
using Rectangle = std::vector<Vec2>;

// --- ESTRUTURAS DE DADOS ---
struct InterpolationResult {
    geometry_msgs::msg::Pose pose;
    int current_segment_index; // Índice do segmento estático onde o robô está
};

struct RobotSimData {
    std::string robot_id;
    std::string frame_id;
    double speed;
    nav_msgs::msg::Path path_msg;
    
    // Otimização: Retângulos Estáticos (Broad Phase)
    std::vector<Rectangle> static_rects;
    std::vector<bool> segment_is_risky; // True se este segmento cruza geometricamente com outro
    
    // Para interpolação de movimento
    std::vector<double> accumulated_distances; 
    double total_duration;
};

// Estrutura para o Relatório Final
struct CollisionReport {
    bool collision_detected = false;
    double start_time = -1.0;
    double end_time = -1.0;
    std::string robot_1_id;
    std::string robot_2_id;
};

class FleetManagement : public rclcpp::Node
{
public:
    FleetManagement() : Node("fleet_traffic_manager")
    {
        // Subscriber: Recebe os planos e velocidades
        sub_paths_ = this->create_subscription<mobile_manipulation_interfaces::msg::FleetPaths>(
            "/fleet/all_robot_plans", 10, std::bind(&FleetManagement::topic_callback, this, _1));

        // Publisher: Manda os marcadores para o RViz
        pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/fleet/debug_markers", 10);
            
        // Timer: Apenas para "tocar" a animação visualmente (não faz cálculo pesado aqui)
        anim_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(ANIMATION_FREQ), 
            std::bind(&FleetManagement::animation_tick, this));

        RCLCPP_INFO(this->get_logger(), "Fleet Manager Iniciado: Pré-cálculo de colisão com intervalo de tempo.");
    }

private:
    std::vector<RobotSimData> fleet_data_;
    std::vector<CollisionReport> active_reports_;
    
    double current_anim_time_ = 0.0;
    double max_sim_duration_ = 0.0;

    // -------------------------------------------------------------------------
    // 1. PROCESSAMENTO DE DADOS (Executado uma vez ao receber mensagem)
    // -------------------------------------------------------------------------
    void topic_callback(const mobile_manipulation_interfaces::msg::FleetPaths::SharedPtr msg)
    {
        fleet_data_.clear();
        active_reports_.clear();
        current_anim_time_ = 0.0;
        max_sim_duration_ = 0.0;

        // A. Construção dos Dados e Geometria Estática
        for (size_t i = 0; i < msg->paths.size(); ++i) 
        {
            RobotSimData data;
            data.robot_id = msg->robot_ids[i];
            data.frame_id = msg->paths[i].header.frame_id;
            data.path_msg = msg->paths[i];
            
            // Define velocidade (default 0.5 se não vier na msg)
            if (msg->robot_speeds.size() > i) 
            {
                data.speed = msg->robot_speeds[i];
            }
            else
            {
                data.speed = 0.5;
            } 

            if (data.path_msg.poses.size() < 2) 
            {
                continue;
            }
            

            double dist_acc = 0.0;
            data.accumulated_distances.push_back(0.0);

            for (size_t j = 0; j < data.path_msg.poses.size() - 1; ++j) 
            {
                // Cria Retângulos Estáticos para Otimização
                auto p1 = data.path_msg.poses[j].pose.position;
                auto p2 = data.path_msg.poses[j+1].pose.position;
                Rectangle rect = calculate_static_rect(p1, p2);
                
                data.static_rects.push_back(rect);
                data.segment_is_risky.push_back(false); // Inicialmente seguro

                dist_acc += std::hypot(p2.x - p1.x, p2.y - p1.y);
                data.accumulated_distances.push_back(dist_acc);
            }
            
            data.total_duration = dist_acc / data.speed;
            if (data.total_duration > max_sim_duration_) 
            {
                max_sim_duration_ = data.total_duration;
            }

            fleet_data_.push_back(data);
        }

        // B. Broad Phase: Identificar Segmentos de Risco (Estático)
        for (size_t i = 0; i < fleet_data_.size(); ++i) 
        {
            for (size_t k = i + 1; k < fleet_data_.size(); ++k) 
            {
                mark_risky_segments(fleet_data_[i], fleet_data_[k]);
            }
        }

        // C. Narrow Phase: Simulação Acelerada (Encontrar Intervalos de Tempo)
        run_collision_simulation();
    }

    // Função auxiliar para Broad Phase
    void mark_risky_segments(RobotSimData& r1, RobotSimData& r2) 
    {
        for (size_t i = 0; i < r1.static_rects.size(); ++i) 
        {
            for (size_t k = 0; k < r2.static_rects.size(); ++k) 
            {
                if (check_sat_intersection(r1.static_rects[i], r2.static_rects[k])) 
                {
                    r1.segment_is_risky[i] = true;
                    r2.segment_is_risky[k] = true;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. SIMULAÇÃO FÍSICA PRÉ-CALCULADA (Passo a passo no tempo)
    // -------------------------------------------------------------------------
    void run_collision_simulation() 
    {
        if (fleet_data_.empty()) return;

        // Cria relatórios vazios para cada par
        int report_count = 0;
        for (size_t i = 0; i < fleet_data_.size(); ++i) 
        {
            for (size_t k = i + 1; k < fleet_data_.size(); ++k) 
            {
                CollisionReport rep;
                rep.robot_1_id = fleet_data_[i].robot_id;
                rep.robot_2_id = fleet_data_[k].robot_id;
                active_reports_.push_back(rep);
                report_count++;
            }
        }

        // Loop de simulação (0.00s -> MaxDuration)
        for (double t = 0.0; t <= max_sim_duration_; t += TIME_STEP) 
        {
            
            // 1. Calcula estado de todos os robôs no tempo T
            std::vector<InterpolationResult> states(fleet_data_.size());
            std::vector<Rectangle> dynamic_boxes(fleet_data_.size());

            for (size_t r = 0; r < fleet_data_.size(); ++r) 
            {
                states[r] = calculate_pose_at_time(fleet_data_[r], t);
                dynamic_boxes[r] = calculate_dynamic_box(states[r].pose);
            }

            // 2. Verifica colisão
            int current_rep_idx = 0;
            for (size_t i = 0; i < fleet_data_.size(); ++i) 
            {
                for (size_t k = i + 1; k < fleet_data_.size(); ++k) 
                {
                    
                    // OTIMIZAÇÃO: Só roda SAT se ambos estiverem em segmentos marcados como arriscados
                    int s1 = states[i].current_segment_index;
                    int s2 = states[k].current_segment_index;
                    
                    bool potential_risk = false;
                    // Verifica limites do vetor
                    if (s1 >= 0 && s1 < (int)fleet_data_[i].segment_is_risky.size() && s2 >= 0 && s2 < (int)fleet_data_[k].segment_is_risky.size()) 
                    {
                        
                        if (fleet_data_[i].segment_is_risky[s1] && fleet_data_[k].segment_is_risky[s2]) 
                        {
                            potential_risk = true;
                        }
                    }

                    if (potential_risk) 
                    {
                        // Teste SAT Real (Caixa Dinâmica vs Caixa Dinâmica)
                        if (check_sat_intersection(dynamic_boxes[i], dynamic_boxes[k])) 
                        {
                            // COLISÃO DETECTADA!
                            auto& rep = active_reports_[current_rep_idx];

                            if (!rep.collision_detected) 
                            {
                                rep.collision_detected = true;
                                rep.start_time = t; // Marca início
                            }

                            rep.end_time = t; // Atualiza fim (estende o intervalo)
                        }
                    }
                    current_rep_idx++;
                }
            }
        }
        
        // Log no terminal para debug
        for (const auto& rep : active_reports_) {
            if (rep.collision_detected) {
                RCLCPP_WARN(this->get_logger(), "COLISÃO PREVISTA: %s e %s entre %.2fs e %.2fs", 
                    rep.robot_1_id.c_str(), rep.robot_2_id.c_str(), rep.start_time, rep.end_time);
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. ANIMAÇÃO VISUAL (RViz)
    // -------------------------------------------------------------------------
    void animation_tick()
    {
        if (fleet_data_.empty()) return;

        current_anim_time_ += 0.05; 
        if (current_anim_time_ > (max_sim_duration_ + 2.0)) current_anim_time_ = 0.0;

        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker del; 
        del.action = visualization_msgs::msg::Marker::DELETEALL; 
        marker_array.markers.push_back(del);
        int id_counter = 0;

        // A. TEXTO HUD (Status Global)
        visualization_msgs::msg::Marker hud;
        hud.header.frame_id = fleet_data_[0].frame_id;
        hud.header.stamp = this->now();
        hud.ns = "hud_status";
        hud.id = id_counter++;
        hud.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        hud.action = visualization_msgs::msg::Marker::ADD;
        hud.pose.position.x = 0; hud.pose.position.y = 0; hud.pose.position.z = 3.0;
        hud.scale.z = 0.5;

        std::stringstream ss;
        bool any_crash = false;
        for (const auto& rep : active_reports_) 
        {
            if (rep.collision_detected) 
            {
                any_crash = true;
                ss << "[CRASH] " << rep.robot_1_id << " x " << rep.robot_2_id 
                   << " (T: " << std::fixed << std::setprecision(2) << rep.start_time << "s - " << rep.end_time << "s)\n";
            }
        }

        if (any_crash) 
        {
            hud.color.r = 1.0; hud.color.g = 0.0; hud.color.b = 0.0; hud.color.a = 1.0; // Vermelho
            hud.text = ss.str();
        } 
        else 
        {
            hud.color.r = 0.0; hud.color.g = 1.0; hud.color.b = 0.0; hud.color.a = 1.0; // Verde
            hud.text = "ALL PATHS CLEAR";
        }
        marker_array.markers.push_back(hud);

        // B. ROBÔS ANIMADOS
        for (const auto& robot : fleet_data_) 
        {
            auto state = calculate_pose_at_time(robot, current_anim_time_);

            visualization_msgs::msg::Marker bot;
            bot.header.frame_id = robot.frame_id;
            bot.header.stamp = this->now();
            bot.ns = "anim_bot_" + robot.robot_id;
            bot.id = 0;
            bot.type = visualization_msgs::msg::Marker::CUBE;
            bot.action = visualization_msgs::msg::Marker::ADD;
            bot.pose = state.pose;
            bot.scale.x = ROBOT_WIDTH; bot.scale.y = ROBOT_WIDTH; bot.scale.z = 0.3;

            // Lógica de Cor: Se o tempo atual está dentro de algum intervalo de colisão deste robô
            bool is_crashing_now = false;
            for (const auto& rep : active_reports_) 
            {
                if (rep.collision_detected && (rep.robot_1_id == robot.robot_id || rep.robot_2_id == robot.robot_id)) 
                {
                    if (current_anim_time_ >= rep.start_time && current_anim_time_ <= rep.end_time) 
                    {
                        is_crashing_now = true;
                    }
                }
            }

            if (is_crashing_now) 
            {
                bot.color.r = 1.0; bot.color.g = 0.0; bot.color.b = 0.0; bot.color.a = 1.0; // Vermelho
            }
            else 
            {
                bot.color.r = 0.0; bot.color.g = 1.0; bot.color.b = 1.0; bot.color.a = 0.8; // Ciano
            }
            marker_array.markers.push_back(bot);

            for (size_t r = 0; r < robot.static_rects.size(); ++r) 
            {
                visualization_msgs::msg::Marker rect_mk;
                rect_mk.header.frame_id = robot.frame_id;
                rect_mk.header.stamp = this->now();
                rect_mk.ns = "path_" + robot.robot_id;
                rect_mk.id = id_counter++;
                rect_mk.type = visualization_msgs::msg::Marker::LINE_STRIP;
                rect_mk.action = visualization_msgs::msg::Marker::ADD;
                rect_mk.scale.x = 0.02;

                if (robot.segment_is_risky[r]) 
                {
                    rect_mk.color.r = 1.0; rect_mk.color.g = 0.5; rect_mk.color.a = 0.5; // Laranja (Risco)
                } 
                else 
                {
                    rect_mk.color.g = 1.0; rect_mk.color.a = 0.1; // Verde transparente (Seguro)
                }

                for(const auto& v : robot.static_rects[r]) 
                {
                    geometry_msgs::msg::Point p; p.x = v.x; p.y = v.y; rect_mk.points.push_back(p);
                }

                geometry_msgs::msg::Point p0; p0.x = robot.static_rects[r][0].x; p0.y = robot.static_rects[r][0].y;
                rect_mk.points.push_back(p0);
                marker_array.markers.push_back(rect_mk);
            }
        }

        pub_markers_->publish(marker_array);
    }

    // -------------------------------------------------------------------------
    // 4. FUNÇÕES MATEMÁTICAS AUXILIARES
    // -------------------------------------------------------------------------
    
    // Interpola pose no tempo T
    InterpolationResult calculate_pose_at_time(const RobotSimData& robot, double sim_time) 
    {
        InterpolationResult res;
        res.current_segment_index = -1;

        if (sim_time >= robot.total_duration) 
        {
            res.pose = robot.path_msg.poses.back().pose;
            res.pose.position.z = 0.15;
            res.current_segment_index = robot.static_rects.size() - 1; 
            return res;
        }

        double dist_target = sim_time * robot.speed;

        for (size_t i = 0; i < robot.accumulated_distances.size() - 1; ++i) 
        {
            double d_start = robot.accumulated_distances[i];
            double d_end = robot.accumulated_distances[i+1];

            if (dist_target >= d_start && dist_target <= d_end) 
            {
                double segment_len = d_end - d_start;
                double ratio = (segment_len > 0.001) ? (dist_target - d_start) / segment_len : 0.0;

                auto p1 = robot.path_msg.poses[i].pose.position;
                auto p2 = robot.path_msg.poses[i+1].pose.position;

                res.pose.position.x = p1.x + (p2.x - p1.x) * ratio;
                res.pose.position.y = p1.y + (p2.y - p1.y) * ratio;
                res.pose.position.z = 0.15;

                double yaw = std::atan2(p2.y - p1.y, p2.x - p1.x);
                tf2::Quaternion q; q.setRPY(0, 0, yaw);
                res.pose.orientation = tf2::toMsg(q);
                
                res.current_segment_index = i;
                return res;
            }
        }
        res.pose = robot.path_msg.poses[0].pose;
        res.current_segment_index = 0;
        return res;
    }

    // Cria Bounding Box Dinâmica (Baseada na Pose Interpolada)
    Rectangle calculate_dynamic_box(const geometry_msgs::msg::Pose& pose) 
    {
        tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
        tf2::Matrix3x3 m(q); double r, p, yaw; m.getRPY(r, p, yaw);

        double half = ROBOT_WIDTH / 2.0;
        std::vector<Vec2> corners = {{half, half}, {half, -half}, {-half, -half}, {-half, half}};
        Rectangle rect;
        double c = std::cos(yaw), s = std::sin(yaw);
        for (const auto& pt : corners) 
        {
            rect.push_back({pose.position.x + pt.x * c - pt.y * s, pose.position.y + pt.x * s + pt.y * c});
        }
        return rect;
    }

    
    Rectangle calculate_static_rect(const geometry_msgs::msg::Point& p_start, const geometry_msgs::msg::Point& p_end) 
    {
        double dx = p_end.x - p_start.x, dy = p_end.y - p_start.y;
        double len = std::hypot(dx, dy);
        if (len < 0.001) return {};
        Vec2 u = { dx / len, dy / len }; Vec2 n = { u.y, -u.x };
        
        Vec2 s = { p_start.x - u.x*LONGITUDINAL_PAD, p_start.y - u.y*LONGITUDINAL_PAD };
        Vec2 e = { p_end.x + u.x*LONGITUDINAL_PAD, p_end.y + u.y*LONGITUDINAL_PAD };
        double hw = ROBOT_WIDTH / 2.0;
        
        return {{e.x + n.x*hw, e.y + n.y*hw}, {e.x - n.x*hw, e.y - n.y*hw}, 
                {s.x - n.x*hw, s.y - n.y*hw}, {s.x + n.x*hw, s.y + n.y*hw}};
    }

    // Teorema do Eixo de Separação (SAT)
    bool check_sat_intersection(const Rectangle& r1, const Rectangle& r2) 
    {
        std::vector<Rectangle> polys = {r1, r2};
        for (const auto& poly : polys) 
        {
            for (size_t i = 0; i < poly.size(); ++i) 
            {
                Vec2 p1 = poly[i], p2 = poly[(i+1)%poly.size()];

                Vec2 edge = p2 - p1; Vec2 normal = {-edge.y, edge.x};
                
                double min1, max1, min2, max2;
                project_polygon(normal, r1, min1, max1);
                project_polygon(normal, r2, min2, max2);
                if (max1 < min2 || max2 < min1) 
                {
                    return false; 
                }
            }
        }
        return true;
    }

    void project_polygon(const Vec2& axis, const Rectangle& poly, double& min, double& max) 
    {
        min = std::numeric_limits<double>::infinity(); max = -std::numeric_limits<double>::infinity();
        for (const auto& p : poly) 
        {
            double proj = p.dot(axis);
            if (proj < min)
            {
                min = proj; 
            } 
            if (proj > max) 
            {
                max = proj;
            }
        }
    }

    rclcpp::Subscription<mobile_manipulation_interfaces::msg::FleetPaths>::SharedPtr sub_paths_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::TimerBase::SharedPtr anim_timer_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FleetManagement>());
    rclcpp::shutdown();
    return 0;
}