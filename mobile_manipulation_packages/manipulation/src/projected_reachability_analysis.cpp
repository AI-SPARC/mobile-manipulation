#include <manipulation/ProjectedReachabilityAnalysis.hpp>
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "rclcpp_components/register_node_macro.hpp"

using namespace std::chrono_literals;

namespace manipulation {

ProjectedReachabilityAnalysis::ProjectedReachabilityAnalysis(const rclcpp::NodeOptions & options)
: Node("gripper_monitor_node", options) 
{
    RCLCPP_INFO(this->get_logger(), "Gripper Monitor Node inicializado (Composable).");

    marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("reachability_visualization", 10);
}   

double ProjectedReachabilityAnalysis::calculate_max_2d_radius(const geometry_msgs::msg::Pose& pose, const double& ROBOT_BASE_Z, const double& MAX_REACH_3D)
{
    // Cateto vertical.
    double vertical_dist = std::abs(pose.position.z - ROBOT_BASE_Z);

    if (vertical_dist > MAX_REACH_3D) 
    {
        RCLCPP_WARN(this->get_logger(), "Objeto inalcançável! Altura relativa (%.2f) > Alcance Max (%.3f)", vertical_dist, MAX_REACH_3D);
        return 0.0; 
    }

    // Cateto horizontal. MAX_REACH_3D é a hipotenusa é 0.9.
    // r_2d = sqrt(R^2 - dz^2)
    double radius_2d = std::sqrt(std::pow(MAX_REACH_3D, 2) - std::pow(vertical_dist, 2));

    RCLCPP_INFO(this->get_logger(), "Raio 2D no chão: %.4f m (Centro X: %.2f, Y: %.2f)", 
                radius_2d, pose.position.x, pose.position.y);


    // --- INÍCIO DA VISUALIZAÇÃO ---

    visualization_msgs::msg::MarkerArray marker_array;
    rclcpp::Time current_time = this->now();

    auto create_base_marker = [&](int id, int type, std::string ns) 
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = "world"; 
        m.header.stamp = current_time;
        m.ns = ns;
        m.id = id;
        m.type = type;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.orientation.w = 1.0; // Orientação padrão neutra
        m.color.a = 1.0; // Alfa padrão (sólido)
        return m;
    };


    // === MARCADOR 1: O Disco Verde no Chão (Seu código original) ===
    visualization_msgs::msg::Marker disk_marker = create_base_marker(0, visualization_msgs::msg::Marker::CYLINDER, "reach_zone");
    disk_marker.pose.position.x = pose.position.x;
    disk_marker.pose.position.y = pose.position.y;
    disk_marker.pose.position.z = 0.0; // No chão
    disk_marker.scale.x = radius_2d * 2.0; // Diâmetro
    disk_marker.scale.y = radius_2d * 2.0; // Diâmetro
    disk_marker.scale.z = 0.015;            // Altura fina
    disk_marker.color.a = 0.3; // Translúcido
    disk_marker.color.b = 1.0; // Verde
    marker_array.markers.push_back(disk_marker);


    // === MARCADOR 2: Quadrado Vermelho no Objeto Alvo ===
    visualization_msgs::msg::Marker target_cube = create_base_marker(1, visualization_msgs::msg::Marker::CUBE, "target_obj");
    target_cube.pose = pose; // Usa a pose exata do objeto
    target_cube.scale.x = 0.05; // Cubo de 5cm
    target_cube.scale.y = 0.05;
    target_cube.scale.z = 0.05;
    target_cube.color.r = 1.0; // Vermelho
    marker_array.markers.push_back(target_cube);


    // --- Definindo os Pontos do Triângulo de Pitágoras ---
    // Para visualizar corretamente a matemática que fizemos (hipotenusa = 0.8),
    // o triângulo deve ser entre a altura do objeto e a altura da base do robô (0.11).
    geometry_msgs::msg::Point p_top = pose.position; // O objeto (x,y,z)
    
    geometry_msgs::msg::Point p_bottom_ref; // Ponto (x,y) na altura da base do robô
    p_bottom_ref.x = pose.position.x;
    p_bottom_ref.y = pose.position.y;
    p_bottom_ref.z = ROBOT_BASE_Z; // 0.11

    geometry_msgs::msg::Point p_horiz_end; // Fim do cateto horizontal
    // Estendemos arbitrariamente ao longo do eixo X para desenhar o triângulo
    p_horiz_end.x = pose.position.x + radius_2d; 
    p_horiz_end.y = pose.position.y;
    p_horiz_end.z = ROBOT_BASE_Z; // 0.11


    // === MARCADOR 3: Linha do Cateto Vertical (Diferença de Altura) ===
    visualization_msgs::msg::Marker vert_line = create_base_marker(2, visualization_msgs::msg::Marker::LINE_STRIP, "triangle_lines");
    vert_line.scale.x = 0.01; // Espessura da linha
    vert_line.color.b = 1.0; // Azul (geralmente representa Z)
    vert_line.points.push_back(p_top);
    vert_line.points.push_back(p_bottom_ref);
    marker_array.markers.push_back(vert_line);

    // === MARCADOR 4: Linha do Cateto Horizontal (Raio 2D) ===
    visualization_msgs::msg::Marker horiz_line = create_base_marker(3, visualization_msgs::msg::Marker::LINE_STRIP, "triangle_lines");
    horiz_line.scale.x = 0.01;
    horiz_line.color.r = 1.0; // Vermelho (geralmente representa X)
    horiz_line.points.push_back(p_bottom_ref);
    horiz_line.points.push_back(p_horiz_end);
    marker_array.markers.push_back(horiz_line);

    // === MARCADOR 5: Linha da Hipotenusa (Alcance Max 3D) ===
    visualization_msgs::msg::Marker hypot_line = create_base_marker(4, visualization_msgs::msg::Marker::LINE_STRIP, "triangle_lines");
    hypot_line.scale.x = 0.01;
    hypot_line.color.g = 1.0; 
    hypot_line.color.r = 1.0; // Amarelo (Vermelho + Verde)
    hypot_line.points.push_back(p_top);
    hypot_line.points.push_back(p_horiz_end);
    marker_array.markers.push_back(hypot_line);




    // PUBLICAR TUDO DE UMA VEZ
    marker_publisher_->publish(marker_array);

    return radius_2d;
}

} // namespace manipulation

RCLCPP_COMPONENTS_REGISTER_NODE(manipulation::ProjectedReachabilityAnalysis)