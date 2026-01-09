#ifndef MANIPULATION__SCAN_OBJECT_HPP_
#define MANIPULATION__SCAN_OBJECT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <mobile_manipulation_interfaces/msg/semantic_pcl.hpp> // Ajuste conforme seu pacote
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <mutex>
#include <string>
#include <vector>
#include <map>

namespace manipulation {

// Estrutura para chaveamento do mapa de voxels
struct VoxelKey {
    int x, y, z;

    // Necessário para usar como chave em std::map
    bool operator<(const VoxelKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

// Estrutura auxiliar interna para guardar os pontos brutos e metadados
struct ScanPoint {
    tf2::Vector3 position;
    tf2::Vector3 target_center; // Necessário para calcular a orientação da Pose
    int face_id;
};

// Dados armazenados por objeto detectado
struct ObjectData {
    std::string label;
    vision_msgs::msg::Detection3D detection;
    std_msgs::msg::Header header;
    std::vector<ScanPoint> valid_scan_grid;
};

class ScanObject : public rclcpp::Node
{
public:
    explicit ScanObject(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ScanObject() = default;

    std::vector<geometry_msgs::msg::Pose> getSortedScanPoses(const std::string& label);

private:
    // --- Parameters ---
    std::string target_frame_;
    std::string odom_topic_;
    double ray_length_;
    double grid_resolution_;
    double voxel_map_resolution_;
    double ray_step_size_;
    std::string target_object_id_;
    bool publish_markers_; // Controle de visualização

    // --- State ---
    tf2::Vector3 robot_position_;
    bool robot_pos_received_;
    std::mutex robot_pos_mutex_;

    // Voxel Map (Substituição leve para Octomap com O(1) médio de acesso via Hash ou O(log n) via Map)
    std::map<VoxelKey, std::string> voxel_grid_;
    std::mutex voxel_mutex_;

    // Objects Data
    std::map<std::string, ObjectData> detected_objects_;
    std::mutex objects_mutex_;
    std_msgs::msg::Header last_header_;

    // Animation / Publishing State
    // IMPORTANTE: Agora armazena Poses completas, não apenas Pontos
    std::vector<geometry_msgs::msg::Pose> poses_to_animate_; 
    size_t current_anim_index_;
    bool is_animating_;
    vision_msgs::msg::Detection3D current_anim_bbox_;
    std::mutex anim_mutex_;

    // --- ROS Interfaces ---
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_detections_;
    rclcpp::Subscription<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr sub_semantic_pcl_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
    
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr animation_timer_;

    // --- Callbacks ---
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
    void semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg);
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void animationTimerCallback();

    // --- Core Logic ---
    
    // Gera os pontos candidatos ao redor do objeto
    std::vector<ScanPoint> computeValidScanningGrid(
        const geometry_msgs::msg::Pose& target_pose,
        const geometry_msgs::msg::Vector3& target_size,
        const std::string& target_label);

    // Verifica colisão no Grid de Voxels (Ray Marching simplificado)
    bool isRayBlocked(
        const tf2::Vector3& start, 
        const tf2::Vector3& end, 
        const std::string& target_label);

    // Converte posição física para índice do Voxel
    VoxelKey pointToVoxel(const tf2::Vector3& pt);

   
};

} // namespace manipulation

#endif // MANIPULATION__SCAN_OBJECT_HPP_