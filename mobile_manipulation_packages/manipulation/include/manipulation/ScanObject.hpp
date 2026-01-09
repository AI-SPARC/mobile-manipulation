#ifndef MANIPULATION__SCAN_OBJECT_HPP_
#define MANIPULATION__SCAN_OBJECT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

// Includes de mensagens customizadas
#include "mobile_manipulation_interfaces/msg/object_data.hpp"
#include "mobile_manipulation_interfaces/msg/semantic_pcl.hpp" 

#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <memory>
#include <utility> // Para std::pair

namespace manipulation {

// --- VOXEL HASHING ---
struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& other) const { 
        return x == other.x && y == other.y && z == other.z; 
    }
};

struct VoxelHash {
    std::size_t operator()(const VoxelKey& k) const {
        size_t h1 = std::hash<int>{}(k.x); 
        size_t h2 = std::hash<int>{}(k.y); 
        size_t h3 = std::hash<int>{}(k.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2); 
    }
};

// --- DADOS DO PONTO DE VARREDURA ---
struct ScanPoint {
    tf2::Vector3 position;        // Ponto da Câmera (Origem do Raio)
    tf2::Vector3 target_center;   // Centro do Objeto (Destino teórico)
    tf2::Vector3 surface_contact; // Ponto REAL onde o raio tocou o objeto
    int face_id;                  // ID da face do cubo para agrupamento
};

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
    virtual ~ScanObject() = default;

    // Retorna PAR de vetores ordenados:
    // first: Poses da Câmera (Orientadas para o objeto)
    // second: Pontos de contato na superfície
    std::pair<std::vector<geometry_msgs::msg::Pose>, std::vector<geometry_msgs::msg::Point>> 
    getSortedScanPoses(const std::string& label);

private:
    // Callbacks
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg);
    void animationTimerCallback();

    // Lógica Geométrica e Ray Casting
    std::vector<ScanPoint> computeValidScanningGrid(
        const geometry_msgs::msg::Pose& target_pose,
        const geometry_msgs::msg::Vector3& target_size,
        const std::string& target_label);

    enum class RayResult { BLOCKED, HIT_TARGET, MISS };
    
    // Analisa o raio passo-a-passo
    std::pair<RayResult, tf2::Vector3> analyzeRay(
        const tf2::Vector3& start, 
        const tf2::Vector3& end, 
        const std::string& target_label);

    // Calcula Quaternion LookAt (Apontando para o alvo)
    geometry_msgs::msg::Quaternion computeLookAtOrientation(
        const tf2::Vector3& camera_pos, 
        const tf2::Vector3& target_pos);

    VoxelKey pointToVoxel(const tf2::Vector3& pt);

    // Membros ROS
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_detections_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
    rclcpp::Subscription<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr sub_semantic_pcl_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr animation_timer_;

    // Dados
    std::mutex robot_pos_mutex_;
    tf2::Vector3 robot_position_;
    bool robot_pos_received_;

    std::mutex objects_mutex_;
    std::map<std::string, ObjectData> detected_objects_;
    std_msgs::msg::Header last_header_;

    std::mutex voxel_mutex_;
    std::unordered_map<VoxelKey, std::string, VoxelHash> voxel_grid_;

    // Controle de Animação
    std::mutex anim_mutex_;
    // Armazena POSE para desenhar setas
    std::vector<std::pair<geometry_msgs::msg::Pose, geometry_msgs::msg::Point>> poses_to_animate_; 
    vision_msgs::msg::Detection3D current_anim_bbox_;
    size_t current_anim_index_; 
    bool is_animating_; 

    // Parâmetros
    std::string target_frame_;
    std::string odom_topic_;
    double ray_length_;
    double grid_resolution_;
    double ray_step_size_;
    std::string target_object_id_;
    double voxel_map_resolution_; 
    bool publish_markers_; // Novo parâmetro
};

} // namespace manipulation

#endif // MANIPULATION__SCAN_OBJECT_HPP_