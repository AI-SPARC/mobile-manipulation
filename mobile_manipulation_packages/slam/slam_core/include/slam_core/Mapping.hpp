#ifndef SLAM_CORE__MAPPING_HPP_
#define SLAM_CORE__MAPPING_HPP_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

// TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

// OpenCV, GTSAM, PCL
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <gtsam/geometry/Pose3.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <mutex>
#include <cmath>

namespace slam_core
{

// =========================================================
// O MAPA GLOBAL NATIVO (VOXEL HASHING)
// =========================================================
struct VoxelKey {
    int x, y, z;
    uint8_t r, g, b; // Guardamos a cor para não perder o visual no RViz!

    // A igualdade checa APENAS a posição espacial. 
    // Se outro ponto cair no mesmo voxel, ele é descartado (O(1)).
    bool operator==(const VoxelKey& other) const {
        return (x == other.x && y == other.y && z == other.z);
    }
};

struct VoxelHash {
    std::size_t operator()(const VoxelKey& k) const {
        // Primos imensos para garantir que as coordenadas espaciais 
        // sejam espalhadas de forma uniforme na memória RAM
        std::size_t p1 = 73856093;
        std::size_t p2 = 19349663;
        std::size_t p3 = 83492791;
        
        // Multiplicação seguida de XOR cria uma chave praticamente única
        return (static_cast<std::size_t>(k.x) * p1) ^ 
               (static_cast<std::size_t>(k.y) * p2) ^ 
               (static_cast<std::size_t>(k.z) * p3);
    }
};

struct KeyframeData {
    cv::Mat rgb;
    cv::Mat depth;
    gtsam::Pose3 pose; 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr local_cloud; 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr global_cloud_cache; 
};

class Mapping : public rclcpp::Node
{
public:
    explicit Mapping(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~Mapping();

    void set_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cam_info, float depth_scale = 1000.0f);
    void add_keyframe_data(int kf_id, const cv::Mat& rgb_img, const cv::Mat& depth_img);
    void update_global_map(const std::vector<std::pair<int, gtsam::Pose3>>& optimized_poses);

private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr generate_local_cloud(const cv::Mat& rgb, const cv::Mat& depth);
    void publish_map_callback();

    std::unordered_map<int, KeyframeData> keyframe_database_;
    
    // =========================================================
    // VARIÁVEIS DE THREADING, HASHING E PUBLICAÇÃO
    // =========================================================
    // Este é o ÚNICO mapa global agora. Adeus, persistent_global_map_!
    std::unordered_set<VoxelKey, VoxelHash> voxel_occupancy_set_;
    std::mutex map_mutex_;
    
    rclcpp::TimerBase::SharedPtr map_publish_timer_;
    double map_publish_rate_; 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

    // =========================================================
    // VARIÁVEIS DO TF2 E FRAMES DINÂMICOS
    // =========================================================
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    std::string main_frame_id_;
    std::string camera_frame_id_;

    bool tf_main_camera_initialized_;
    Eigen::Matrix4f T_main_camera_;

    // =========================================================
    // INTRÍNSECOS DA CÂMERA
    // =========================================================
    bool camera_initialized_ = false;
    double fx_, fy_, cx_, cy_;
    float depth_scale_;
    float voxel_leaf_size_;
};

} // namespace slam_core

#endif // SLAM_CORE__MAPPING_HPP_