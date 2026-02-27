#ifndef SLAM_CORE_MAPPING_HPP_
#define SLAM_CORE_MAPPING_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/opencv.hpp>
#include <gtsam/geometry/Pose3.h>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <vector>
#include <string>

namespace slam_core
{

// Estrutura para representar a chave de um Voxel na Voxel Grid global
struct VoxelKey {
    int x, y, z;
    uint8_t r, g, b;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

} // namespace slam_core

// Função de Hash especializada para VoxelKey (necessária para std::unordered_set)
namespace std {
    template <>
    struct hash<slam_core::VoxelKey> {
        size_t operator()(const slam_core::VoxelKey& k) const {
            // Combina as coordenadas x, y, z em um hash único
            size_t h1 = std::hash<int>()(k.x);
            size_t h2 = std::hash<int>()(k.y);
            size_t h3 = std::hash<int>()(k.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

namespace slam_core
{

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
    ~Mapping() override;

    // Atualizado: recebe a mensagem, o main_frame_id e a escala de profundidade
    void set_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cam_info, 
                         const std::string& main_frame_id, 
                         float depth_scale);

    void add_keyframe_data(int kf_id, const cv::Mat& rgb_img, const cv::Mat& depth_img);
    void update_global_map(const std::vector<std::pair<int, gtsam::Pose3>>& optimized_poses);

private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr generate_local_cloud(const cv::Mat& rgb, const cv::Mat& depth);
    void publish_map_callback();

    // Parâmetros ROS
    float voxel_leaf_size_;
    double map_publish_rate_;

    // Câmera e TFs
    bool camera_initialized_ = false;
    float fx_, fy_, cx_, cy_, depth_scale_;
    
    std::string main_frame_id_;
    std::string camera_frame_id_;
    
    bool tf_main_camera_initialized_;
    Eigen::Matrix4f T_main_camera_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

    // Mapa e Dados
    std::unordered_map<int, KeyframeData> keyframe_database_;
    std::unordered_set<VoxelKey> voxel_occupancy_set_;
    std::mutex map_mutex_;

    // Publicadores e Timers
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
    rclcpp::TimerBase::SharedPtr map_publish_timer_;
};

} // namespace slam_core

#endif // SLAM_CORE_MAPPING_HPP_