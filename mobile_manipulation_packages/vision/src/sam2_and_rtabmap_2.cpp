#include <memory>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

// ==============================================================================
// 1. ESTRUTURAS DE DADOS
// ==============================================================================
struct PointXYZRGB {
    float x, y, z;
    uint8_t r, g, b;
};

struct Voxel {
    int x, y, z;
    bool operator==(const Voxel& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelHash {
    std::size_t operator()(const Voxel& v) const {
        constexpr std::size_t p1 = 73856093;
        constexpr std::size_t p2 = 19349663;
        constexpr std::size_t p3 = 83492791;
        return (static_cast<std::size_t>(v.x) * p1) ^ 
               (static_cast<std::size_t>(v.y) * p2) ^ 
               (static_cast<std::size_t>(v.z) * p3);
    }
};

// ==============================================================================
// 2. CLASSE PRINCIPAL
// ==============================================================================
class SegmentedCloudExtractor : public rclcpp::Node
{
public:
    SegmentedCloudExtractor()
    : Node("segmented_cloud_extractor")
    {
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info", 1, 
            std::bind(&SegmentedCloudExtractor::camera_info_callback, this, std::placeholders::_1));

        mask_sub_.subscribe(this, "/gsam2/instance_mask", rmw_qos_profile_sensor_data);
        depth_sub_.subscribe(this, "/camera/camera/aligned_depth_to_color/image_raw", rmw_qos_profile_sensor_data);
        rgb_sub_.subscribe(this, "/camera/camera/color/image_raw", rmw_qos_profile_sensor_data);

        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(
            SyncPolicy(100), mask_sub_, depth_sub_, rgb_sub_));
        
        sync_->registerCallback(
            std::bind(&SegmentedCloudExtractor::sync_callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

        segmented_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/gsam2/segmented_cloud", 10);

        RCLCPP_INFO(this->get_logger(), "Nó iniciado! Duplo Filtro (Cor da Mesa + Flying Pixels) ATIVADO.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;

    message_filters::Subscriber<sensor_msgs::msg::Image> mask_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr segmented_cloud_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::unordered_map<Voxel, PointXYZRGB, VoxelHash> accumulated_points_;
    const float voxel_size_ = 0.005f; 

    // --- PARÂMETROS DE FILTRAGEM ---
    const uint8_t min_brightness_ = 100;       // Filtro de Cor: Brilho mínimo da mesa
    const uint8_t max_saturation_diff_ = 40;   // Filtro de Cor: Quão neutra é a cor da mesa
    const int max_spatial_jump_mm_ = 30;       // Flying Pixels: Salto máximo permitido entre pixels vizinhos (3 cm)

    // ==============================================================================
    // 3. CALLBACKS
    // ==============================================================================
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (!camera_info_) {
            camera_info_ = msg;
            RCLCPP_INFO(this->get_logger(), "Matriz intrínseca recebida.");
        }
    }

    void sync_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& mask_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg)
    {
        if (!camera_info_) return;

        uint32_t width = depth_msg->width;
        uint32_t height = depth_msg->height;

        if (mask_msg->width != width || rgb_msg->width != width) return;

        const uint8_t* mask_data = mask_msg->data.data();
        const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth_msg->data.data());
        const uint8_t* rgb_data = rgb_msg->data.data();

        // ======================================================================
        // PASSO 1: DUPLA FILTRAGEM (COR DA MESA E FLYING PIXELS)
        // ======================================================================
        size_t valid_points = 0;
        std::vector<bool> valid_mask(width * height, false);

        for (uint32_t v = 0; v < height; ++v) {
            for (uint32_t u = 0; u < width; ++u) {
                uint32_t i = v * width + u;

                if (mask_data[i] > 0 && depth_data[i] > 0) {
                    
                    // --- FILTRO 1: CHROMA KEY (MESA BRANCA/CINZA) ---
                    uint8_t r = rgb_data[i * 3];
                    uint8_t g = rgb_data[i * 3 + 1];
                    uint8_t b = rgb_data[i * 3 + 2];

                    uint8_t max_c = std::max({r, g, b});
                    uint8_t min_c = std::min({r, g, b});

                    if (max_c > min_brightness_ && (max_c - min_c) < max_saturation_diff_) {
                        continue; // Aniquila o pixel da mesa!
                    }

                    // --- FILTRO 2: FLYING PIXELS (GRADIENTE ESPACIAL) ---
                    int depth = static_cast<int>(depth_data[i]);
                    bool is_flying_pixel = false;
                    
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) continue; // Pula o próprio pixel
                            
                            int nu = u + dx;
                            int nv = v + dy;
                            
                            // Garante que o vizinho está dentro dos limites da imagem
                            if (nu >= 0 && nu < (int)width && nv >= 0 && nv < (int)height) {
                                int neighbor_depth = static_cast<int>(depth_data[nv * width + nu]);
                                
                                // Se o vizinho tem profundidade e o salto espacial for brutal, é borda falsa!
                                if (neighbor_depth > 0 && std::abs(depth - neighbor_depth) > max_spatial_jump_mm_) {
                                    is_flying_pixel = true;
                                    break;
                                }
                            }
                        }
                        if (is_flying_pixel) break;
                    }

                    if (is_flying_pixel) continue; // Destrói o flying pixel!

                    // Se sobreviveu a tudo, faz parte da ferramenta real
                    valid_mask[i] = true;
                    valid_points++;
                }
            }
        }

        if (valid_points == 0) return;

        // ======================================================================
        // PASSO 2: ALOCAÇÃO E DESPROJEÇÃO
        // ======================================================================
        sensor_msgs::msg::PointCloud2 local_cloud;
        local_cloud.header = depth_msg->header; 
        local_cloud.height = 1;              
        local_cloud.width = valid_points;
        local_cloud.is_dense = true;

        sensor_msgs::PointCloud2Modifier modifier(local_cloud);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(valid_points);

        sensor_msgs::PointCloud2Iterator<float> out_x(local_cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(local_cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(local_cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> out_r(local_cloud, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> out_g(local_cloud, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> out_b(local_cloud, "b");

        float fx = camera_info_->k[0];
        float cx = camera_info_->k[2];
        float fy = camera_info_->k[4];
        float cy = camera_info_->k[5];

        for (uint32_t v = 0; v < height; ++v) {
            for (uint32_t u = 0; u < width; ++u) {
                uint32_t i = v * width + u;
                
                if (valid_mask[i]) {
                    float z = static_cast<float>(depth_data[i]) / 1000.0f;
                    
                    *out_x = (static_cast<float>(u) - cx) * z / fx;
                    *out_y = (static_cast<float>(v) - cy) * z / fy;
                    *out_z = z;

                    *out_r = rgb_data[i * 3];
                    *out_g = rgb_data[i * 3 + 1];
                    *out_b = rgb_data[i * 3 + 2];

                    ++out_x; ++out_y; ++out_z; ++out_r; ++out_g; ++out_b;
                }
            }
        }

        // ======================================================================
        // PASSO 3: TRANSFORMAÇÃO TF2 E ACÚMULO
        // ======================================================================
        sensor_msgs::msg::PointCloud2 transformed_cloud;
        try {
            geometry_msgs::msg::TransformStamped transform = 
                tf_buffer_->lookupTransform("map", depth_msg->header.frame_id, depth_msg->header.stamp, rclcpp::Duration::from_seconds(0.1));
            tf2::doTransform(local_cloud, transformed_cloud, transform);
        } 
        catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "Aguardando TF: %s", ex.what());
            return;
        }

        sensor_msgs::PointCloud2ConstIterator<float> trans_x(transformed_cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> trans_y(transformed_cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> trans_z(transformed_cloud, "z");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> trans_r(transformed_cloud, "r");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> trans_g(transformed_cloud, "g");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> trans_b(transformed_cloud, "b");

        for (; trans_x != trans_x.end(); ++trans_x, ++trans_y, ++trans_z, ++trans_r, ++trans_g, ++trans_b) {
            if (std::isnan(*trans_x)) continue;

            Voxel v;
            v.x = static_cast<int>(std::floor(*trans_x / voxel_size_));
            v.y = static_cast<int>(std::floor(*trans_y / voxel_size_));
            v.z = static_cast<int>(std::floor(*trans_z / voxel_size_));

            if (accumulated_points_.find(v) == accumulated_points_.end()) {
                accumulated_points_[v] = {*trans_x, *trans_y, *trans_z, *trans_r, *trans_g, *trans_b};
            }
        }

        // ======================================================================
        // PASSO 4: PUBLICAÇÃO
        // ======================================================================
        sensor_msgs::msg::PointCloud2 pub_cloud;
        pub_cloud.header.frame_id = "map";
        pub_cloud.header.stamp = this->now();
        pub_cloud.height = 1;
        pub_cloud.width = accumulated_points_.size();
        pub_cloud.is_dense = true;

        sensor_msgs::PointCloud2Modifier pub_modifier(pub_cloud);
        pub_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        pub_modifier.resize(accumulated_points_.size());

        sensor_msgs::PointCloud2Iterator<float> pub_x(pub_cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> pub_y(pub_cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> pub_z(pub_cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> pub_r(pub_cloud, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> pub_g(pub_cloud, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> pub_b(pub_cloud, "b");

        for (const auto& [voxel, pt] : accumulated_points_) {
            *pub_x = pt.x; *pub_y = pt.y; *pub_z = pt.z;
            *pub_r = pt.r; *pub_g = pt.g; *pub_b = pt.b;
            ++pub_x; ++pub_y; ++pub_z; ++pub_r; ++pub_g; ++pub_b;
        }

        segmented_cloud_pub_->publish(pub_cloud);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SegmentedCloudExtractor>());
    rclcpp::shutdown();
    return 0;
}