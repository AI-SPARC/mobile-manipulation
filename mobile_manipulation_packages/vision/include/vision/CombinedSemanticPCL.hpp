#ifndef VISION__COMBINED_SEMANTIC_PCL_HPP_
#define VISION__COMBINED_SEMANTIC_PCL_HPP_

#include <rclcpp/rclcpp.hpp>

// Mensagens Padrão
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

// Mensagens de Visão e Customizadas
#include <vision_msgs/msg/detection3_d_array.hpp>
#include "mobile_manipulation_interfaces/msg/object_data.hpp"

// Message Filters (Sincronização)
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// C++ Standard
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <mutex>
#include <array>

namespace semantic_pcl
{

class CombinedSemanticPCL : public rclcpp::Node
{
public:
    explicit CombinedSemanticPCL(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    virtual ~CombinedSemanticPCL() = default;

private:
    // --- Callbacks ---
    
    // Recebe o dicionário de labels (JSON) do Isaac Sim
    void labelsCallback(const std_msgs::msg::String::SharedPtr msg);

    // Recebe as Bounding Boxes 3D (assíncrono em relação à nuvem)
    void detectionsCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
    
    // Callback principal sincronizado (Imagem de Segmentação + Nuvem de Profundidade)
    void syncedCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr & seg_msg,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pcl_msg);

    // --- Métodos Auxiliares ---

    void parseLabelsJson(const std::string & json_str);
    
    std::tuple<uint8_t, uint8_t, uint8_t> getColorForId(int32_t obj_id);

    // Cria uma mensagem PointCloud2 a partir de um vetor de pontos XYZ
    sensor_msgs::msg::PointCloud2 createPCLMsg(
        const std::vector<std::array<float, 3>>& points, 
        const std_msgs::msg::Header& header);

    void publishSemanticPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

    void publishColoredPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

    // --- Subscribers ---

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr labels_sub_;
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr detections_sub_;
    
    // Message Filters
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> seg_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pcl_sub_;

    // Política de Sincronização
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::PointCloud2
    > SyncPolicy;
    
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // --- Publishers ---

    // Publica nuvem completa com campo 'semantic_id'
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_semantic_;
    // Publica nuvem completa colorida (RGB) para RViz
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_colored_;
    // Publica dados isolados de cada objeto (Objeto + Nuvem Recortada)
    rclcpp::Publisher<mobile_manipulation_interfaces::msg::ObjectData>::SharedPtr pub_object_data_;

    // --- TF2 ---
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // --- Dados e Parâmetros ---

    std::string target_frame_;
    std::string topic_segmentation_;
    std::string topic_pointcloud_;
    std::string topic_labels_;
    std::string topic_output_semantic_;
    std::string topic_output_colored_;
    
    int downsample_step_;
    size_t frame_count_;

    // Mapas de ID -> Info
    std::map<int32_t, std::string> id_to_label_;
    
    // Busca Reversa (Nome -> ID). ESSENCIAL PARA O CÓDIGO FUNCIONAR
    std::map<std::string, int32_t> label_to_id_;

    std::map<int32_t, std::tuple<uint8_t, uint8_t, uint8_t>> color_map_;

    // Armazenamento Thread-Safe das Detecções
    std::mutex detections_mutex_;
    vision_msgs::msg::Detection3DArray::SharedPtr last_detections_;
};

} // namespace semantic_pcl

#endif // VISION__COMBINED_SEMANTIC_PCL_HPP_