#ifndef VISION_COMBINED_SEMANTIC_PCL_HPP_
#define VISION_COMBINED_SEMANTIC_PCL_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <regex>
#include <mutex>
#include <array>

// Interface customizada
#include "mobile_manipulation_interfaces/msg/semantic_pcl.hpp"

namespace semantic_pcl
{

class CombinedSemanticPCL : public rclcpp::Node
{
public:
    explicit CombinedSemanticPCL(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    virtual ~CombinedSemanticPCL() = default;

private:
    // --- Definições de Tipos ---

    // Sincroniza Imagem e PCL da MESMA câmera (isso precisa ser preciso)
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, 
        sensor_msgs::msg::PointCloud2
    >;

    // Estrutura para armazenar o dado mais recente de cada câmera
    struct CameraFrameData 
    {
        sensor_msgs::msg::Image::ConstSharedPtr seg_msg;
        sensor_msgs::msg::PointCloud2::ConstSharedPtr pcl_msg;
        rclcpp::Time receive_time; // Hora que chegou no nó (para timeout)
        bool has_data = false;
    };

    // Módulo para manter os subscribers vivos
    struct CameraModules 
    {
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> seg_sub;
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> pcl_sub;
        std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync;
        rclcpp::Subscription<std_msgs::msg::String>::SharedPtr labels_sub;
    };

    // --- Callbacks ---

    void labelsCallback(const std_msgs::msg::String::SharedPtr msg);
    
    // Callback que recebe o par sincronizado de uma câmera e salva no slot
    void syncedCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr & seg_msg,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pcl_msg,
        int camera_index);

    // Callback do Timer que junta tudo e publica
    void publishEvent();

    // --- Core Logic ---

    // Processa os dados acumulados e transforma para o frame global
    void processAndMerge(
        std::vector<std::array<float, 3>>& out_points,
        std::vector<int32_t>& out_ids,
        rclcpp::Time& out_stamp);

    // --- Helpers ---

    void parseLabelsJson(const std::string & json_str);
    std::string extractCleanLabel(std::string raw_label);
    std::tuple<uint8_t, uint8_t, uint8_t> getColorForId(int32_t obj_id);

    // Funções de Publicação
    void publishSemanticPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

    void publishColoredPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

    void publishSplitSemanticPCL(
        const std::vector<std::array<float, 3>> & points,
        const std::vector<int32_t> & semantic_ids,
        const std_msgs::msg::Header & header);

    sensor_msgs::msg::PointCloud2 createPCLMsg(
        const std::vector<std::array<float, 3>>& points, 
        const std_msgs::msg::Header& header);

    // --- Membros ---

    std::string target_frame_;
    int downsample_step_;
    int num_cameras_;
    double publish_rate_; // Frequência de publicação (Hz)
    double data_timeout_; // Tempo máximo para considerar um dado válido (segundos)

    // TF
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Dados Globais
    std::map<int32_t, std::string> id_to_label_;
    std::map<int32_t, std::tuple<uint8_t, uint8_t, uint8_t>> color_map_;
    
    // Gestão de Câmeras
    std::vector<std::shared_ptr<CameraModules>> cameras_;
    
    // Armazena o frame mais recente de cada câmera
    std::vector<CameraFrameData> latest_frames_; 
    std::mutex data_mutex_; // Protege latest_frames_ e mapas

    // Loop de Publicação
    rclcpp::TimerBase::SharedPtr pub_timer_;

    // Publishers Unificados
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_semantic_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_colored_;
    rclcpp::Publisher<mobile_manipulation_interfaces::msg::SemanticPcl>::SharedPtr pub_custom_msg_;
};

} // namespace semantic_pcl

#endif // VISION_COMBINED_SEMANTIC_PCL_HPP_