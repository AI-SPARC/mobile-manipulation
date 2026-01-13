#include "vision/CombinedSemanticPCL.hpp"

#include <rclcpp/qos.hpp>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <random>
#include <cctype>

// Includes de Transformação e Geometria
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>

namespace semantic_pcl
{

CombinedSemanticPCL::CombinedSemanticPCL(const rclcpp::NodeOptions & options)
    : Node("combined_semantic_pcl", options),
      frame_count_(0)
{
    this->declare_parameter<std::string>("target_frame", "world"); 
    this->declare_parameter<std::string>("topic_segmentation", "/semantic_segmentation");
    this->declare_parameter<std::string>("topic_pointcloud", "/depth_pcl");
    this->declare_parameter<std::string>("topic_labels", "/semantic_labels");
    this->declare_parameter<std::string>("topic_output_semantic", "/semantic_pcl");
    this->declare_parameter<std::string>("topic_output_colored", "/semantic_pcl_colored");
    this->declare_parameter<std::string>("topic_custom_msg", "/semantic_pcl_array"); 
    
    this->declare_parameter<int>("downsample_step", 4); 

    target_frame_ = this->get_parameter("target_frame").as_string();
    topic_segmentation_ = this->get_parameter("topic_segmentation").as_string();
    topic_pointcloud_ = this->get_parameter("topic_pointcloud").as_string();
    topic_labels_ = this->get_parameter("topic_labels").as_string();
    topic_output_semantic_ = this->get_parameter("topic_output_semantic").as_string();
    topic_output_colored_ = this->get_parameter("topic_output_colored").as_string();
    topic_custom_msg_ = this->get_parameter("topic_custom_msg").as_string();
    
    downsample_step_ = this->get_parameter("downsample_step").as_int();
    if (downsample_step_ < 1) downsample_step_ = 1;

    RCLCPP_INFO(this->get_logger(), "=== SEMANTIC PCL NODE (REGEX HYBRID) ===");
    RCLCPP_INFO(this->get_logger(), "Target Frame: %s", target_frame_.c_str());

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    labels_sub_ = this->create_subscription<std_msgs::msg::String>(
        topic_labels_,
        10,
        std::bind(&CombinedSemanticPCL::labelsCallback, this, std::placeholders::_1)
    );

    auto sensor_qos = rclcpp::SensorDataQoS(); 
    auto rmw_qos = sensor_qos.get_rmw_qos_profile();
    
    seg_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        this, topic_segmentation_, rmw_qos);
    
    pcl_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
        this, topic_pointcloud_, rmw_qos);
    
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10), *seg_sub_, *pcl_sub_);
    
    sync_->registerCallback(std::bind(&CombinedSemanticPCL::syncedCallback, this, std::placeholders::_1, std::placeholders::_2));
    
    pub_semantic_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_output_semantic_, 10);
    pub_colored_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_output_colored_, 10);
    pub_custom_msg_ = this->create_publisher<mobile_manipulation_interfaces::msg::SemanticPcl>(topic_custom_msg_, 10);
}

void CombinedSemanticPCL::labelsCallback(const std_msgs::msg::String::SharedPtr msg)
{
    parseLabelsJson(msg->data);
}

// -----------------------------------------------------------------------------
// LÓGICA DE PARSING COM REGEX (Baseada no TranslatorNode + Fix para Strings)
// -----------------------------------------------------------------------------

std::string CombinedSemanticPCL::extractCleanLabel(std::string raw_label)
{
    // Remove aspas se sobrarem
    raw_label.erase(std::remove(raw_label.begin(), raw_label.end(), '\"'), raw_label.end());

    // Se vier "box:redbox_10", pega apenas "redbox_10"
    // Se vier "class:BACKGROUND", pega "BACKGROUND"
    size_t colon_pos = raw_label.find(':');
    if (colon_pos != std::string::npos && colon_pos < raw_label.size() - 1)
    {
        return raw_label.substr(colon_pos + 1);
    }
    return raw_label;
}

void CombinedSemanticPCL::parseLabelsJson(const std::string & json_str)
{
    if (json_str.empty()) return;

    // Regex 1: Formato Objeto (O que seu TranslatorNode usa)
    // Exemplo: "2": { "box": "redbox_10" }
    // Captura 1: ID, Captura 3: Valor
    std::regex object_regex("\"([0-9]+)\"\\s*:\\s*\\{[^}]*\"([A-Za-z0-9_]+)\"\\s*:\\s*\"([A-Za-z0-9_:.\\-]+)\"");

    // Regex 2: Formato String Direta (O que apareceu no seu log de erro)
    // Exemplo: "2": "box:redbox_10"
    // Captura 1: ID, Captura 2: Valor
    std::regex string_regex("\"([0-9]+)\"\\s*:\\s*\"([A-Za-z0-9_:.\\-]+)\"");

    std::smatch match;
    std::string::const_iterator search_start(json_str.cbegin());

    // --- PASSADA 1: Busca Objetos JSON (Prioridade) ---
    while (std::regex_search(search_start, json_str.cend(), match, object_regex))
    {
        try {
            int32_t id = std::stoi(match[1].str());
            std::string val = match[3].str();
            
            // Limpa o label (remove box:, class:, etc)
            id_to_label_[id] = extractCleanLabel(val);
        } catch (...) {}

        search_start = match.suffix().first;
    }

    // --- PASSADA 2: Busca Strings Diretas (Fallback/Complemento) ---
    search_start = json_str.cbegin();
    while (std::regex_search(search_start, json_str.cend(), match, string_regex))
    {
        try {
            int32_t id = std::stoi(match[1].str());
            // Se já achamos via objeto, preferimos o objeto (geralmente mais detalhado)
            // mas se o objeto não existia, pegamos daqui.
            if (id_to_label_.find(id) == id_to_label_.end()) {
                std::string val = match[2].str();
                id_to_label_[id] = extractCleanLabel(val);
            }
        } catch (...) {}
        
        search_start = match.suffix().first;
    }

    // LOG DEBUG (Opcional: Descomente se precisar ver no terminal)
    // if (!id_to_label_.empty()) {
    //     RCLCPP_INFO(this->get_logger(), "Labels parseados: %zu", id_to_label_.size());
    // }
}

// -----------------------------------------------------------------------------

std::tuple<uint8_t, uint8_t, uint8_t> CombinedSemanticPCL::getColorForId(int32_t obj_id)
{
    if (obj_id == 0) return std::make_tuple(50, 50, 50); 
    auto it = color_map_.find(obj_id);
    if (it != color_map_.end()) return it->second;
    
    std::mt19937 gen(obj_id * 137);
    std::uniform_int_distribution<> dis(60, 255);
    uint8_t r = static_cast<uint8_t>(dis(gen));
    uint8_t g = static_cast<uint8_t>(dis(gen));
    uint8_t b = static_cast<uint8_t>(dis(gen));
    auto color = std::make_tuple(r, g, b);
    color_map_[obj_id] = color;
    return color;
}

sensor_msgs::msg::PointCloud2 CombinedSemanticPCL::createPCLMsg(
    const std::vector<std::array<float, 3>>& points, 
    const std_msgs::msg::Header& header)
{
    sensor_msgs::msg::PointCloud2 msg;
    msg.header = header;
    msg.height = 1;
    msg.width = static_cast<uint32_t>(points.size());
    msg.is_dense = true;
    msg.is_bigendian = false;
    msg.fields.resize(3);
    msg.fields[0].name = "x"; msg.fields[0].offset = 0; msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[0].count = 1;
    msg.fields[1].name = "y"; msg.fields[1].offset = 4; msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[1].count = 1;
    msg.fields[2].name = "z"; msg.fields[2].offset = 8; msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32; msg.fields[2].count = 1;
    msg.point_step = 12; 
    msg.row_step = msg.point_step * msg.width;
    msg.data.resize(msg.row_step);
    
    uint8_t* ptr = msg.data.data();
    for(const auto& p : points) {
        std::memcpy(ptr, &p[0], 12);
        ptr += 12;
    }
    return msg;
}


void CombinedSemanticPCL::syncedCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & seg_msg,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pcl_msg)
{
    frame_count_++;

    geometry_msgs::msg::TransformStamped t_stamped;
    bool has_transform = false;
    rclcpp::Time sensor_time = pcl_msg->header.stamp;

    try {
        t_stamped = tf_buffer_->lookupTransform(
            target_frame_, pcl_msg->header.frame_id, sensor_time,
            rclcpp::Duration::from_seconds(0.1));
        has_transform = true;
    } catch (const tf2::TransformException & ex) {}

    if (!has_transform)
    {
        try {
            t_stamped = tf_buffer_->lookupTransform(
                target_frame_, pcl_msg->header.frame_id, tf2::TimePointZero);
            has_transform = true;
        } catch (const tf2::TransformException & ex) {
            if (frame_count_ % 60 == 0) 
                RCLCPP_ERROR(this->get_logger(), "TF Error: %s", ex.what());
            return;
        }
    }

    try
    {
        tf2::Transform transform;
        tf2::fromMsg(t_stamped.transform, transform);

        uint32_t img_height = seg_msg->height;
        uint32_t img_width = seg_msg->width;
        size_t total_pixels = static_cast<size_t>(img_height) * img_width;
        const int32_t * seg_data = reinterpret_cast<const int32_t *>(seg_msg->data.data());

        uint32_t pcl_height = pcl_msg->height;
        uint32_t pcl_width = pcl_msg->width;
        size_t num_points = static_cast<size_t>(pcl_height) * pcl_width;

        int x_off = -1, y_off = -1, z_off = -1;
        for (const auto & f : pcl_msg->fields) {
            if (f.name == "x") x_off = f.offset;
            else if (f.name == "y") y_off = f.offset;
            else if (f.name == "z") z_off = f.offset;
        }
        if (x_off < 0 || y_off < 0 || z_off < 0) return;

        std::vector<std::array<float, 3>> world_points;
        std::vector<int32_t> valid_ids;
        size_t estimated = (num_points / downsample_step_) + 1;
        world_points.reserve(estimated);
        valid_ids.reserve(estimated);

        const uint8_t * pcl_data = pcl_msg->data.data();
        uint32_t point_step = pcl_msg->point_step;

        for (size_t i = 0; i < num_points; i += downsample_step_)
        {
            const uint8_t * ptr = pcl_data + (i * point_step);
            float x_cam, y_cam, z_cam;
            std::memcpy(&x_cam, ptr + x_off, sizeof(float));
            std::memcpy(&y_cam, ptr + y_off, sizeof(float));
            std::memcpy(&z_cam, ptr + z_off, sizeof(float));

            if (!std::isfinite(x_cam) || !std::isfinite(y_cam) || !std::isfinite(z_cam)) continue;
            if (z_cam < 0.1f) continue; 

            tf2::Vector3 point_camera(x_cam, y_cam, z_cam);
            tf2::Vector3 point_world = transform * point_camera;

            int32_t sem_id = 0;
            if (num_points == total_pixels) {
                sem_id = seg_data[i];
            } else {
                size_t idx = static_cast<size_t>((double)i * total_pixels / num_points);
                if (idx >= total_pixels) idx = total_pixels - 1;
                sem_id = seg_data[idx];
            }

            world_points.push_back({
                (float)point_world.x(), (float)point_world.y(), (float)point_world.z()
            });
            valid_ids.push_back(sem_id);
        }

        if (world_points.empty()) return;

        std_msgs::msg::Header output_header = pcl_msg->header;
        output_header.frame_id = target_frame_; 
        
        publishSemanticPCL(world_points, valid_ids, output_header);
        publishColoredPCL(world_points, valid_ids, output_header);
        publishSplitSemanticPCL(world_points, valid_ids, output_header);
    }
    catch (const std::exception & e)
    {
        RCLCPP_ERROR(this->get_logger(), "Erro de processamento: %s", e.what());
    }
}

void CombinedSemanticPCL::publishSplitSemanticPCL(
    const std::vector<std::array<float, 3>> & points,
    const std::vector<int32_t> & semantic_ids,
    const std_msgs::msg::Header & header)
{
    std::map<int32_t, std::vector<std::array<float, 3>>> grouped_points;

    for (size_t i = 0; i < points.size(); ++i)
    {
        int32_t id = semantic_ids[i];
        if (id == 0) continue; 
        grouped_points[id].push_back(points[i]);
    }

    mobile_manipulation_interfaces::msg::SemanticPcl custom_msg;
    custom_msg.header = header;

    for (const auto & [id, pts] : grouped_points)
    {
        if (pts.size() < 10) continue;

        std::string label_str;
        if (id_to_label_.count(id)) 
        {
            label_str = id_to_label_.at(id);
        }
        else
        {
            continue;
        }

        sensor_msgs::msg::PointCloud2 obj_cloud = createPCLMsg(pts, header);
        custom_msg.labels.push_back(label_str);
        custom_msg.clouds.push_back(obj_cloud);
    }

    if (!custom_msg.labels.empty()) {
        pub_custom_msg_->publish(custom_msg);
    }
}

void CombinedSemanticPCL::publishSemanticPCL(
    const std::vector<std::array<float, 3>> & points,
    const std::vector<int32_t> & semantic_ids,
    const std_msgs::msg::Header & header)
{
    auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    msg->header = header; 

    msg->fields.resize(4);
    msg->fields[0].name = "x"; msg->fields[0].offset = 0; msg->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32; msg->fields[0].count = 1;
    msg->fields[1].name = "y"; msg->fields[1].offset = 4; msg->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32; msg->fields[1].count = 1;
    msg->fields[2].name = "z"; msg->fields[2].offset = 8; msg->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32; msg->fields[2].count = 1;
    msg->fields[3].name = "semantic_id"; msg->fields[3].offset = 12; msg->fields[3].datatype = sensor_msgs::msg::PointField::UINT32; msg->fields[3].count = 1;

    msg->point_step = 16;
    msg->height = 1;
    msg->width = static_cast<uint32_t>(points.size());
    msg->row_step = msg->point_step * msg->width;
    msg->is_dense = true;
    msg->is_bigendian = false;
    msg->data.resize(msg->row_step);

    uint8_t * data_ptr = msg->data.data();
    for (size_t i = 0; i < points.size(); ++i) {
        uint8_t * ptr = data_ptr + (i * msg->point_step);
        std::memcpy(ptr + 0, &points[i][0], 4);
        std::memcpy(ptr + 4, &points[i][1], 4);
        std::memcpy(ptr + 8, &points[i][2], 4);
        uint32_t sid = static_cast<uint32_t>(semantic_ids[i]);
        std::memcpy(ptr + 12, &sid, 4);
    }
    pub_semantic_->publish(std::move(msg));
}

void CombinedSemanticPCL::publishColoredPCL(
    const std::vector<std::array<float, 3>> & points,
    const std::vector<int32_t> & semantic_ids,
    const std_msgs::msg::Header & header)
{
    auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
    msg->header = header; 

    msg->fields.resize(4);
    msg->fields[0].name = "x"; msg->fields[0].offset = 0; msg->fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32; msg->fields[0].count = 1;
    msg->fields[1].name = "y"; msg->fields[1].offset = 4; msg->fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32; msg->fields[1].count = 1;
    msg->fields[2].name = "z"; msg->fields[2].offset = 8; msg->fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32; msg->fields[2].count = 1;
    msg->fields[3].name = "rgb"; msg->fields[3].offset = 12; msg->fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32; msg->fields[3].count = 1;

    msg->point_step = 16;
    msg->height = 1;
    msg->width = static_cast<uint32_t>(points.size());
    msg->row_step = msg->point_step * msg->width;
    msg->is_dense = true;
    msg->is_bigendian = false;
    msg->data.resize(msg->row_step);

    uint8_t * data_ptr = msg->data.data();
    for (size_t i = 0; i < points.size(); ++i) 
    {
        uint8_t * ptr = data_ptr + (i * msg->point_step);
        std::memcpy(ptr + 0, &points[i][0], 4);
        std::memcpy(ptr + 4, &points[i][1], 4);
        std::memcpy(ptr + 8, &points[i][2], 4);
        
        auto [r, g, b] = getColorForId(semantic_ids[i]);
        
        uint32_t rgb = (static_cast<uint32_t>(255) << 24) | 
                       (static_cast<uint32_t>(r) << 16) | 
                       (static_cast<uint32_t>(g) << 8) | 
                       (static_cast<uint32_t>(b));
        float rgb_float;
        std::memcpy(&rgb_float, &rgb, sizeof(float));
        
        std::memcpy(ptr + 12, &rgb_float, 4);
    }
    pub_colored_->publish(std::move(msg));
}

} // namespace semantic_pcl

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<semantic_pcl::CombinedSemanticPCL>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}