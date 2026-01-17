#include "vision/CombinedSemanticPCL.hpp"

#include <rclcpp/qos.hpp>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <random>
#include <cctype>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>

namespace semantic_pcl
{

CombinedSemanticPCL::CombinedSemanticPCL(const rclcpp::NodeOptions & options)
    : Node("combined_semantic_pcl", options)
{
    // --- Parâmetros ---
    this->declare_parameter<std::string>("target_frame", "world"); 
    this->declare_parameter<int>("downsample_step", 1); 
    this->declare_parameter<int>("num_cameras", 3); 
    this->declare_parameter<double>("publish_rate", 5.0); // 5Hz é suficiente para PCL pesada
    this->declare_parameter<double>("data_timeout", 1.0); // 1 segundo sem dado = descarta câmera

    target_frame_ = this->get_parameter("target_frame").as_string();
    
    downsample_step_ = this->get_parameter("downsample_step").as_int();
    if (downsample_step_ < 1) downsample_step_ = 1;

    num_cameras_ = this->get_parameter("num_cameras").as_int();
    if (num_cameras_ < 1) num_cameras_ = 1;

    publish_rate_ = this->get_parameter("publish_rate").as_double();
    data_timeout_ = this->get_parameter("data_timeout").as_double();

    RCLCPP_INFO(this->get_logger(), "=== SEMANTIC PCL NODE (ASYNC FUSION) ===");
    RCLCPP_INFO(this->get_logger(), "Target Frame: %s", target_frame_.c_str());
    RCLCPP_INFO(this->get_logger(), "Cameras: %d | Pub Rate: %.1f Hz", num_cameras_, publish_rate_);

    // --- TF Init ---
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Inicializa vetor de dados
    latest_frames_.resize(num_cameras_);

    // QoS padrão
    auto sensor_qos = rclcpp::SensorDataQoS(); 
    auto rmw_qos = sensor_qos.get_rmw_qos_profile();

    // --- Inicialização Dinâmica das Câmeras ---
    for (int i = 0; i < num_cameras_; ++i)
    {
        std::string suffix = "_" + std::to_string(i);
        
        std::string seg_topic = "semantic_segmentation" + suffix;
        std::string pcl_topic = "depth_pcl" + suffix;
        std::string lbl_topic = "semantic_labels" + suffix;

        auto cam_module = std::make_shared<CameraModules>();

        // 1. Labels Individual
        cam_module->labels_sub = this->create_subscription<std_msgs::msg::String>(
            lbl_topic, 10,
            std::bind(&CombinedSemanticPCL::labelsCallback, this, std::placeholders::_1)
        );

        // 2. Subscribers para Sincronização Local (Img + PCL da mesma cam)
        cam_module->seg_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, seg_topic, rmw_qos);
        
        cam_module->pcl_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
            this, pcl_topic, rmw_qos);

        // Usamos uma fila maior (20) para tolerar jitter interno da câmera
        cam_module->sync = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(20), *cam_module->seg_sub, *cam_module->pcl_sub);

        cam_module->sync->registerCallback(
            std::bind(&CombinedSemanticPCL::syncedCallback, this, std::placeholders::_1, std::placeholders::_2, i)
        );

        cameras_.push_back(cam_module);
        RCLCPP_INFO(this->get_logger(), "Camera %d inscrita: %s", i, pcl_topic.c_str());
    }
    
    // --- Publishers Unificados ---
    pub_semantic_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("semantic_pcl", 10);
    pub_colored_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("semantic_pcl_colored", 10);
    pub_custom_msg_ = this->create_publisher<mobile_manipulation_interfaces::msg::SemanticPcl>("semantic_pcl_array", 10);

    // --- Timer de Publicação ---
    // Publica periodicamente independente da chegada dos dados

    int ms_period = static_cast<int>(1000.0 / publish_rate_);
    pub_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(ms_period),
        std::bind(&CombinedSemanticPCL::publishEvent, this)
    );
}

void CombinedSemanticPCL::labelsCallback(const std_msgs::msg::String::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(data_mutex_);
    parseLabelsJson(msg->data);
}


void CombinedSemanticPCL::syncedCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & seg_msg,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & pcl_msg,
    int camera_index)
{
    // Apenas salva o dado mais recente (Latching strategy)
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (camera_index >= 0 && camera_index < static_cast<int>(latest_frames_.size()))
    {
        latest_frames_[camera_index].seg_msg = seg_msg;
        latest_frames_[camera_index].pcl_msg = pcl_msg;
        latest_frames_[camera_index].receive_time = this->get_clock()->now();
        latest_frames_[camera_index].has_data = true;
    }
}

void CombinedSemanticPCL::publishEvent()
{
    std::vector<std::array<float, 3>> merged_points;
    std::vector<int32_t> merged_ids;
    rclcpp::Time output_stamp = this->get_clock()->now(); // O timestamp da saída é o "agora"

    // Bloqueia apenas para copiar ou processar
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        processAndMerge(merged_points, merged_ids, output_stamp);
    }

    if (!merged_points.empty())
    {
        std_msgs::msg::Header header;
        header.frame_id = target_frame_;
        header.stamp = output_stamp;

        publishSemanticPCL(merged_points, merged_ids, header);
        publishColoredPCL(merged_points, merged_ids, header);
        publishSplitSemanticPCL(merged_points, merged_ids, header);
    }
}

void CombinedSemanticPCL::processAndMerge(
    std::vector<std::array<float, 3>>& out_points,
    std::vector<int32_t>& out_ids,
    rclcpp::Time& out_stamp)
{
    
    for (int i = 0; i < num_cameras_; ++i)
    {
        const auto& frame = latest_frames_[i];

        // 1. Verifica se existe dado
        if (!frame.has_data) continue;

        // 2. Verifica se o dado é muito velho (Câmera morreu?)
        double age = (out_stamp - frame.receive_time).seconds();
        if (age > data_timeout_) 
        {
            continue; 
        }

        auto& pcl = frame.pcl_msg;
        auto& img = frame.seg_msg;

        // 3. TF Lookup (Frame da nuvem -> Frame Alvo)
        geometry_msgs::msg::TransformStamped t_stamped;
        try {
            // Tenta pegar a TF exata do momento que a nuvem foi capturada
            t_stamped = tf_buffer_->lookupTransform(
                target_frame_, pcl->header.frame_id, pcl->header.stamp,
                rclcpp::Duration::from_seconds(0.1));
        } catch (const tf2::TransformException & ex) 
        {
            
            try {
                t_stamped = tf_buffer_->lookupTransform(
                    target_frame_, pcl->header.frame_id, tf2::TimePointZero);
            } catch (...) { continue; } 
        }

        tf2::Transform transform;
        tf2::fromMsg(t_stamped.transform, transform);

      
        uint32_t width = pcl->width;
        uint32_t height = pcl->height;
        size_t num_points = width * height;
        const int32_t* seg_data = reinterpret_cast<const int32_t*>(img->data.data());
        const uint8_t* pcl_data = pcl->data.data();
        uint32_t point_step = pcl->point_step;

        int x_off = -1, y_off = -1, z_off = -1;
        for (const auto & f : pcl->fields) {
            if (f.name == "x") x_off = f.offset;
            else if (f.name == "y") y_off = f.offset;
            else if (f.name == "z") z_off = f.offset;
        }
        if (x_off < 0) continue;

        size_t total_pixels = img->width * img->height;

        for (size_t k = 0; k < num_points; k += downsample_step_)
        {
            const uint8_t * ptr = pcl_data + (k * point_step);
            float x_cam, y_cam, z_cam;
            std::memcpy(&x_cam, ptr + x_off, sizeof(float));
            std::memcpy(&y_cam, ptr + y_off, sizeof(float));
            std::memcpy(&z_cam, ptr + z_off, sizeof(float));

            if (!std::isfinite(x_cam) || !std::isfinite(y_cam) || !std::isfinite(z_cam)) continue;
            
            if (z_cam < 0.02f) continue; 

            
            tf2::Vector3 pt_cam(x_cam, y_cam, z_cam);
            tf2::Vector3 pt_world = transform * pt_cam;

            
            int32_t sem_id = 0;
            if (num_points == total_pixels) 
            {
                sem_id = seg_data[k];
            } 
            else 
            {
                size_t idx = static_cast<size_t>((double)k * total_pixels / num_points);
                if (idx >= total_pixels) idx = total_pixels - 1;
                sem_id = seg_data[idx];
            }

            out_points.push_back({(float)pt_world.x(), (float)pt_world.y(), (float)pt_world.z()});
            out_ids.push_back(sem_id);
        }
    }
}


std::string CombinedSemanticPCL::extractCleanLabel(std::string raw_label)
{
    raw_label.erase(std::remove(raw_label.begin(), raw_label.end(), '\"'), raw_label.end());
    size_t colon_pos = raw_label.find(':');
    if (colon_pos != std::string::npos && colon_pos < raw_label.size() - 1) {
        return raw_label.substr(colon_pos + 1);
    }
    return raw_label;
}

void CombinedSemanticPCL::parseLabelsJson(const std::string & json_str)
{
    if (json_str.empty()) return;
    std::regex object_regex("\"([0-9]+)\"\\s*:\\s*\\{[^}]*\"([A-Za-z0-9_]+)\"\\s*:\\s*\"([A-Za-z0-9_:.\\-]+)\"");
    std::regex string_regex("\"([0-9]+)\"\\s*:\\s*\"([A-Za-z0-9_:.\\-]+)\"");

    std::smatch match;
    std::string::const_iterator search_start(json_str.cbegin());

    while (std::regex_search(search_start, json_str.cend(), match, object_regex)) {
        try {
            int32_t id = std::stoi(match[1].str());
            std::string val = match[3].str();
            id_to_label_[id] = extractCleanLabel(val);
        } catch (...) {}
        search_start = match.suffix().first;
    }

    search_start = json_str.cbegin();
    while (std::regex_search(search_start, json_str.cend(), match, string_regex)) {
        try {
            int32_t id = std::stoi(match[1].str());
            if (id_to_label_.find(id) == id_to_label_.end()) {
                std::string val = match[2].str();
                id_to_label_[id] = extractCleanLabel(val);
            }
        } catch (...) {}
        search_start = match.suffix().first;
    }
}

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
    
    if (!points.empty()) {
        uint8_t* ptr = msg.data.data();
        for(const auto& p : points) {
            std::memcpy(ptr, &p[0], 12);
            ptr += 12;
        }
    }
    return msg;
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
        if (pts.size() < 10) continue; // Ignora ruído pequeno

        std::string label_str;
        if (id_to_label_.count(id)) label_str = id_to_label_.at(id);
        else continue;

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