#include "slam_core/Mapping.hpp"
#include <chrono>
#include "rclcpp_components/register_node_macro.hpp"
#include <pcl/common/transforms.h>

namespace slam_core 
{

Mapping::Mapping(const rclcpp::NodeOptions & options)
: Node("mapping_node", options)
{
    this->declare_parameter<float>("voxel_leaf_size", 0.01f); 
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();

    this->declare_parameter<double>("map_publish_rate", 1.0);
    map_publish_rate_ = this->get_parameter("map_publish_rate").as_double();


    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    std::string map_topic = "/slam/global_map";
    global_map_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, 1);
    
    double timer_period_sec = 1.0 / map_publish_rate_;
    map_publish_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(timer_period_sec),
        std::bind(&Mapping::publish_map_callback, this));

    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    geometry_msgs::msg::TransformStamped static_transformStamped;
    static_transformStamped.header.stamp = this->now();
    
  
    static_transformStamped.header.frame_id = "/map";
    static_transformStamped.child_frame_id = "/odom"; 
    
    static_transformStamped.transform.translation.x = 0.0;
    static_transformStamped.transform.translation.y = 0.0;
    static_transformStamped.transform.translation.z = 0.0;
    static_transformStamped.transform.rotation.x = 0.0;
    static_transformStamped.transform.rotation.y = 0.0;
    static_transformStamped.transform.rotation.z = 0.0;
    static_transformStamped.transform.rotation.w = 1.0;
    
    static_tf_broadcaster_->sendTransform(static_transformStamped);
    voxel_occupancy_set_.reserve(2000000);

    RCLCPP_INFO(this->get_logger(), "Mapping Node criado. Publicacao assincrona a %.1f Hz.", map_publish_rate_);
}

Mapping::~Mapping() {}

void Mapping::set_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cam_info, 
                              const std::string& main_frame_id, 
                              float depth_scale)
{
    if (camera_initialized_) return;
    
    fx_ = cam_info->k[0]; cx_ = cam_info->k[2];
    fy_ = cam_info->k[4]; cy_ = cam_info->k[5];
    depth_scale_ = depth_scale;
    main_frame_id_ = main_frame_id;
    
    RCLCPP_INFO(this->get_logger(), "Mapping: Camera Info Inicializada. Main Frame vinculado a: '%s'", main_frame_id_.c_str());
    camera_initialized_ = true;
}

void Mapping::add_keyframe_data(int kf_id, const cv::Mat& rgb_img, const cv::Mat& depth_img, 
                                const std::string& rgb_frame, const std::string& depth_frame)
{
    if (!camera_initialized_) return;
    KeyframeData data;
    data.rgb = rgb_img.clone();
    data.depth = depth_img.clone();
    
    // Passa os frames detectados em tempo real para o gerador de nuvens
    data.local_cloud = generate_local_cloud(data.rgb, data.depth, rgb_frame, depth_frame);
    {
        std::lock_guard<std::mutex> lock(map_mutex_);

        keyframe_database_[kf_id] = data;

    }
    
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Mapping::generate_local_cloud(
    const cv::Mat& rgb, const cv::Mat& depth, const std::string& rgb_frame, const std::string& depth_frame)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_camera(new pcl::PointCloud<pcl::PointXYZRGB>());

    // 1. TF do Base Link para a Câmera de Profundidade (Onde a geometria 3D nasce)
    Eigen::Matrix4f T_main_depth = Eigen::Matrix4f::Identity();
    if (main_frame_id_ != depth_frame) 
    {
        try {
            geometry_msgs::msg::TransformStamped tf_stamped = tf_buffer_->lookupTransform(
                main_frame_id_, depth_frame, tf2::TimePointZero);

            Eigen::Quaternionf q(tf_stamped.transform.rotation.w, tf_stamped.transform.rotation.x,
                                 tf_stamped.transform.rotation.y, tf_stamped.transform.rotation.z);
            Eigen::Vector3f t(tf_stamped.transform.translation.x, tf_stamped.transform.translation.y,
                              tf_stamped.transform.translation.z);

            T_main_depth.block<3,3>(0,0) = q.matrix();
            T_main_depth.block<3,1>(0,3) = t;
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                 "Mapeamento: Aguardando TF entre %s e %s...", 
                                 main_frame_id_.c_str(), depth_frame.c_str());
            return cloud_camera; // Falhou, retorna nuvem vazia e tenta na próxima
        }
    }

    // 2. TF da Câmera de Profundidade para a Câmera RGB (Ajuste Automático de Parallax)
    Eigen::Matrix4f T_rgb_depth = Eigen::Matrix4f::Identity();
    bool has_rgb_offset = false;
    
    if (rgb_frame != depth_frame) 
    {
        try {
            geometry_msgs::msg::TransformStamped tf_color = tf_buffer_->lookupTransform(
                rgb_frame, depth_frame, tf2::TimePointZero);
            
            Eigen::Quaternionf q(tf_color.transform.rotation.w, tf_color.transform.rotation.x,
                                 tf_color.transform.rotation.y, tf_color.transform.rotation.z);
            Eigen::Vector3f t(tf_color.transform.translation.x, tf_color.transform.translation.y,
                              tf_color.transform.translation.z);
            
            T_rgb_depth.block<3,3>(0,0) = q.matrix();
            T_rgb_depth.block<3,1>(0,3) = t;

            // Se houver qualquer deslocamento físico entre as câmeras, ativamos o Offset
            if (t.norm() > 0.001 || std::abs(q.w() - 1.0) > 0.001) {
                has_rgb_offset = true;
            }
        } catch (const tf2::TransformException & ex) {
            // Se falhar, assumimos que as imagens já estão perfeitamente sobrepostas (Aligned)
        }
    }

    // 3. Geração da Nuvem
    for (int v = 0; v < depth.rows; v += 2) {
        for (int u = 0; u < depth.cols; u += 2) {
            float z = 0.0f;
            if (depth.type() == CV_32FC1) z = depth.at<float>(v, u);
            else if (depth.type() == CV_16UC1) z = depth.at<uint16_t>(v, u) / depth_scale_;

            if (z <= 0.1f || z > 40.0f) 
            {
                continue; 
            }
            // Ponto 3D no Frame da Profundidade
            float x_d = (u - cx_) * z / fx_;
            float y_d = (v - cy_) * z / fy_;
            float z_d = z;

            int u_rgb = u;
            int v_rgb = v;

            // Se o RGB e o Depth estiverem em lugares diferentes, projeta o 3D no Frame RGB para pegar a cor certa
            if (has_rgb_offset) 
            {
                Eigen::Vector4f p_d(x_d, y_d, z_d, 1.0f);
                Eigen::Vector4f p_rgb = T_rgb_depth * p_d;
                
                if (p_rgb.z() > 0) {
                    u_rgb = std::round((p_rgb.x() * fx_ / p_rgb.z()) + cx_);
                    v_rgb = std::round((p_rgb.y() * fy_ / p_rgb.z()) + cy_);
                } 
                else 
                {
                    continue; // Ponto está atrás da câmera RGB
                }
            }

            // Atribui a cor se o pixel calculado estiver dentro da imagem RGB
            if (u_rgb >= 0 && u_rgb < rgb.cols && v_rgb >= 0 && v_rgb < rgb.rows) {
                pcl::PointXYZRGB pt;
                pt.x = x_d; pt.y = y_d; pt.z = z_d;
                
                cv::Vec3b color = rgb.at<cv::Vec3b>(v_rgb, u_rgb);
                pt.r = color[2]; pt.g = color[1]; pt.b = color[0];
                
                cloud_camera->points.push_back(pt);
            }
        }
    }
    
    // Transforma toda a nuvem do Depth Frame para o Main Frame (Base Link)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_main_frame(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*cloud_camera, *cloud_main_frame, T_main_depth);

    // Downsample com Voxel Grid
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_local_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(cloud_main_frame);
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_filter.filter(*downsampled_local_cloud);

    return downsampled_local_cloud;
}

void Mapping::update_global_map(const std::vector<std::pair<int, gtsam::Pose3>>& changed_poses)
{
    if (changed_poses.empty()) return; 

    auto start_time = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lock(map_mutex_);

    // Como chegam TODAS as poses, limpamos o mapa atual para não gerar nuvens "fantasmas" (pontos duplicados no passado)
    voxel_occupancy_set_.clear();

    for (const auto& pair : changed_poses) 
    {
        int kf_id = pair.first;
        if (keyframe_database_.find(kf_id) != keyframe_database_.end()) 
        {
            auto& kf_data = keyframe_database_[kf_id]; 
            
            // Só faz a transformação matricial PESADA se for uma pose nova ou se o GTSAM tiver movido ela (Loop Closure)
            if (!kf_data.global_cloud_cache || !kf_data.pose.equals(pair.second, 1e-4)) 
            {
                if (kf_data.local_cloud && !kf_data.local_cloud->empty()) {
                    Eigen::Matrix4f T_map_main = pair.second.matrix().cast<float>();
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZRGB>());
                    
                    // Transforma a nuvem local para a nova pose
                    pcl::transformPointCloud(*(kf_data.local_cloud), *cloud_map, T_map_main);
                    
                    kf_data.pose = pair.second;
                    kf_data.global_cloud_cache = cloud_map;
                }
            }

            // Adiciona os pontos (recalculados ou em cache) diretamente no conjunto de voxels
            if (kf_data.global_cloud_cache) 
            {
                for (const auto& pt : kf_data.global_cloud_cache->points) 
                {
                    VoxelKey key{
                        static_cast<int>(std::floor(pt.x / voxel_leaf_size_)),
                        static_cast<int>(std::floor(pt.y / voxel_leaf_size_)),
                        static_cast<int>(std::floor(pt.z / voxel_leaf_size_)),
                        pt.r, pt.g, pt.b
                    };
                    voxel_occupancy_set_.insert(key);
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end_time - start_time;

    RCLCPP_INFO(this->get_logger(), "[MAPPING] Calculo Interno (Reconstrucao): %.2f ms", ms_double.count());
}




void Mapping::publish_map_callback()
{
    if (global_map_pub_->get_subscription_count() == 0) return;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_to_publish(new pcl::PointCloud<pcl::PointXYZRGB>());

    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (voxel_occupancy_set_.empty()) return;

        cloud_to_publish->points.reserve(voxel_occupancy_set_.size());

        for (const auto& key : voxel_occupancy_set_) 
        {
            pcl::PointXYZRGB pt;
            pt.x = (key.x + 0.5f) * voxel_leaf_size_;
            pt.y = (key.y + 0.5f) * voxel_leaf_size_;
            pt.z = (key.z + 0.5f) * voxel_leaf_size_;
            pt.r = key.r;
            pt.g = key.g;
            pt.b = key.b;

            cloud_to_publish->points.push_back(pt);
        }
    }

    cloud_to_publish->width = cloud_to_publish->points.size();
    cloud_to_publish->height = 1;
    cloud_to_publish->is_dense = false;

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*cloud_to_publish, output_msg);
    output_msg.header.frame_id = "camera_link"; 
    output_msg.header.stamp = this->now();

    global_map_pub_->publish(output_msg);
}

} // namespace slam_core

RCLCPP_COMPONENTS_REGISTER_NODE(slam_core::Mapping)