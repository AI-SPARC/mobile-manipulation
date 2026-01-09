#include "manipulation/ScanObject.hpp"

#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <map>

using namespace std::chrono_literals;

namespace manipulation {

ScanObject::ScanObject(const rclcpp::NodeOptions & options)
 : Node("scan_object_visualizer", options),
   robot_position_(0.0, 0.0, 0.0),
   robot_pos_received_(false),
   current_anim_index_(0),
   is_animating_(false)
{
    // Declaração de Parâmetros
    this->declare_parameter<std::string>("target_frame", "world");
    this->declare_parameter<std::string>("odom_topic", "/odom");
    this->declare_parameter<double>("ray_length", 0.20);
    this->declare_parameter<double>("grid_resolution", 0.04); 
    this->declare_parameter<double>("voxel_map_resolution", 0.02); 
    this->declare_parameter<double>("ray_step_size", 0.01);       
    this->declare_parameter<std::string>("target_object_id", "redbox_09"); 
    
    // Novo parâmetro para ativar/desativar markers
    this->declare_parameter<bool>("publish_markers", false); 

    target_frame_ = this->get_parameter("target_frame").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    ray_length_ = this->get_parameter("ray_length").as_double();
    grid_resolution_ = this->get_parameter("grid_resolution").as_double();
    voxel_map_resolution_ = this->get_parameter("voxel_map_resolution").as_double();
    ray_step_size_ = this->get_parameter("ray_step_size").as_double();
    target_object_id_ = this->get_parameter("target_object_id").as_string();
    publish_markers_ = this->get_parameter("publish_markers").as_bool();

    // Verificação de Segurança
    if (ray_step_size_ > voxel_map_resolution_) 
    {
        RCLCPP_WARN(this->get_logger(), 
            "PERIGO: ray_step_size (%.3f) > voxel_map_resolution (%.3f). O raio pode pular obstáculos!",
            ray_step_size_, voxel_map_resolution_);
    }

    sub_detections_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
        "/bbox_3d_with_labels", 10,
        std::bind(&ScanObject::detectionCallback, this, std::placeholders::_1));

    sub_semantic_pcl_ = this->create_subscription<mobile_manipulation_interfaces::msg::SemanticPcl>(
        "/semantic_pcl_array", 10, 
        std::bind(&ScanObject::semanticPclCallback, this, std::placeholders::_1));

    sub_odometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 10,
        std::bind(&ScanObject::odometryCallback, this, std::placeholders::_1));

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/visualization_marker_array", 10);

    // Timer de Animação
    animation_timer_ = this->create_wall_timer(
        200ms, std::bind(&ScanObject::animationTimerCallback, this));

    RCLCPP_INFO(this->get_logger(), "ScanObject Visualizer Iniciado.");
    RCLCPP_INFO(this->get_logger(), "  Publish Markers: %s", publish_markers_ ? "TRUE" : "FALSE");
}

// ==================== CALLBACKS DE DADOS ====================

void ScanObject::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(robot_pos_mutex_);
    robot_position_ = tf2::Vector3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    robot_pos_received_ = true;
}

void ScanObject::semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(voxel_mutex_);
    voxel_grid_.clear();
    
    if (msg->labels.size() != msg->clouds.size()) return;

    size_t total_points = 0;
    for (size_t i = 0; i < msg->labels.size(); ++i) {
        std::string label = msg->labels[i];
        const auto& cloud = msg->clouds[i];
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x"), iter_y(cloud, "y"), iter_z(cloud, "z");
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            VoxelKey key;
            key.x = static_cast<int>(std::floor(*iter_x / voxel_map_resolution_));
            key.y = static_cast<int>(std::floor(*iter_y / voxel_map_resolution_));
            key.z = static_cast<int>(std::floor(*iter_z / voxel_map_resolution_));
            voxel_grid_[key] = label;
            total_points++;
        }
    }
}

void ScanObject::detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg) {
    last_header_.stamp = msg->header.stamp;
    last_header_.frame_id = target_frame_;

    // 1. Processamento Geométrico
    {
        std::lock_guard<std::mutex> lock(objects_mutex_);
        detected_objects_.clear();
        for (const auto& detection : msg->detections) {
            if (detection.results.empty()) continue;
            std::string label = detection.results[0].hypothesis.class_id;
            
            ObjectData obj_data;
            obj_data.label = label;
            obj_data.detection = detection;
            obj_data.header = last_header_;
            
            if (target_object_id_.empty() || target_object_id_ == label) {
                // Aqui chamamos o cálculo que usa o Voxel Grid
                obj_data.valid_scan_grid = computeValidScanningGrid(
                    detection.bbox.center, detection.bbox.size, label);
            }
            detected_objects_[label] = obj_data;
        }
    }

    // 2. Preparação da Animação (Snapshot)
    // Se publish_markers_ for false, nem precisamos preparar o buffer de animação
    if (!publish_markers_) return;

    std::lock_guard<std::mutex> anim_lock(anim_mutex_);
    
    if (!is_animating_) {
        std::string label_to_viz = target_object_id_;
        
        // Se alvo não definido, pega o primeiro
        if (label_to_viz.empty() && !msg->detections.empty()) 
            label_to_viz = msg->detections[0].results[0].hypothesis.class_id;

        if (!label_to_viz.empty()) {
            {
                std::lock_guard<std::mutex> v_lock(voxel_mutex_);
                if (voxel_grid_.empty()) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                        "Voxel Grid vazio! Aguardando SemanticPcl...");
                    return;
                }
            }

            // --- Pega Poses + Pontos Ordenados ---
            auto result_pair = getSortedScanPoses(label_to_viz);
            auto& cam_poses = result_pair.first;
            auto& surf_points = result_pair.second;

            if (!cam_poses.empty()) {
                poses_to_animate_.clear();
                poses_to_animate_.reserve(cam_poses.size());
                
                for(size_t i=0; i<cam_poses.size(); i++) {
                    poses_to_animate_.push_back({cam_poses[i], surf_points[i]});
                }

                current_anim_index_ = 0;
                
                std::lock_guard<std::mutex> obj_lock(objects_mutex_);
                if(detected_objects_.count(label_to_viz)) {
                    current_anim_bbox_ = detected_objects_[label_to_viz].detection;
                }
                
                is_animating_ = true;
                RCLCPP_INFO(this->get_logger(), "Iniciando animação: %zu poses geradas.", poses_to_animate_.size());
            } 
        }
    }
}

// ==================== VISUALIZAÇÃO (TIMER 200ms) ====================

void ScanObject::animationTimerCallback() {
    // Se o usuário desativou os markers, retorna imediatamente
    if (!publish_markers_) return;

    std::lock_guard<std::mutex> lock(anim_mutex_);

    if (poses_to_animate_.empty()) {
        is_animating_ = false; return;
    }

    if (current_anim_index_ < poses_to_animate_.size()) {
        current_anim_index_++;
    } else {
        is_animating_ = false;
    }

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker del; 
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(del);

    // 1. BBox
    visualization_msgs::msg::Marker bbox;
    bbox.header = last_header_; bbox.header.stamp = this->now();
    bbox.ns = "anim_bbox"; bbox.id = 0; bbox.type = visualization_msgs::msg::Marker::CUBE;
    bbox.action = visualization_msgs::msg::Marker::ADD;
    bbox.pose = current_anim_bbox_.bbox.center; bbox.scale = current_anim_bbox_.bbox.size;
    bbox.color.r = 1.0; bbox.color.g = 1.0; bbox.color.b = 1.0; bbox.color.a = 0.1;
    markers.markers.push_back(bbox);

    // 2. Pontos Câmera (Posição) - Ciano
    visualization_msgs::msg::Marker cam_pts;
    cam_pts.header = bbox.header; cam_pts.ns = "pts_camera"; cam_pts.id = 1;
    cam_pts.type = visualization_msgs::msg::Marker::SPHERE_LIST; cam_pts.action = visualization_msgs::msg::Marker::ADD;
    cam_pts.scale.x = 0.02; cam_pts.scale.y = 0.02; cam_pts.scale.z = 0.02;
    cam_pts.color.r = 0.0; cam_pts.color.g = 1.0; cam_pts.color.b = 1.0; cam_pts.color.a = 0.8;

    // 3. Pontos Superfície - Laranja
    visualization_msgs::msg::Marker surf_pts;
    surf_pts.header = bbox.header; surf_pts.ns = "pts_surface"; surf_pts.id = 2;
    surf_pts.type = visualization_msgs::msg::Marker::SPHERE_LIST; surf_pts.action = visualization_msgs::msg::Marker::ADD;
    surf_pts.scale.x = 0.015; surf_pts.scale.y = 0.015; surf_pts.scale.z = 0.015;
    surf_pts.color.r = 1.0; surf_pts.color.g = 0.6; surf_pts.color.b = 0.0; surf_pts.color.a = 0.9;

    // 4. Linhas (Raios)
    visualization_msgs::msg::Marker lines;
    lines.header = bbox.header; lines.ns = "rays_vis"; lines.id = 3;
    lines.type = visualization_msgs::msg::Marker::LINE_LIST; lines.action = visualization_msgs::msg::Marker::ADD;
    lines.scale.x = 0.002; lines.color.r = 1.0; lines.color.g = 1.0; lines.color.b = 1.0; lines.color.a = 0.3;

    // Acumula histórico
    for (size_t i = 0; i < current_anim_index_; i++) {
        cam_pts.points.push_back(poses_to_animate_[i].first.position);
        surf_pts.points.push_back(poses_to_animate_[i].second);
        
        lines.points.push_back(poses_to_animate_[i].first.position);
        lines.points.push_back(poses_to_animate_[i].second);
    }
    markers.markers.push_back(cam_pts);
    markers.markers.push_back(surf_pts);
    markers.markers.push_back(lines);

    // 5. Destaque Atual com SETA (Pose)
    if (current_anim_index_ > 0 && current_anim_index_ <= poses_to_animate_.size()) {
        visualization_msgs::msg::Marker arrow;
        arrow.header = bbox.header; arrow.ns = "anim_pose_arrow"; arrow.id = 4;
        arrow.type = visualization_msgs::msg::Marker::ARROW; 
        arrow.action = visualization_msgs::msg::Marker::ADD;
        
        // Usa a Pose calculada (Posição + Orientação)
        arrow.pose = poses_to_animate_[current_anim_index_ - 1].first;
        
        arrow.scale.x = 0.10; arrow.scale.y = 0.01; arrow.scale.z = 0.01; 
        arrow.color.r = 1.0; arrow.color.g = 0.0; arrow.color.b = 0.0; arrow.color.a = 1.0; 
        markers.markers.push_back(arrow);
    }

    marker_pub_->publish(markers);
}

// ==================== GEOMETRIA E CÁLCULO DE ORIENTAÇÃO ====================

geometry_msgs::msg::Quaternion ScanObject::computeLookAtOrientation(
    const tf2::Vector3& camera_pos, const tf2::Vector3& target_pos)
{
    // Vetor Forward: Câmera -> Alvo
    tf2::Vector3 forward = (target_pos - camera_pos).normalize();
    tf2::Vector3 global_up(0, 0, 1);
    
    tf2::Vector3 right = forward.cross(global_up).normalize();
    if (std::abs(forward.z()) > 0.999) { 
        right = forward.cross(tf2::Vector3(0,1,0)).normalize();
    }
    
    tf2::Vector3 up = right.cross(forward).normalize();

    // Matriz de Rotação (X = Frente)
    tf2::Matrix3x3 rot_mat(
        forward.x(), -right.x(), up.x(),
        forward.y(), -right.y(), up.y(),
        forward.z(), -right.z(), up.z()
    );

    tf2::Quaternion q;
    rot_mat.getRotation(q);
    return tf2::toMsg(q);
}

// ==================== RAY MARCHING ====================

VoxelKey ScanObject::pointToVoxel(const tf2::Vector3& pt) {
    VoxelKey key;
    key.x = static_cast<int>(std::floor(pt.x() / voxel_map_resolution_));
    key.y = static_cast<int>(std::floor(pt.y() / voxel_map_resolution_));
    key.z = static_cast<int>(std::floor(pt.z() / voxel_map_resolution_));
    return key;
}

std::pair<ScanObject::RayResult, tf2::Vector3> ScanObject::analyzeRay(
    const tf2::Vector3& start, const tf2::Vector3& end, const std::string& target_label)
{
    std::lock_guard<std::mutex> lock(voxel_mutex_);
    if (voxel_grid_.empty()) return {RayResult::MISS, tf2::Vector3(0,0,0)};

    tf2::Vector3 dir = end - start;
    double len = dir.length();
    if (len < 1e-4) return {RayResult::MISS, tf2::Vector3(0,0,0)};
    dir.normalize();

    for (double d = 0.0; d < len; d += ray_step_size_) {
        tf2::Vector3 current_pt = start + (dir * d);
        auto it = voxel_grid_.find(pointToVoxel(current_pt));
        
        if (it != voxel_grid_.end()) {
            if (it->second == target_label) {
                return {RayResult::HIT_TARGET, current_pt};
            } else {
                return {RayResult::BLOCKED, tf2::Vector3(0,0,0)};
            }
        }
    }
    return {RayResult::MISS, tf2::Vector3(0,0,0)};
}

std::vector<ScanPoint> ScanObject::computeValidScanningGrid(const geometry_msgs::msg::Pose& p, const geometry_msgs::msg::Vector3& s, const std::string& l) {
    std::vector<ScanPoint> pts;
    tf2::Vector3 c(p.position.x, p.position.y, p.position.z);
    tf2::Quaternion q(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
    tf2::Matrix3x3 m(q);
    double dx = s.x/2.0 + ray_length_, dy = s.y/2.0 + ray_length_, dz = s.z/2.0 + ray_length_;
    
    struct F { tf2::Vector3 o, u, v; double ul, vl; int id; };
    std::vector<F> fs = {
        {{dx,-dy,-dz},{0,1,0},{0,0,1},2*dy,2*dz, 0}, {{-dx,-dy,-dz},{0,1,0},{0,0,1},2*dy,2*dz, 1},
        {{-dx,dy,-dz},{1,0,0},{0,0,1},2*dx,2*dz, 2}, {{-dx,-dy,-dz},{1,0,0},{0,0,1},2*dx,2*dz, 3},
        {{-dx,-dy,dz},{1,0,0},{0,1,0},2*dx,2*dy, 4}
    };

    for(auto& f : fs) 
    {
        for(double u=0; u<=f.ul+1e-4; u+=grid_resolution_) 
        {
            for(double v=0; v<=f.vl+1e-4; v+=grid_resolution_) 
            {
                tf2::Vector3 pw = c + m * (f.o + f.u*u + f.v*v);
                auto res = analyzeRay(pw, c, l);
                if (res.first == RayResult::HIT_TARGET) {
                    ScanPoint sp;
                    sp.position = pw;
                    sp.target_center = c; 
                    sp.surface_contact = res.second;
                    sp.face_id = f.id;
                    pts.push_back(sp);
                }
            }
        }
    }
    return pts;
}

// ==================== ORDENAÇÃO E GERAÇÃO DE POSES ====================

std::pair<std::vector<geometry_msgs::msg::Pose>, std::vector<geometry_msgs::msg::Point>> 
ScanObject::getSortedScanPoses(const std::string& label)
{
    tf2::Vector3 robot_pos;
    {
        std::lock_guard<std::mutex> lock(robot_pos_mutex_);
        robot_pos = robot_pos_received_ ? robot_position_ : tf2::Vector3(0,0,0);
    }

    std::lock_guard<std::mutex> lock(objects_mutex_);
    auto it = detected_objects_.find(label);
    if (it == detected_objects_.end()) return {{},{}};

    const auto& obj_data = it->second;
    const auto& points = obj_data.valid_scan_grid;
    double center_z = obj_data.detection.bbox.center.position.z;

    // 1. Agrupar por Face
    std::map<int, std::vector<ScanPoint>> face_map;
    for(const auto& p : points) face_map[p.face_id].push_back(p);

    // 2. Ordenar Faces
    struct FaceInfo { int id; double dist; };
    std::vector<FaceInfo> faces_by_dist;
    for (const auto& [fid, pts] : face_map) {
        tf2::Vector3 cent(0,0,0);
        for(const auto& p : pts) cent += p.position;
        if (!pts.empty()) cent /= (double)pts.size();
        faces_by_dist.push_back({fid, (cent - robot_pos).length()});
    }
    std::sort(faces_by_dist.begin(), faces_by_dist.end(), [](auto& a, auto& b){ return a.dist < b.dist; });

    int hidden_id = faces_by_dist.empty() ? -1 : faces_by_dist.back().id;

    struct Group { std::vector<ScanPoint> pts; };
    Group vis_above, vis_below, hidden;

    for (const auto& f : faces_by_dist) {
        int fid = f.id;
        auto& pts = face_map[fid];
        std::sort(pts.begin(), pts.end(), [&](const ScanPoint& a, const ScanPoint& b){
            return (a.position - robot_pos).length2() < (b.position - robot_pos).length2();
        });

        if (fid == hidden_id && faces_by_dist.size() > 1) {
            hidden.pts.insert(hidden.pts.end(), pts.begin(), pts.end());
        } else {
            for(const auto& p : pts) {
                if(p.position.z() > center_z) vis_above.pts.push_back(p);
                else vis_below.pts.push_back(p);
            }
        }
    }

    std::vector<geometry_msgs::msg::Pose> poses_vec;
    std::vector<geometry_msgs::msg::Point> surf_vec;
    
    auto add = [&](const std::vector<ScanPoint>& src) 
    {
        for(const auto& p : src) {
            geometry_msgs::msg::Pose pose;
            pose.position.x = p.position.x();
            pose.position.y = p.position.y();
            pose.position.z = p.position.z();
            
            // LookAt: Orientação apontando para o objeto
            pose.orientation = computeLookAtOrientation(p.position, p.target_center);

            geometry_msgs::msg::Point s;
            s.x = p.surface_contact.x(); s.y = p.surface_contact.y(); s.z = p.surface_contact.z();
            
            poses_vec.push_back(pose);
            surf_vec.push_back(s);
        }
    };

    add(vis_above.pts);
    add(vis_below.pts);
    add(hidden.pts);

    return {poses_vec, surf_vec};
}

} // namespace manipulation

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<manipulation::ScanObject>());
    rclcpp::shutdown();
    return 0;
}