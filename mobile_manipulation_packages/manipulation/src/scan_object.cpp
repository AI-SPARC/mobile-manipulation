#include "manipulation/ScanObject.hpp"

#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <map>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> // Necessário para conversões de quaternion

using namespace std::chrono_literals;

namespace manipulation {


ScanObject::ScanObject(const rclcpp::NodeOptions & options)
 : Node("scan_object_visualizer", options),
   robot_position_(0.0, 0.0, 0.0),
   robot_pos_received_(false),
   current_anim_index_(0),
   is_animating_(false)
{
    this->declare_parameter<std::string>("target_frame", "world");
    this->declare_parameter<std::string>("odom_topic", "/odom");
    this->declare_parameter<double>("ray_length", 0.20);
    this->declare_parameter<double>("grid_resolution", 0.04); 
    this->declare_parameter<double>("voxel_map_resolution", 0.02); 
    this->declare_parameter<double>("ray_step_size", 0.01);       
    this->declare_parameter<std::string>("target_object_id", "");
    
    // Novo parâmetro solicitado
    this->declare_parameter<bool>("publish_markers", false); 

    target_frame_ = this->get_parameter("target_frame").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    ray_length_ = this->get_parameter("ray_length").as_double();
    grid_resolution_ = this->get_parameter("grid_resolution").as_double();
    voxel_map_resolution_ = this->get_parameter("voxel_map_resolution").as_double();
    ray_step_size_ = this->get_parameter("ray_step_size").as_double();
    target_object_id_ = this->get_parameter("target_object_id").as_string();
    publish_markers_ = this->get_parameter("publish_markers").as_bool();

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

        animation_timer_ = this->create_wall_timer(
            200ms, std::bind(&ScanObject::animationTimerCallback, this));
    
    

    RCLCPP_INFO(this->get_logger(), "ScanObject iniciado. Retornando Poses orientadas ao centro.");
}

// ==================== CALLBACKS ====================

void ScanObject::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(robot_pos_mutex_);
    robot_position_ = tf2::Vector3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    robot_pos_received_ = true;
}

void ScanObject::semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(voxel_mutex_);
    voxel_grid_.clear();
    if (msg->labels.size() != msg->clouds.size()) return;
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
        }
    }
}

void ScanObject::detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg) {
    last_header_.stamp = msg->header.stamp;
    last_header_.frame_id = target_frame_;

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
                obj_data.valid_scan_grid = computeValidScanningGrid(
                    detection.bbox.center, detection.bbox.size, label);
            }
            detected_objects_[label] = obj_data;
        }
    }

    std::lock_guard<std::mutex> anim_lock(anim_mutex_);
    if (!is_animating_) {
        std::string label_to_viz = target_object_id_;
        if (label_to_viz.empty() && !msg->detections.empty()) 
            label_to_viz = msg->detections[0].results[0].hypothesis.class_id;

        if (!label_to_viz.empty()) {
            // Agora recebe um vetor de Poses
            auto sorted_poses = getSortedScanPoses(label_to_viz);
            if (!sorted_poses.empty()) {
                poses_to_animate_ = sorted_poses; // Variável renomeada para clareza, mas deve estar no .hpp
                current_anim_index_ = 0;
                
                std::lock_guard<std::mutex> obj_lock(objects_mutex_);
                if(detected_objects_.count(label_to_viz)) {
                    current_anim_bbox_ = detected_objects_[label_to_viz].detection;
                }
                is_animating_ = true;
            }
        }
    }
}

// ==================== VISUALIZAÇÃO ====================

void ScanObject::animationTimerCallback() {
    // Verifica se deve publicar marcadores
    if (!publish_markers_) return;

    std::lock_guard<std::mutex> lock(anim_mutex_);
    if (poses_to_animate_.empty()) {
        is_animating_ = false; return;
    }

    if (current_anim_index_ < poses_to_animate_.size()) current_anim_index_++;
    else is_animating_ = false;

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker del; del.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(del);

    visualization_msgs::msg::Marker bbox;
    bbox.header = last_header_; bbox.header.stamp = this->now();
    bbox.ns = "anim_bbox"; bbox.id = 0; bbox.type = visualization_msgs::msg::Marker::CUBE;
    bbox.action = visualization_msgs::msg::Marker::ADD;
    bbox.pose = current_anim_bbox_.bbox.center; bbox.scale = current_anim_bbox_.bbox.size;
    bbox.color.r = 1.0; bbox.color.g = 1.0; bbox.color.b = 1.0; bbox.color.a = 0.1;
    markers.markers.push_back(bbox);

    visualization_msgs::msg::Marker hist;
    hist.header = bbox.header; hist.ns = "anim_hist"; hist.id = 1;
    hist.type = visualization_msgs::msg::Marker::SPHERE_LIST; hist.action = visualization_msgs::msg::Marker::ADD;
    hist.scale.x = 0.02; hist.scale.y = 0.02; hist.scale.z = 0.02;
    hist.color.r = 0.0; hist.color.g = 1.0; hist.color.b = 1.0; hist.color.a = 0.8;

    visualization_msgs::msg::Marker line;
    line.header = bbox.header; line.ns = "anim_path"; line.id = 2;
    line.type = visualization_msgs::msg::Marker::LINE_STRIP; line.action = visualization_msgs::msg::Marker::ADD;
    line.scale.x = 0.005; line.color.r = 1.0; line.color.g = 1.0; line.color.b = 1.0; line.color.a = 0.3;

    // Adiciona o histórico de posições
    for (size_t i = 0; i < current_anim_index_; i++) {
        hist.points.push_back(poses_to_animate_[i].position);
        line.points.push_back(poses_to_animate_[i].position);
    }
    markers.markers.push_back(hist); markers.markers.push_back(line);

    // Marcador atual (Cabeça) com a ORIENTAÇÃO correta
    if (current_anim_index_ > 0 && current_anim_index_ <= poses_to_animate_.size()) {
        visualization_msgs::msg::Marker cur;
        cur.header = bbox.header; cur.ns = "anim_head"; cur.id = 3;
        // Usar ARROW ajuda a verificar a orientação, mas mantive SPHERE conforme original.
        // Se quiser ver a orientação, mude type para visualization_msgs::msg::Marker::ARROW
        cur.type = visualization_msgs::msg::Marker::ARROW; 
        cur.action = visualization_msgs::msg::Marker::ADD;
        
        // AQUI ESTÁ A MÁGICA: A Pose completa (posição + orientação calculada)
        cur.pose = poses_to_animate_[current_anim_index_ - 1]; 
        
        cur.scale.x = 0.05; cur.scale.y = 0.01; cur.scale.z = 0.01; // Escala para Seta
        cur.color.r = 1.0; cur.color.g = 0.0; cur.color.b = 0.0; cur.color.a = 1.0; 
        markers.markers.push_back(cur);
    }
    marker_pub_->publish(markers);
}

// ==================== LÓGICA DE ORDENAÇÃO E CÁLCULO DE POSE ====================

// Função auxiliar para calcular orientação olhando para o alvo
geometry_msgs::msg::Pose createPoseLookingAt(const tf2::Vector3& origin, const tf2::Vector3& target) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = origin.x();
    pose.position.y = origin.y();
    pose.position.z = origin.z();

    // Vetor Forward (X) aponta da Origem (Câmera) para o Alvo (Objeto)
    tf2::Vector3 forward = (target - origin).normalize();
    
    // Calcula Up e Right para construir a rotação
    tf2::Vector3 world_up(0, 0, 1);
    tf2::Vector3 right;
    
    // Tratamento de singularidade se estiver olhando quase verticalmente
    if (std::abs(forward.dot(world_up)) > 0.99) {
        right = tf2::Vector3(0, 1, 0).cross(forward).normalize();
    } else {
        right = world_up.cross(forward).normalize(); // Z cross X = Y (Esquerda no padrão RVIZ, mas Right vector matematico)
    }
    
    tf2::Vector3 up = forward.cross(right).normalize();

    // Cria matriz de rotação (Colunas: X, Y, Z)
    tf2::Matrix3x3 rot_mat;
    rot_mat.setValue(
        forward.x(), right.x(), up.x(),
        forward.y(), right.y(), up.y(),
        forward.z(), right.z(), up.z()
    );

    tf2::Quaternion q;
    rot_mat.getRotation(q);
    pose.orientation = tf2::toMsg(q);

    return pose;
}

std::vector<geometry_msgs::msg::Pose> ScanObject::getSortedScanPoses(const std::string& label)
{
    tf2::Vector3 robot_pos;
    {
        std::lock_guard<std::mutex> lock(robot_pos_mutex_);
        robot_pos = robot_pos_received_ ? robot_position_ : tf2::Vector3(0,0,0);
    }

    std::lock_guard<std::mutex> lock(objects_mutex_);
    auto it = detected_objects_.find(label);
    if (it == detected_objects_.end()) return {};

    const auto& obj_data = it->second;
    const auto& points = obj_data.valid_scan_grid;
    double center_z = obj_data.detection.bbox.center.position.z;

    std::map<int, std::vector<ScanPoint>> face_map;
    for(const auto& p : points) face_map[p.face_id].push_back(p);

    struct FaceInfo { int id; double dist; };
    std::vector<FaceInfo> faces_by_dist;

    for (const auto& [fid, pts] : face_map) {
        tf2::Vector3 centroid(0,0,0);
        for(const auto& p : pts) centroid += p.position;
        if (!pts.empty()) centroid /= (double)pts.size();
        
        double d = (centroid - robot_pos).length();
        faces_by_dist.push_back({fid, d});
    }

    std::sort(faces_by_dist.begin(), faces_by_dist.end(), [](const FaceInfo& a, const FaceInfo& b){
        return a.dist < b.dist;
    });

    int furthest_face_id = -1;
    if (!faces_by_dist.empty()) furthest_face_id = faces_by_dist.back().id;

    std::vector<geometry_msgs::msg::Pose> visible_above;
    std::vector<geometry_msgs::msg::Pose> visible_below;
    std::vector<geometry_msgs::msg::Pose> hidden_all; 

    for (const auto& f : faces_by_dist) 
    {
        int fid = f.id;
        auto& pts = face_map[fid];

        // Ordena pontos pela distância ao robô
        std::sort(pts.begin(), pts.end(), [&](const ScanPoint& a, const ScanPoint& b){
            return (a.position - robot_pos).length2() < (b.position - robot_pos).length2();
        });

        if (fid == furthest_face_id && faces_by_dist.size() > 1) 
        { 
            for(const auto& p : pts) 
            {
                // Cria Pose orientada para o centro do objeto
                hidden_all.push_back(createPoseLookingAt(p.position, p.target_center));
            }
        } 
        else 
        {
            for(const auto& p : pts) 
            {
                auto pose = createPoseLookingAt(p.position, p.target_center);
                if (p.position.z() > center_z) visible_above.push_back(pose);
                else visible_below.push_back(pose);
            }
        }
    }

    std::vector<geometry_msgs::msg::Pose> result;
    result.insert(result.end(), visible_above.begin(), visible_above.end());
    result.insert(result.end(), visible_below.begin(), visible_below.end());
    result.insert(result.end(), hidden_all.begin(), hidden_all.end());

    return result;
}

VoxelKey ScanObject::pointToVoxel(const tf2::Vector3& pt) {
    VoxelKey key;
    key.x = static_cast<int>(std::floor(pt.x() / voxel_map_resolution_));
    key.y = static_cast<int>(std::floor(pt.y() / voxel_map_resolution_));
    key.z = static_cast<int>(std::floor(pt.z() / voxel_map_resolution_));
    return key;
}

bool ScanObject::isRayBlocked(const tf2::Vector3& s, const tf2::Vector3& e, const std::string& l) {
    std::lock_guard<std::mutex> lock(voxel_mutex_);
    if (voxel_grid_.empty()) return false;
    tf2::Vector3 dir = e - s; double len = dir.length();
    if (len < 1e-4) return false; dir.normalize();
    for (double d = 0.0; d < (len - 0.05); d += ray_step_size_) {
        auto it = voxel_grid_.find(pointToVoxel(s + dir * d));
        if (it != voxel_grid_.end() && it->second != l) return true;
    }
    return false;
}

std::vector<ScanPoint> ScanObject::computeValidScanningGrid(const geometry_msgs::msg::Pose& p, const geometry_msgs::msg::Vector3& s, const std::string& l) {
    std::vector<ScanPoint> pts;
    tf2::Vector3 c(p.position.x, p.position.y, p.position.z);
    tf2::Quaternion q(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
    tf2::Matrix3x3 m(q);
    double dx = s.x/2.0 + ray_length_, dy = s.y/2.0 + ray_length_, dz = s.z/2.0 + ray_length_;
    
    struct F { tf2::Vector3 o, u, v; double ul, vl; int id; };
    std::vector<F> fs = {
        {{dx,-dy,-dz},{0,1,0},{0,0,1},2*dy,2*dz, 0}, 
        {{-dx,-dy,-dz},{0,1,0},{0,0,1},2*dy,2*dz, 1},
        {{-dx,dy,-dz},{1,0,0},{0,0,1},2*dx,2*dz, 2}, 
        {{-dx,-dy,-dz},{1,0,0},{0,0,1},2*dx,2*dz, 3}, 
        {{-dx,-dy,dz},{1,0,0},{0,1,0},2*dx,2*dy, 4}  
    };

    for(auto& f : fs) {
        for(double u=0; u<=f.ul+1e-4; u+=grid_resolution_) {
            for(double v=0; v<=f.vl+1e-4; v+=grid_resolution_) {
                tf2::Vector3 pw = c + m * (f.o + f.u*u + f.v*v);
                if(!isRayBlocked(pw, c, l)) pts.push_back({pw, c, f.id});
            }
        }
    }
    return pts;
}

} // namespace manipulation

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<manipulation::ScanObject>());
    rclcpp::shutdown();
    return 0;
}