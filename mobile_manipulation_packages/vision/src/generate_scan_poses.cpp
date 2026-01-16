#include "vision/GenerateScanPoses.hpp"

// INCLUSÃO CRUCIAL
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <map>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>

using namespace std::chrono_literals;

namespace vision {

geometry_msgs::msg::Pose createPoseLookingAt(const tf2::Vector3& origin, const tf2::Vector3& target) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = origin.x();
    pose.position.y = origin.y();
    pose.position.z = origin.z();

    tf2::Vector3 forward = (target - origin).normalize();
    tf2::Vector3 world_up(0, 0, 1);
    tf2::Vector3 right;
    
    if (std::abs(forward.dot(world_up)) > 0.99) right = tf2::Vector3(0, 1, 0).cross(forward).normalize();
    else right = world_up.cross(forward).normalize();
    
    tf2::Vector3 up = forward.cross(right).normalize();
    tf2::Matrix3x3 rot_mat;
    rot_mat.setValue(forward.x(), right.x(), up.x(), forward.y(), right.y(), up.y(), forward.z(), right.z(), up.z());
    
    tf2::Quaternion q; rot_mat.getRotation(q);
    pose.orientation = tf2::toMsg(q);
    return pose;
}

GenerateScanPoses::GenerateScanPoses(const rclcpp::NodeOptions & options)
 : Node("generate_scan_poses_visualizer", options),
   robot_position_(0.0, 0.0, 0.0),
   robot_pos_received_(false)
{
    this->declare_parameter<std::string>("target_frame", "world");
    this->declare_parameter<std::string>("odom_topic", "/odom");
    this->declare_parameter<double>("ray_length", 0.25); 
    this->declare_parameter<double>("grid_resolution", 0.04); 
    this->declare_parameter<double>("voxel_map_resolution", 0.02); 
    this->declare_parameter<double>("ray_step_size", 0.01);       
    this->declare_parameter<std::string>("target_object_id", "");
    this->declare_parameter<bool>("publish_markers", true); 

    this->declare_parameter<double>("camera_fov_h_deg", 153.16); 
    this->declare_parameter<double>("camera_fov_v_deg", 134.02);
    this->declare_parameter<double>("target_surface_res", 0.005);
    this->declare_parameter<double>("min_coverage_percent", 0.96);
    
    this->declare_parameter<double>("max_incidence_angle_deg", 80.0);

    target_frame_ = this->get_parameter("target_frame").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    ray_length_ = this->get_parameter("ray_length").as_double();
    grid_resolution_ = this->get_parameter("grid_resolution").as_double();
    voxel_map_resolution_ = this->get_parameter("voxel_map_resolution").as_double();
    ray_step_size_ = this->get_parameter("ray_step_size").as_double();
    target_object_id_ = this->get_parameter("target_object_id").as_string();
    publish_markers_ = this->get_parameter("publish_markers").as_bool();
    
    double fov_h = this->get_parameter("camera_fov_h_deg").as_double();
    double fov_v = this->get_parameter("camera_fov_v_deg").as_double();
    camera_fov_h_rad_ = fov_h * (M_PI / 180.0);
    camera_fov_v_rad_ = fov_v * (M_PI / 180.0);
    target_surface_res_ = this->get_parameter("target_surface_res").as_double();
    min_coverage_percent_ = this->get_parameter("min_coverage_percent").as_double();
    
    double max_inc_deg = this->get_parameter("max_incidence_angle_deg").as_double();
    max_incidence_angle_rad_ = max_inc_deg * (M_PI / 180.0);

    sub_detections_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
        "/bbox_3d_with_labels_0", 10, std::bind(&GenerateScanPoses::detectionCallback, this, std::placeholders::_1));

    sub_semantic_pcl_ = this->create_subscription<mobile_manipulation_interfaces::msg::SemanticPcl>(
        "/semantic_pcl_array", 10, std::bind(&GenerateScanPoses::semanticPclCallback, this, std::placeholders::_1));

    sub_odometry_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 10, std::bind(&GenerateScanPoses::odometryCallback, this, std::placeholders::_1));

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/visualization_marker_array", 10);
    animation_timer_ = this->create_wall_timer(200ms, std::bind(&GenerateScanPoses::animationTimerCallback, this));

    RCLCPP_INFO(this->get_logger(), "GenerateScanPoses: Modo Greedy Online (2 Passadas) Iniciado.");
}

std::vector<TargetVoxel> GenerateScanPoses::generateTargetVoxels(const vision_msgs::msg::Detection3D& detection) {
    std::vector<TargetVoxel> targets;
    tf2::Vector3 center(detection.bbox.center.position.x, detection.bbox.center.position.y, detection.bbox.center.position.z);
    tf2::Quaternion q(detection.bbox.center.orientation.x, detection.bbox.center.orientation.y, detection.bbox.center.orientation.z, detection.bbox.center.orientation.w);
    tf2::Matrix3x3 rot(q);
    double sx = detection.bbox.size.x, sy = detection.bbox.size.y;
    double res = target_surface_res_;

    auto fill_face = [&](tf2::Vector3 normal, double offset, double len_u, double len_v) {
        tf2::Vector3 world_normal = rot * normal;
        
        if (world_normal.z() < -0.5) return; 

        for(double u = -len_u/2.0; u <= len_u/2.0; u+=res) {
            for(double v = -len_v/2.0; v <= len_v/2.0; v+=res) {
                tf2::Vector3 local_pt;
                if (normal.x() != 0) local_pt = tf2::Vector3(offset, u, v);
                else if (normal.y() != 0) local_pt = tf2::Vector3(u, offset, v);
                else local_pt = tf2::Vector3(u, v, offset);
                targets.push_back({center + (rot * local_pt), world_normal, false});
            }
        }
    };
    fill_face({1,0,0}, sx/2, sy, detection.bbox.size.z);   // +X
    fill_face({-1,0,0}, -sx/2, sy, detection.bbox.size.z); // -X
    fill_face({0,1,0}, sy/2, sx, detection.bbox.size.z);   // +Y
    fill_face({0,-1,0}, -sy/2, sx, detection.bbox.size.z); // -Y
    fill_face({0,0,1}, detection.bbox.size.z/2, sx, sy);   // +Z
    return targets;
}


bool GenerateScanPoses::isVoxelVisible(const geometry_msgs::msg::Pose& pose, const TargetVoxel& voxel) {
    tf2::Transform tf_cam; 
    tf2::fromMsg(pose, tf_cam);
    
    
    tf2::Vector3 pt_cam = tf_cam.inverse() * voxel.position;
    
    double depth = pt_cam.x();
    
    if (depth < 0.05 || depth > (ray_length_ * 2.5)) return false;
    
    double angle_h = std::atan2(std::abs(pt_cam.y()), depth);
    double angle_v = std::atan2(std::abs(pt_cam.z()), depth);

    if (angle_h > (camera_fov_h_rad_ / 2.0)) return false;
    if (angle_v > (camera_fov_v_rad_ / 2.0)) return false;

    tf2::Vector3 cam_pos(pose.position.x, pose.position.y, pose.position.z);
    
    tf2::Vector3 voxel_to_cam = (cam_pos - voxel.position).normalize();
    
    double dot = voxel_to_cam.dot(voxel.normal);

    if (dot < 0.0) return false;


    if (std::acos(dot) > max_incidence_angle_rad_) return false;

    return true;
}

std::vector<geometry_msgs::msg::Pose> GenerateScanPoses::filterPosesByCoverage(
    const std::vector<geometry_msgs::msg::Pose>& candidates, 
    std::vector<TargetVoxel>& targets) 
{
    if (targets.empty() || candidates.empty()) return {};

    std::vector<geometry_msgs::msg::Pose> final_poses;
    std::vector<bool> pose_used(candidates.size(), false);
    int total_covered = 0;
    int total_targets = targets.size();

    // Passada 1: Greedy (maximizar cobertura nova)
    for (size_t i = 0; i < candidates.size(); ++i) {
        if ((double)total_covered / total_targets >= min_coverage_percent_) break;
        std::vector<int> newly_seen_indices;
        bool sees_old = false;

        for (size_t j = 0; j < targets.size(); ++j) {
            if (isVoxelVisible(candidates[i], targets[j])) {
                if (targets[j].covered) { sees_old = true; break; }
                else newly_seen_indices.push_back(j);
            }
        }

        if (!newly_seen_indices.empty() && !sees_old) {
            final_poses.push_back(candidates[i]);
            pose_used[i] = true;
            for (int idx : newly_seen_indices) {
                targets[idx].covered = true;
                total_covered++;
            }
        }
    }

    // Passada 2: Limpeza (pegar o resto)
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (pose_used[i]) continue;
        if ((double)total_covered / total_targets >= min_coverage_percent_) break;
        std::vector<int> newly_seen_indices;
        
        for (size_t j = 0; j < targets.size(); ++j) {
            if (targets[j].covered) continue;
            if (isVoxelVisible(candidates[i], targets[j])) {
                newly_seen_indices.push_back(j);
            }
        }

        if (!newly_seen_indices.empty()) {
            final_poses.push_back(candidates[i]);
            pose_used[i] = true;
            for (int idx : newly_seen_indices) {
                targets[idx].covered = true;
                total_covered++;
            }
        }
    }
    
    RCLCPP_INFO(this->get_logger(), "Filtro 2-Passadas: %lu -> %lu poses. Cobertura: %.1f%%", 
        candidates.size(), final_poses.size(), ((double)total_covered/total_targets)*100.0);

    return final_poses;
}

std::optional<std::pair<std::vector<geometry_msgs::msg::Pose>, tf2::Vector3>> 
GenerateScanPoses::getSortedScanPoses(const std::string& label)
{
    tf2::Vector3 robot_pos;
    {
        std::lock_guard<std::mutex> lock(robot_pos_mutex_);
        if (!robot_pos_received_) return std::nullopt; 
        robot_pos = robot_position_;
    }

    std::lock_guard<std::mutex> lock(objects_mutex_);
    auto it = detected_objects_.find(label);
    if (it == detected_objects_.end()) return std::nullopt;

    const auto& obj_data = it->second;
    
    std::vector<ScanPoint> pending_points = obj_data.valid_scan_grid;

    if (pending_points.empty()) return std::nullopt;

    std::vector<geometry_msgs::msg::Pose> sorted_candidates;
    sorted_candidates.reserve(pending_points.size());

  
    auto start_it = std::min_element(pending_points.begin(), pending_points.end(),
        [&](const ScanPoint& a, const ScanPoint& b) {
            return (a.position - robot_pos).length2() < (b.position - robot_pos).length2();
        });

    
    tf2::Vector3 current_pos = start_it->position;
    sorted_candidates.push_back(createPoseLookingAt(start_it->position, start_it->target_center));
    
    
    pending_points.erase(start_it);

    
    while (!pending_points.empty())
    {
        
        auto nearest_it = std::min_element(pending_points.begin(), pending_points.end(),
            [&](const ScanPoint& a, const ScanPoint& b) {
                return (a.position - current_pos).length2() < (b.position - current_pos).length2();
            });

        
        current_pos = nearest_it->position;
        
       
        sorted_candidates.push_back(createPoseLookingAt(nearest_it->position, nearest_it->target_center));

        
        pending_points.erase(nearest_it);
    }

    return std::make_pair(sorted_candidates, robot_pos);
}

std::vector<geometry_msgs::msg::Pose> GenerateScanPoses::getOptimizedScanPoses(
    const std::vector<geometry_msgs::msg::Pose>& input_poses, 
    const std::string& label)
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

    if (points.empty() || input_poses.empty()) return {};

    std::map<int, std::vector<ScanPoint>> face_map;
    for(const auto& p : points) face_map[p.face_id].push_back(p);

    std::vector<int> face_order;
    for(auto const& [fid, _] : face_map) face_order.push_back(fid);
    
    
    std::sort(face_order.begin(), face_order.end(), [&](int a, int b)
    {
        return (face_map[a][0].position - robot_pos).length() < (face_map[b][0].position - robot_pos).length();
    });

   
    std::vector<geometry_msgs::msg::Pose> reordered_poses;
    reordered_poses.reserve(input_poses.size());

    for(int fid : face_order) 
    {
        auto& pts = face_map[fid];
        
        std::sort(pts.begin(), pts.end(), [&](const ScanPoint& a, const ScanPoint& b)
        {
            return (a.position - robot_pos).length2() < (b.position - robot_pos).length2();
        });
        
        for(const auto& p : pts) 
        {
            
            geometry_msgs::msg::Pose candidate_pose = createPoseLookingAt(p.position, p.target_center);

          
            bool exists_in_input = false;
            for(const auto& in_pose : input_poses)
            {
               
                double dist_sq = std::pow(in_pose.position.x - candidate_pose.position.x, 2) +
                                 std::pow(in_pose.position.y - candidate_pose.position.y, 2) +
                                 std::pow(in_pose.position.z - candidate_pose.position.z, 2);
                
                if(dist_sq < 0.0001) { 
                    exists_in_input = true;
                    break;
                }
            }

            if (exists_in_input) {
                reordered_poses.push_back(candidate_pose);
            }
        }
    }

   
    std::vector<TargetVoxel> targets = generateTargetVoxels(obj_data.detection);
    std::vector<geometry_msgs::msg::Pose> optimized_poses = filterPosesByCoverage(reordered_poses, targets);
    debug_voxels_ = targets;
    poses_to_animate_ = optimized_poses;

    return optimized_poses;
}

void GenerateScanPoses::detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg) 
{
    last_header_.stamp = msg->header.stamp;
    last_header_.frame_id = target_frame_;

    {
        std::lock_guard<std::mutex> lock(objects_mutex_);
        detected_objects_.clear();
        for (const auto& detection : msg->detections) {
            if (detection.results.empty()) continue;
            std::string label = detection.results[0].hypothesis.class_id;
            ObjectData obj_data; obj_data.label = label; obj_data.detection = detection; obj_data.header = last_header_;
            if (target_object_id_.empty() || target_object_id_ == label) {
                obj_data.valid_scan_grid = computeValidScanningGrid(detection.bbox.center, detection.bbox.size, label);
            }
            detected_objects_[label] = obj_data;
        }
    }
    
    std::lock_guard<std::mutex> anim_lock(anim_mutex_);
    std::string label = target_object_id_.empty() && !msg->detections.empty() ? msg->detections[0].results[0].hypothesis.class_id : target_object_id_;
    
    // if(!label.empty()) poses_to_animate_ = getSortedScanPoses(label);
}

void GenerateScanPoses::animationTimerCallback() 
{
    if (!publish_markers_) return;
    std::lock_guard<std::mutex> lock(anim_mutex_);
    if (poses_to_animate_.empty()) return;

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker del; del.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(del);

    for(size_t i=0; i<poses_to_animate_.size(); ++i) {
        visualization_msgs::msg::Marker arrow;
        arrow.header = last_header_; arrow.header.stamp = this->now();
        arrow.ns = "final_poses"; arrow.id = i; 
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;
        arrow.pose = poses_to_animate_[i];
        arrow.scale.x = 0.10; arrow.scale.y = 0.015; arrow.scale.z = 0.015;
        arrow.color.r = 0.0; arrow.color.g = 1.0; arrow.color.b = 0.0; arrow.color.a = 1.0;
        markers.markers.push_back(arrow);
    }
    

    marker_pub_->publish(markers);
}

VoxelKey GenerateScanPoses::pointToVoxel(const tf2::Vector3& pt) 
{
    VoxelKey key; key.x = static_cast<int>(std::floor(pt.x() / voxel_map_resolution_));
    key.y = static_cast<int>(std::floor(pt.y() / voxel_map_resolution_));
    key.z = static_cast<int>(std::floor(pt.z() / voxel_map_resolution_));
    return key;
}

bool GenerateScanPoses::isRayBlocked(const tf2::Vector3& s, const tf2::Vector3& e, const std::string& l) 
{
    std::lock_guard<std::mutex> lock(voxel_mutex_);

    if (voxel_grid_.empty()) return false;

    tf2::Vector3 dir = e - s; double len = dir.length();
    
    if (len < 1e-4) return false; 

    dir.normalize();

    for (double d = 0.0; d < (len - 0.05); d += ray_step_size_) 
    {
        auto it = voxel_grid_.find(pointToVoxel(s + dir * d));
        if (it != voxel_grid_.end() && it->second != l) return true;
    }

    return false;
}

std::vector<ScanPoint> GenerateScanPoses::computeValidScanningGrid(const geometry_msgs::msg::Pose& p, 
    const geometry_msgs::msg::Vector3& s, 
    const std::string& l) {
    std::vector<ScanPoint> pts;
    tf2::Vector3 c(p.position.x, p.position.y, p.position.z);
    tf2::Quaternion q(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
    tf2::Matrix3x3 m(q);
    double dx = s.x/2.0 + ray_length_, dy = s.y/2.0 + ray_length_, dz = s.z/2.0 + ray_length_;
    double object_base_z = p.position.z - (s.z / 2.0);
    double min_safe_height = object_base_z + 0.05; 

    struct F { tf2::Vector3 o, u, v; double ul, vl; int id; };
    std::vector<F> fs = {
        {{dx,-dy,-dz},{0,1,0},{0,0,1},2*dy,2*dz, 0}, 
        {{-dx,-dy,-dz},{0,1,0},{0,0,1},2*dy,2*dz, 1},
        {{-dx,dy,-dz},{1,0,0},{0,0,1},2*dx,2*dz, 2}, 
        {{-dx,-dy,-dz},{1,0,0},{0,0,1},2*dx,2*dz, 3}, 
        {{-dx,-dy,dz},{1,0,0},{0,1,0},2*dx,2*dy, 4}
    };

    for(auto& f : fs) {
        for(double u=0; u<=f.ul+1e-4; u+=grid_resolution_) 
        {
            for(double v=0; v<=f.vl+1e-4; v+=grid_resolution_) 
            {
                tf2::Vector3 pw = c + m * (f.o + f.u*u + f.v*v);
                if (pw.z() < min_safe_height) continue;
                if(!isRayBlocked(pw, c, l)) {
                    ScanPoint sp;
                    sp.position = pw;
                    sp.target_center = c;
                    sp.face_id = f.id;
                    sp.pose = createPoseLookingAt(pw, c);
                    pts.push_back(sp);
                }
            }
        }
    }
    return pts;
}

void GenerateScanPoses::odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg) 
{
    std::lock_guard<std::mutex> lock(robot_pos_mutex_);
    robot_position_ = tf2::Vector3(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    robot_pos_received_ = true;
}

void GenerateScanPoses::semanticPclCallback(const mobile_manipulation_interfaces::msg::SemanticPcl::SharedPtr msg) 
{
    std::lock_guard<std::mutex> lock(voxel_mutex_);

    voxel_grid_.clear();
    if (msg->labels.size() != msg->clouds.size()) return;

    for (size_t i = 0; i < msg->labels.size(); ++i) 
    {
        std::string label = msg->labels[i];
        const auto& cloud = msg->clouds[i];

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x"), iter_y(cloud, "y"), iter_z(cloud, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) 
        {
            VoxelKey key; key.x = static_cast<int>(std::floor(*iter_x / voxel_map_resolution_));
            key.y = static_cast<int>(std::floor(*iter_y / voxel_map_resolution_));
            key.z = static_cast<int>(std::floor(*iter_z / voxel_map_resolution_));
            voxel_grid_[key] = label;
        }
    }
}

} // namespace vision

RCLCPP_COMPONENTS_REGISTER_NODE(vision::GenerateScanPoses)