#include "vision/GenerateGraspPoses.hpp" 

#include <rclcpp_components/register_node_macro.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <sys/resource.h>
#include <limits> 

// TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

// PCL
#include <pcl/io/vtk_lib_io.h>
#include <pcl/filters/random_sample.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std::chrono_literals;

namespace vision 
{

GenerateGraspPoses::GenerateGraspPoses(const rclcpp::NodeOptions & options) 
: Node("generate_grasp_poses", options) 
{
    this->declare_parameter<bool>("use_pcd_file", false);
    this->declare_parameter<std::string>("pcd_path", "/home/momesso/pibic/nuvem.pcd");
    
    this->declare_parameter<std::string>("gripper_mesh_path", "/home/momesso/gripper_model.obj");
    this->declare_parameter<double>("gripper_mesh_scale", 1.0);
    
    this->declare_parameter<double>("mesh_offset_x", 0.025);
    this->declare_parameter<double>("mesh_offset_y", 0.0);
    this->declare_parameter<double>("mesh_offset_z", 0.0);
    
    this->declare_parameter<double>("mesh_rot_roll", 0.0);
    this->declare_parameter<double>("mesh_rot_pitch", -1.57);
    this->declare_parameter<double>("mesh_rot_yaw", 0.0); 

    this->declare_parameter<double>("grid_res", 0.005);
    this->declare_parameter<double>("cloud_voxel_size", 0.001);
    
    this->declare_parameter<double>("cylinder_radius", 0.005); 
    this->declare_parameter<double>("cylinder_height", 0.005);
    this->declare_parameter<double>("analysis_step_size", 0.01);
    
    this->declare_parameter<double>("max_gripper_width", 0.07); 
    this->declare_parameter<double>("finger_offset", 0.027); 
    
    this->declare_parameter<int>("min_points_per_segment", 2);
    this->declare_parameter<double>("weight_orientation", 0.5); 
    this->declare_parameter<double>("weight_symmetry", 0.5);
    this->declare_parameter<double>("weight_planarity", 0.0);
    
    this->declare_parameter<bool>("use_mean_filter", true); 
    this->declare_parameter<int>("mean_filter_k", 15);

    this->declare_parameter<int>("num_best_grasps", 1);
    this->declare_parameter<double>("rotation_step_deg", 30.0);

    this->declare_parameter<int>("num_random_orientations", 20);

    use_pcd_file = this->get_parameter("use_pcd_file").as_bool();
    pcd_path_ = this->get_parameter("pcd_path").as_string();
    
    gripper_mesh_path_ = this->get_parameter("gripper_mesh_path").as_string();
    gripper_mesh_scale_ = static_cast<float>(this->get_parameter("gripper_mesh_scale").as_double());

    mesh_offset_x_ = static_cast<float>(this->get_parameter("mesh_offset_x").as_double());
    mesh_offset_y_ = static_cast<float>(this->get_parameter("mesh_offset_y").as_double());
    mesh_offset_z_ = static_cast<float>(this->get_parameter("mesh_offset_z").as_double());

    mesh_rot_roll_ = static_cast<float>(this->get_parameter("mesh_rot_roll").as_double());
    mesh_rot_pitch_ = static_cast<float>(this->get_parameter("mesh_rot_pitch").as_double());
    mesh_rot_yaw_ = static_cast<float>(this->get_parameter("mesh_rot_yaw").as_double());

    grid_res_ = static_cast<float>(this->get_parameter("grid_res").as_double());
    cloud_voxel_size_ = static_cast<float>(this->get_parameter("cloud_voxel_size").as_double());

    cylinder_radius_ = static_cast<float>(this->get_parameter("cylinder_radius").as_double());
    cylinder_height_ = static_cast<float>(this->get_parameter("cylinder_height").as_double());
    analysis_step_size_ = static_cast<float>(this->get_parameter("analysis_step_size").as_double());

    max_gripper_width_ = static_cast<float>(this->get_parameter("max_gripper_width").as_double());
    finger_offset_ = static_cast<float>(this->get_parameter("finger_offset").as_double());

    min_points_per_segment_ = this->get_parameter("min_points_per_segment").as_int();
    
    weight_orientation_ = static_cast<float>(this->get_parameter("weight_orientation").as_double());
    weight_symmetry_ = static_cast<float>(this->get_parameter("weight_symmetry").as_double());
    weight_planarity_ = static_cast<float>(this->get_parameter("weight_planarity").as_double());

    mean_filter = this->get_parameter("use_mean_filter").as_bool();
    mean_filter_k_ = this->get_parameter("mean_filter_k").as_int();

    num_best_grasps_ = this->get_parameter("num_best_grasps").as_int();
    total_orientations_ = this->get_parameter("num_random_orientations").as_int();
    rotation_step_deg_ = static_cast<float>(this->get_parameter("rotation_step_deg").as_double());
    

    pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("input_cloud", 10);
    pub_rays_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("candidate_rays", 10);
    pub_bbox_  = this->create_publisher<visualization_msgs::msg::Marker>("bounding_box", 10);
    pub_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("best_grasps_markers", 10);
    pub_poses_ = this->create_publisher<geometry_msgs::msg::PoseArray>("best_grasps_poses", 10);
    pub_debug_inliers_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_ray_inliers", 10);
    
    pub_gripper_model_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_gripper_model", 10);
    pub_gripper_boxes_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("debug_gripper_boxes", 10);
    pub_debug_grasps_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_grasps_cloud", 10);

    stored_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    gripper_dense_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>); 
    
    
    extractBoundingBoxesFromOBJ(); 
    
    
    publishGripperModel();
    publishGripperCollisionBoxes();

    RCLCPP_INFO(this->get_logger(), "MODO ARQUIVO: Carregando PCD de %s...", pcd_path_.c_str());

    if(use_pcd_file == true)
    {
        loadAndProcess(pcd_path_);
    }
   
    timer_ = this->create_wall_timer(1000ms, std::bind(&GenerateGraspPoses::timerCallback, this));
}


void GenerateGraspPoses::loadAndProcess(const std::string& path)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *temp_cloud) == -1) 
    {
        RCLCPP_ERROR(this->get_logger(), "Falha ao ler arquivo PCD: %s", path.c_str());
        return;
    }
    
    processCloud(temp_cloud, temp_cloud);
}

geometry_msgs::msg::PoseArray GenerateGraspPoses::processCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr target, pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment)
{
    if (!target || target->empty()) return geometry_msgs::msg::PoseArray();

    // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    // sor.setInputCloud(target);
    // sor.setMeanK(50); 
    // sor.setStddevMulThresh(2.0); 
    // sor.filter(*target);

    if (use_pcd_file == true)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI); 
        float rot_x = dis(gen); float rot_y = dis(gen); float rot_z = dis(gen);
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*target, centroid);
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.translation() = centroid.head<3>();
        transform.rotate(Eigen::AngleAxisf(rot_x, Eigen::Vector3f::UnitX())); 
        transform.rotate(Eigen::AngleAxisf(rot_y, Eigen::Vector3f::UnitY()));
        transform.rotate(Eigen::AngleAxisf(rot_z, Eigen::Vector3f::UnitZ())); 
        transform.translate(-centroid.head<3>());
        pcl::transformPointCloud(*target, *target, transform);
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (use_pcd_file && cloud_voxel_size_ > 0.001f) {
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(target);
        sor.setLeafSize(cloud_voxel_size_, cloud_voxel_size_, cloud_voxel_size_);
        sor.filter(*voxel_cloud);
    } else {
        *voxel_cloud = *target;
    }

    if (mean_filter) 
    {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(voxel_cloud);
        stored_cloud_->points.resize(voxel_cloud->points.size());
        stored_cloud_->width = voxel_cloud->width;
        stored_cloud_->height = voxel_cloud->height;
        stored_cloud_->is_dense = voxel_cloud->is_dense;
        stored_cloud_->header = voxel_cloud->header; 
        int K = mean_filter_k_; 

        tbb::parallel_for(tbb::blocked_range<size_t>(0, voxel_cloud->points.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::vector<int> pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    if (kdtree.nearestKSearch(voxel_cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                        float sum_x = 0, sum_y = 0, sum_z = 0;
                        int valid_pts = 0;
                        for (int j = 0; j < K; ++j) {
                            const auto& neighbor = voxel_cloud->points[pointIdxNKNSearch[j]];
                            sum_x += neighbor.x; sum_y += neighbor.y; sum_z += neighbor.z; valid_pts++;
                        }
                        stored_cloud_->points[i].x = sum_x / valid_pts;
                        stored_cloud_->points[i].y = sum_y / valid_pts;
                        stored_cloud_->points[i].z = sum_z / valid_pts;
                    } else {
                        stored_cloud_->points[i] = voxel_cloud->points[i];
                    }
                }
            }
        );
    } else {
        *stored_cloud_ = *voxel_cloud;
    }
    
    stored_cloud_->header.frame_id = "world";
    pcl::getMinMax3D(*stored_cloud_, min_pt_, max_pt_);
    float padding = 0.03; 
    min_pt_.array() -= padding; max_pt_.array() += padding;
    
    if (total_orientations_ < 1) total_orientations_ = 1;
    
    std::vector<Eigen::Matrix3f> search_matrices;
    search_matrices.reserve(total_orientations_);
    search_matrices.push_back(Eigen::Matrix3f::Identity());

    if (total_orientations_ > 1) 
    {
        std::random_device rd; std::mt19937 gen(rd()); 
        std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI);
        for (int i = 0; i < total_orientations_ - 1; ++i) {
            float roll  = dis(gen); float pitch = dis(gen); float yaw   = dis(gen);
            Eigen::Matrix3f m;
            m = Eigen::AngleAxisf(roll,  Eigen::Vector3f::UnitX())
                * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
                * Eigen::AngleAxisf(yaw,   Eigen::Vector3f::UnitZ());
            search_matrices.push_back(m);
        }
    }
    
    all_candidates_ = generateMultiOrientedRays(min_pt_, max_pt_, grid_res_, search_matrices);
    return evaluateGrasps(target_environment);
}

std::vector<geometry_msgs::msg::Pose> GenerateGraspPoses::generateMultiOrientedRays(
    const Eigen::Vector4f& min, const Eigen::Vector4f& max, float res, 
    const std::vector<Eigen::Matrix3f>& rotation_matrices) 
{
    std::vector<geometry_msgs::msg::Pose> poses;
    Eigen::Vector3f center = (min.head<3>() + max.head<3>()) / 2.0f;
    Eigen::Vector3f size = max.head<3>() - min.head<3>();
    Eigen::Vector3f start_dims = size; 
    
    auto add_rotated_ray = [&](Eigen::Vector3f local_pos, Eigen::Vector3f local_dir, const Eigen::Matrix3f& rot_mat) 
    {
        Eigen::Vector3f global_pos = (rot_mat * local_pos) + center;
        Eigen::Vector3f global_dir = rot_mat * local_dir;

        geometry_msgs::msg::Pose p;
        p.position.x = global_pos.x(); 
        p.position.y = global_pos.y(); 
        p.position.z = global_pos.z();
        
        Eigen::Quaternionf q; 
        q.setFromTwoVectors(Eigen::Vector3f::UnitX(), global_dir);
        p.orientation.x = q.x(); p.orientation.y = q.y(); 
        p.orientation.z = q.z(); p.orientation.w = q.w();
        poses.push_back(p);
    };

    for (const auto& R : rotation_matrices)
    {
        float half_x = start_dims.x() / 2.0f;
        float half_y = start_dims.y() / 2.0f;
        float half_z = start_dims.z() / 2.0f;

        for(float y = -half_y; y < half_y; y += res)
            for(float z = -half_z; z < half_z; z += res) add_rotated_ray({half_x, y + res/2, z + res/2}, {-1, 0, 0}, R);

        for(float x = -half_x; x < half_x; x += res)
            for(float z = -half_z; z < half_z; z += res) add_rotated_ray({x + res/2, half_y, z + res/2}, {0, -1, 0}, R);

        for(float x = -half_x; x < half_x; x += res)
            for(float y = -half_y; y < half_y; y += res) add_rotated_ray({x + res/2, y + res/2, half_z}, {0, 0, -1}, R);
    }
    return poses;
}

StepAnalysis GenerateGraspPoses::analyzeLocalCylinder(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& ray_dir,
    float radius,
    float height)
{
    StepAnalysis result; result.valid = false; result.center = center;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Vector3f u = ray_dir.unitOrthogonal(); 
    Eigen::Vector3f v = ray_dir.cross(u);
    std::vector<int> sector_counts(12, 0);

    for (const auto& pt : cloud->points) 
    {
        Eigen::Vector3f p(pt.x, pt.y, pt.z);
        Eigen::Vector3f diff = p - center;
        if (std::abs(diff.dot(ray_dir)) > height/2.0f) continue;
        if (diff.cross(ray_dir).norm() > radius) continue;

        local_cloud->points.push_back(pt);
        float coord_u = diff.dot(u);
        float coord_v = diff.dot(v);
        float deg = std::atan2(coord_v, coord_u) * 180.0f / M_PI;
        if (deg < 0) deg += 360.0f;
        int sector_idx = std::min((int)(deg / 30.0f), 11);
        sector_counts[sector_idx]++;
    }

    result.point_count = local_cloud->size();
    if (result.point_count <= min_points_per_segment_) return result;

    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(local_cloud);
    
    Eigen::Vector3f ev = pca.getEigenValues();
    float sum = ev.sum();
    result.curvature = (sum > 1e-6) ? ev[2] / sum : 0.0;
    Eigen::Vector3f normal = pca.getEigenVectors().col(2);
    result.normal_vector = normal;
    float dot = std::abs(ray_dir.dot(normal));
    result.angle_to_normal_deg = std::acos(dot) * 180.0f / M_PI;
    
    int occupied = 0;
    for (int c : sector_counts) if (c > 0) occupied++;
    result.symmetry_score = (float)occupied / 12.0f;
    result.valid = true;
    return result;
}


void GenerateGraspPoses::extractBoundingBoxesFromOBJ()
{
    RCLCPP_INFO(this->get_logger(), "Carregando modelo 3D (Assimp) de: %s", gripper_mesh_path_.c_str());
    
    Assimp::Importer importer;
    
    const aiScene* scene = importer.ReadFile(gripper_mesh_path_, aiProcess_Triangulate);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        RCLCPP_ERROR(this->get_logger(), "ERRO CRÍTICO ASSIMP: %s", importer.GetErrorString());
        return;
    }

    gripper_boxes_.clear();
    gripper_dense_cloud_->clear();

    
    std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> colors = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255}, {255, 0, 255}
    };

    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];
        
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        float max_z = std::numeric_limits<float>::lowest();

        auto color = colors[i % colors.size()];
        uint8_t r = std::get<0>(color);
        uint8_t g = std::get<1>(color);
        uint8_t b = std::get<2>(color);

        for (unsigned int v = 0; v < mesh->mNumVertices; v++) 
        {
            aiVector3D p_raw = mesh->mVertices[v];
            
           
            Eigen::Vector3f p_local(
                p_raw.x * gripper_mesh_scale_,
                p_raw.y * gripper_mesh_scale_,
                p_raw.z * gripper_mesh_scale_
            );

            
            if (p_local.x() < min_x) min_x = p_local.x();
            if (p_local.y() < min_y) min_y = p_local.y();
            if (p_local.z() < min_z) min_z = p_local.z();
            
            if (p_local.x() > max_x) max_x = p_local.x();
            if (p_local.y() > max_y) max_y = p_local.y();
            if (p_local.z() > max_z) max_z = p_local.z();

            
            pcl::PointXYZRGB pt_rgb;
            pt_rgb.x = p_local.x();
            pt_rgb.y = p_local.y();
            pt_rgb.z = p_local.z();
            pt_rgb.r = r; pt_rgb.g = g; pt_rgb.b = b;
            gripper_dense_cloud_->points.push_back(pt_rgb);
        }

        
        LocalBox box;
        float margin = 0.001f;
        box.min_pt = Eigen::Vector3f(min_x - margin, min_y - margin, min_z - margin);
        box.max_pt = Eigen::Vector3f(max_x + margin, max_y + margin, max_z + margin);
        
        
        box.center = (box.min_pt + box.max_pt) / 2.0f;
        box.dimensions = box.max_pt - box.min_pt;

        gripper_boxes_.push_back(box);
    }
    
    gripper_dense_cloud_->width = gripper_dense_cloud_->points.size();
    gripper_dense_cloud_->height = 1;
    gripper_dense_cloud_->is_dense = true;

    RCLCPP_INFO(this->get_logger(), "Extraídas %lu caixas de colisão de %s", gripper_boxes_.size(), gripper_mesh_path_.c_str());
}

void GenerateGraspPoses::publishGripperModel()
{
    if (!gripper_dense_cloud_ || gripper_dense_cloud_->empty()) return;
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*gripper_dense_cloud_, msg);
    msg.header.frame_id = "world"; 
    msg.header.stamp = this->now();
    pub_gripper_model_->publish(msg);
}


void GenerateGraspPoses::publishGripperCollisionBoxes()
{
    if (gripper_boxes_.empty()) return;

    visualization_msgs::msg::MarkerArray ma;
    auto t = this->now();

    
    Eigen::Affine3f visual_tf = Eigen::Affine3f::Identity();
    visual_tf.translation() = Eigen::Vector3f(mesh_offset_x_, mesh_offset_y_, mesh_offset_z_);
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ());
    visual_tf.linear() = rot;
    Eigen::Quaternionf q_rot(rot);

    for(size_t i = 0; i < gripper_boxes_.size(); i++)
    {
        const auto& box = gripper_boxes_[i];

        visualization_msgs::msg::Marker m;
        m.header.frame_id = "world";
        m.header.stamp = t;
        m.ns = "gripper_collision_boxes";
        m.id = i;
        m.type = visualization_msgs::msg::Marker::CUBE;
        m.action = visualization_msgs::msg::Marker::ADD;

        Eigen::Vector3f center_transformed = visual_tf * box.center;

        m.pose.position.x = center_transformed.x();
        m.pose.position.y = center_transformed.y();
        m.pose.position.z = center_transformed.z();

        
        m.pose.orientation.x = q_rot.x();
        m.pose.orientation.y = q_rot.y();
        m.pose.orientation.z = q_rot.z();
        m.pose.orientation.w = q_rot.w();

        m.scale.x = box.dimensions.x();
        m.scale.y = box.dimensions.y();
        m.scale.z = box.dimensions.z();

        if (i == 0)      { m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; }
        else if (i == 1) { m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; }
        else             { m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0; }
        m.color.a = 0.4; 

        ma.markers.push_back(m);
    }
    pub_gripper_boxes_->publish(ma);
}


bool GenerateGraspPoses::check_collision(const ScoredGrasp& grasp, const pcl::KdTreeFLANN<pcl::PointXYZ>& env_kdtree)
{
    if (gripper_boxes_.empty()) return true; 
    if (!env_kdtree.getInputCloud() || env_kdtree.getInputCloud()->empty()) return true;

   
    Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
    Eigen::Quaternionf grasp_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);
    
    Eigen::Affine3f tf_world_to_tcp = Eigen::Translation3f(grasp_pos) * grasp_rot;
    
    
    Eigen::Affine3f tf_tcp_to_geometry = Eigen::Affine3f::Identity();
    
    Eigen::Matrix3f rot_geom;
    rot_geom = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
             * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ());
    
    tf_tcp_to_geometry.linear() = rot_geom;
    tf_tcp_to_geometry.translation() = Eigen::Vector3f(mesh_offset_x_, mesh_offset_y_, mesh_offset_z_);

    
    Eigen::Affine3f tf_world_to_local = (tf_world_to_tcp * tf_tcp_to_geometry).inverse();

    float search_radius = 0.15f; 
    std::vector<int> pointIdx;
    std::vector<float> pointSqDist;
    pcl::PointXYZ searchPoint;
    searchPoint.x = grasp_pos.x(); searchPoint.y = grasp_pos.y(); searchPoint.z = grasp_pos.z();

    
    if (env_kdtree.radiusSearch(searchPoint, search_radius, pointIdx, pointSqDist) > 0)
    {
        for (int idx : pointIdx)
        {
            auto pt_world = env_kdtree.getInputCloud()->points[idx];
            
           
            Eigen::Vector3f p_w(pt_world.x, pt_world.y, pt_world.z);
            Eigen::Vector3f p_local = tf_world_to_local * p_w;

           
            for (const auto& box : gripper_boxes_)
            {
                if (p_local.x() >= box.min_pt.x() && p_local.x() <= box.max_pt.x() &&
                    p_local.y() >= box.min_pt.y() && p_local.y() <= box.max_pt.y() &&
                    p_local.z() >= box.min_pt.z() && p_local.z() <= box.max_pt.z())
                {
                    return false; 
                }
            }
        }
    }
    return true; 
}

Eigen::Quaternionf GenerateGraspPoses::findBestOrientation(const Eigen::Vector3f& p_f1, const Eigen::Vector3f& p_f2)
{
    Eigen::Vector3f finger_axis = (p_f2 - p_f1).normalized();
    
    Eigen::Vector3f base_approach;
    if (std::abs(finger_axis.dot(Eigen::Vector3f::UnitZ())) > 0.95) {
            base_approach = finger_axis.cross(Eigen::Vector3f::UnitX()).normalized();
    } else {
            base_approach = finger_axis.cross(Eigen::Vector3f::UnitZ()).normalized();
    }

    float best_score = -1000.0;
    Eigen::Quaternionf best_q = Eigen::Quaternionf::Identity();
    
    double step_rad = rotation_step_deg_ * M_PI / 180.0;

    for (double angle = 0; angle < 2 * M_PI; angle += step_rad) 
    {
        Eigen::AngleAxisf rotation(angle, finger_axis);
        Eigen::Vector3f candidate_approach = rotation * base_approach; 
        Eigen::Vector3f candidate_up = candidate_approach.cross(finger_axis);

        Eigen::Matrix3f rot_mat;
        rot_mat.col(0) = candidate_approach; 
        rot_mat.col(1) = finger_axis;        
        rot_mat.col(2) = candidate_up;       

        float score = candidate_approach.dot(-Eigen::Vector3f::UnitZ());

        if (score > best_score) {
            best_score = score;
            best_q = Eigen::Quaternionf(rot_mat);
        }
    }
    return best_q;
}

geometry_msgs::msg::PoseArray GenerateGraspPoses::evaluateGrasps(pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment)
{
    hit_candidates_.clear(); 

    pcl::KdTreeFLANN<pcl::PointXYZ> env_kdtree;
    if (target_environment->empty()) {
        RCLCPP_WARN(this->get_logger(), "Ambiente vazio, ignorando colisão.");
    } else {
        env_kdtree.setInputCloud(target_environment);
    }

    float voxel_size = 0.01f;
    std::unordered_map<long, VoxelBucket> voxel_grid;
    
    auto get_key = [&](int x, int y, int z) -> long {
        return ((long)x * 73856093) ^ ((long)y * 19349663) ^ ((long)z * 83492791);
    };
    
    for (const auto& pt : stored_cloud_->points) {
        int ix = std::floor(pt.x / voxel_size);
        int iy = std::floor(pt.y / voxel_size);
        int iz = std::floor(pt.z / voxel_size);
        long key = get_key(ix, iy, iz);
        
        if (voxel_grid.find(key) == voxel_grid.end()) {
            voxel_grid[key].center = Eigen::Vector3f(
                (ix + 0.5f) * voxel_size, 
                (iy + 0.5f) * voxel_size, 
                (iz + 0.5f) * voxel_size
            );
        }
        voxel_grid[key].points.push_back(pt);
    }

    float voxel_radius = (voxel_size * 1.73205f) / 2.0f;
    float voxel_check_threshold = cylinder_radius_ + voxel_radius; 
    
    tbb::enumerable_thread_specific<std::vector<ScoredGrasp>> local_best_grasps;
    tbb::enumerable_thread_specific<std::vector<geometry_msgs::msg::Pose>> local_hit_candidates;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, all_candidates_.size()),
        [&](const tbb::blocked_range<size_t>& range)
        {
            auto& my_scored_grasps = local_best_grasps.local();
            auto& my_hit_candidates = local_hit_candidates.local();

            for (size_t i = range.begin(); i != range.end(); ++i) 
            {
                const auto& raw_pose = all_candidates_[i];
                Eigen::Quaternionf q(raw_pose.orientation.w, raw_pose.orientation.x, raw_pose.orientation.y, raw_pose.orientation.z);
                Eigen::Vector3f ray_origin(raw_pose.position.x, raw_pose.position.y, raw_pose.position.z);
                Eigen::Vector3f ray_dir = q * Eigen::Vector3f::UnitX(); 

                float t_min = 1e6, t_max = -1e6;
                bool hit = false;
                
                pcl::PointCloud<pcl::PointXYZ> current_inliers;
                pcl::PointCloud<pcl::PointXYZ>::Ptr current_inliers_ptr(new pcl::PointCloud<pcl::PointXYZ>);

                for (const auto& [key, bucket] : voxel_grid) 
                {
                    Eigen::Vector3f diff = bucket.center - ray_origin;
                    float dist_sq_to_ray = (diff.cross(ray_dir)).squaredNorm();
                    if (dist_sq_to_ray > (voxel_check_threshold * voxel_check_threshold)) continue;

                    for (const auto& pt : bucket.points) 
                    {
                        Eigen::Vector3f p(pt.x, pt.y, pt.z);
                        float t = (p - ray_origin).dot(ray_dir);
                        if ((p - (ray_origin + t*ray_dir)).norm() < cylinder_radius_) 
                        {
                            if (t < t_min) t_min = t;
                            if (t > t_max) t_max = t;
                            hit = true;
                            current_inliers.points.push_back(pt);
                            current_inliers_ptr->points.push_back(pt);
                        }
                    }
                }
                
                if (!hit || (t_max - t_min) < 0.005) continue;
                my_hit_candidates.push_back(raw_pose);

                std::vector<StepAnalysis> steps;
                for (float t = t_min; t <= t_max; t += analysis_step_size_) 
                {
                    Eigen::Vector3f center = ray_origin + ray_dir * t;
                    StepAnalysis res = analyzeLocalCylinder(current_inliers_ptr, center, ray_dir, cylinder_radius_, cylinder_height_);
                    if (res.valid) steps.push_back(res);
                }
                if (steps.empty()) continue;

                StepAnalysis& entry = steps.front();
                StepAnalysis& exit = steps.back();

                float real_thickness = t_max - t_min;
                if (real_thickness > max_gripper_width_) continue; 

                float current_offset = finger_offset_;
                float total_width_needed = real_thickness + (2.0f * current_offset);
                if (total_width_needed > max_gripper_width_) {
                    current_offset = (max_gripper_width_ - real_thickness) / 2.0f;
                    if (current_offset < 0.01f) current_offset = 0.01f;
                }
                total_width_needed = real_thickness + (2.0f * current_offset);
                if(total_width_needed > max_gripper_width_) continue;

                Eigen::Vector3f p_f1 = ray_origin + ray_dir * (t_min - current_offset);
                Eigen::Vector3f p_f2 = ray_origin + ray_dir * (t_max + current_offset);
                Eigen::Vector3f center_grasp = (p_f1 + p_f2) / 2.0f;
                Eigen::Quaternionf best_orientation = findBestOrientation(p_f1, p_f2);

                float score_ang_entry = 1.0f - (std::min(entry.angle_to_normal_deg, 90.0f) / 90.0f);
                float score_ang_exit  = 1.0f - (std::min(exit.angle_to_normal_deg, 90.0f) / 90.0f);
                float score_plan_entry = std::max(0.0f, 1.0f - (entry.curvature * 20.0f)); 
                float score_plan_exit  = std::max(0.0f, 1.0f - (exit.curvature * 20.0f));
                float orient_factor_entry = (score_plan_entry > 0.3) ? 1.0f : 0.5f;
                float orient_factor_exit = (score_plan_exit > 0.3) ? 1.0f : 0.5f;
                float score_sym_entry = entry.symmetry_score;
                float score_sym_exit  = exit.symmetry_score;

                double total = (score_ang_entry * weight_orientation_ * orient_factor_entry + score_sym_entry * weight_symmetry_ + score_plan_entry * weight_planarity_) * 0.5 + 
                            (score_ang_exit * weight_orientation_ * orient_factor_exit + score_sym_exit * weight_symmetry_ + score_plan_exit * weight_planarity_) * 0.5;

                if (real_thickness < 0.015) total *= 0.1;

                ScoredGrasp sg;
                sg.pose_center.position.x = center_grasp.x(); sg.pose_center.position.y = center_grasp.y(); sg.pose_center.position.z = center_grasp.z();
                sg.pose_center.orientation.x = best_orientation.x(); sg.pose_center.orientation.y = best_orientation.y(); sg.pose_center.orientation.z = best_orientation.z(); sg.pose_center.orientation.w = best_orientation.w();
                sg.pose_finger1 = sg.pose_center;
                sg.pose_finger1.position.x = p_f1.x(); sg.pose_finger1.position.y = p_f1.y(); sg.pose_finger1.position.z = p_f1.z();
                sg.pose_finger2 = sg.pose_center;
                sg.pose_finger2.position.x = p_f2.x(); sg.pose_finger2.position.y = p_f2.y(); sg.pose_finger2.position.z = p_f2.z();
                sg.raw_ray_dir = ray_dir;
                sg.raw_p_f1 = p_f1;
                sg.raw_p_f2 = p_f2;
                sg.total_score = total;
                sg.entry_angle = entry.angle_to_normal_deg;
                sg.exit_angle = exit.angle_to_normal_deg;
                sg.entry_planarity = 1.0 - entry.curvature;
                sg.exit_planarity = 1.0 - exit.curvature;
                sg.entry_normal = entry.normal_vector;
                sg.debug_entry_pt = ray_origin + ray_dir * t_min;
                sg.debug_exit_pt = ray_origin + ray_dir * t_max;
                sg.debug_inliers = current_inliers;
                
                my_scored_grasps.push_back(sg);
            }
        }
    );

    std::vector<ScoredGrasp> initial_candidates; 
    for (const auto& local_vec : local_best_grasps) {
        initial_candidates.insert(initial_candidates.end(), local_vec.begin(), local_vec.end());
    }
    for (const auto& local_hits : local_hit_candidates) {
        hit_candidates_.insert(hit_candidates_.end(), local_hits.begin(), local_hits.end());
    }

    if (initial_candidates.empty()) 
    {
        has_best_ = false; 
        return geometry_msgs::msg::PoseArray();
    }

    std::sort(initial_candidates.begin(), initial_candidates.end(), 
        [](const ScoredGrasp& a, const ScoredGrasp& b) { return a.total_score > b.total_score; });

    best_grasps_.clear();
    
    
    for (const auto& sg : initial_candidates) 
    {
        if (best_grasps_.size() >= (size_t)num_best_grasps_) break;
        if (check_collision(sg, env_kdtree))
        {
            best_grasps_.push_back(sg);
        }
    }
    has_best_ = !best_grasps_.empty();
    
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.frame_id = "world"; 
    pose_array.header.stamp = this->now(); 

    for(int i = 0; i < num_best_grasps_ && i < (int)best_grasps_.size(); i++)
    {
        pose_array.poses.push_back(best_grasps_[i].pose_center);
    }
    return pose_array;
}

void GenerateGraspPoses::timerCallback() 
{
    publishGripperModel();
    publishGripperCollisionBoxes();

    auto t = this->now();
    sensor_msgs::msg::PointCloud2 m; 
    pcl::toROSMsg(*stored_cloud_, m); 
    m.header.stamp = t; m.header.frame_id = "world"; 
    pub_cloud_->publish(m);
    
    visualization_msgs::msg::Marker bbox_marker;
    bbox_marker.header.frame_id = "world"; bbox_marker.header.stamp = t;
    bbox_marker.ns = "bbox"; bbox_marker.id = 0;
    bbox_marker.type = visualization_msgs::msg::Marker::CUBE; bbox_marker.action = 0;
    bbox_marker.pose.position.x = (min_pt_[0] + max_pt_[0]) / 2.0;
    bbox_marker.pose.position.y = (min_pt_[1] + max_pt_[1]) / 2.0;
    bbox_marker.pose.position.z = (min_pt_[2] + max_pt_[2]) / 2.0;
    bbox_marker.pose.orientation.w = 1.0;
    bbox_marker.scale.x = max_pt_[0] - min_pt_[0]; bbox_marker.scale.y = max_pt_[1] - min_pt_[1]; bbox_marker.scale.z = max_pt_[2] - min_pt_[2];
    bbox_marker.color.r = 0.8; bbox_marker.color.g = 0.8; bbox_marker.color.b = 0.8; bbox_marker.color.a = 0.2; 
    pub_bbox_->publish(bbox_marker);

    visualization_msgs::msg::MarkerArray ma_rays; 
    float ray_len = 0.15; 
    size_t hit_lim = std::min((size_t)500, hit_candidates_.size());
    for(size_t i=0; i<hit_lim; ++i) {
        visualization_msgs::msg::Marker k; 
        k.header.frame_id="world"; k.header.stamp=t; k.ns="rays_hit"; k.id=i; k.type=0; k.action=0; 
        k.scale.x=ray_len * 0.5; k.scale.y=0.002; k.scale.z=0.002; 
        k.color.r=0.0; k.color.g=1.0; k.color.b=1.0; k.color.a=0.5; 
        k.pose = hit_candidates_[i];
        ma_rays.markers.push_back(k);
    }
    pub_rays_->publish(ma_rays);
    if(has_best_) publishBest();
}

void GenerateGraspPoses::publishBest() 
{
    visualization_msgs::msg::MarkerArray ma; 
    geometry_msgs::msg::PoseArray pose_array_msg;
    auto t = this->now();
    
    pose_array_msg.header.frame_id = "world";
    pose_array_msg.header.stamp = t;

    
    Eigen::Affine3f tf_geometry_to_tcp = Eigen::Affine3f::Identity();
    
    Eigen::Matrix3f rot_geom;
    rot_geom = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
             * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ());
    
    tf_geometry_to_tcp.linear() = rot_geom;
    tf_geometry_to_tcp.translation() = Eigen::Vector3f(mesh_offset_x_, mesh_offset_y_, mesh_offset_z_);

    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    bool has_dense_model = (gripper_dense_cloud_ && !gripper_dense_cloud_->empty());

    
    for(size_t i = 0; i < best_grasps_.size(); i++)
    {
        const auto& grasp = best_grasps_[i];
        pose_array_msg.poses.push_back(grasp.pose_center);
        
        
        Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
        Eigen::Quaternionf grasp_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);
        Eigen::Affine3f tf_tcp_to_world = Eigen::Translation3f(grasp_pos) * grasp_rot;


        Eigen::Affine3f tf_final = tf_tcp_to_world * tf_geometry_to_tcp;
        
        
        Eigen::Quaternionf q_final(tf_final.rotation());

        
        for(size_t b = 0; b < gripper_boxes_.size(); b++)
        {
            const auto& box = gripper_boxes_[b];
            
            visualization_msgs::msg::Marker mk;
            mk.header.frame_id = "world";
            mk.header.stamp = t;
            
            mk.ns = "debug_boxes_grasp_" + std::to_string(i); 
            mk.id = b;
            mk.type = visualization_msgs::msg::Marker::CUBE;
            mk.action = visualization_msgs::msg::Marker::ADD;

            
            Eigen::Vector3f center_world = tf_final * box.center;
            
            mk.pose.position.x = center_world.x();
            mk.pose.position.y = center_world.y();
            mk.pose.position.z = center_world.z();

            
            mk.pose.orientation.x = q_final.x();
            mk.pose.orientation.y = q_final.y();
            mk.pose.orientation.z = q_final.z();
            mk.pose.orientation.w = q_final.w();

            
            mk.scale.x = box.dimensions.x();
            mk.scale.y = box.dimensions.y();
            mk.scale.z = box.dimensions.z();

            
            if (i == 0) { mk.color.r = 1.0; mk.color.g = 0.0; mk.color.b = 0.0; }
            else        { mk.color.r = 0.0; mk.color.g = 0.0; mk.color.b = 1.0; }
            mk.color.a = 0.3; 

            ma.markers.push_back(mk);
        }

        
        if (has_dense_model)
        {
            for (const auto& pt_local : gripper_dense_cloud_->points)
            {
                
                Eigen::Vector3f p_world = tf_final * Eigen::Vector3f(pt_local.x, pt_local.y, pt_local.z);
                
                pcl::PointXYZRGB p_colored = pt_local; 
                p_colored.x = p_world.x();
                p_colored.y = p_world.y();
                p_colored.z = p_world.z();
                accumulated_cloud->points.push_back(p_colored);
            }
        }
        
        
        int base_id = i * 1000; 
        visualization_msgs::msg::Marker txt; 
        txt.header.frame_id="world"; txt.header.stamp=t; txt.ns="txt"; txt.id=base_id; txt.type=9; txt.action=0; 
        txt.pose=grasp.pose_center; txt.pose.position.z+=0.05; txt.scale.z=0.03; 
        txt.color.r=1; txt.color.g=1; txt.color.b=1; txt.color.a=1.0;
        char buf[128]; 
        if (i==0) sprintf(buf, "TOP 1\nS:%.2f", grasp.total_score);
        else sprintf(buf, "#%lu", i+1);
        txt.text=buf; 
        ma.markers.push_back(txt);
    }
    
    
    if (!accumulated_cloud->empty()) 
    {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*accumulated_cloud, cloud_msg);
        cloud_msg.header.frame_id = "world";
        cloud_msg.header.stamp = t;
        pub_debug_grasps_cloud_->publish(cloud_msg);
    }

    pub_markers_->publish(ma);
    
    pub_poses_->publish(pose_array_msg);
}

} // namespace vision

RCLCPP_COMPONENTS_REGISTER_NODE(vision::GenerateGraspPoses)