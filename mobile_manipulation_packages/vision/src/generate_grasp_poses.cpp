#include "vision/GenerateGraspPoses.hpp" 

#include <rclcpp_components/register_node_macro.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <sys/resource.h>

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

using namespace std::chrono_literals;

namespace vision 
{

GenerateGraspPoses::GenerateGraspPoses(const rclcpp::NodeOptions & options) 
: Node("generate_grasp_poses", options) 
{
    this->declare_parameter<bool>("use_pcd_file", false);
    this->declare_parameter<std::string>("pcd_path", "/home/momesso/pibic/nuvem.pcd");
    
    this->declare_parameter<std::string>("gripper_mesh_path", "/home/momesso/hand_and_finger.stl");
    this->declare_parameter<double>("gripper_mesh_scale", 1.0);
    
    this->declare_parameter<double>("mesh_offset_x", 0.025);
    this->declare_parameter<double>("mesh_offset_y", 0.0);
    this->declare_parameter<double>("mesh_offset_z", 0.0);
    
    this->declare_parameter<double>("mesh_rot_roll", 0.0);
    this->declare_parameter<double>("mesh_rot_pitch", -1.57);
    this->declare_parameter<double>("mesh_rot_yaw", 0.0); 

    this->declare_parameter<double>("grid_res", 0.0075);
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

    this->declare_parameter<int>("num_best_grasps", 2);
    this->declare_parameter<double>("rotation_step_deg", 30.0);

    this->declare_parameter<int>("num_random_orientations", 1);

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
    pub_gripper_model_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "debug_gripper_model", 10);
    pub_debug_grasps_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_grasps_cloud", 10);

    stored_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    loadGripperAsOctree();
    publishGripperModel();
    RCLCPP_INFO(this->get_logger(), "MODO ARQUIVO: Carregando PCD de %s...", pcd_path_.c_str());

    if(use_pcd_file == false)
    {
        loadAndProcess(pcd_path_);
    }
   
    
    timer_ = this->create_wall_timer(1000ms, std::bind(&GenerateGraspPoses::timerCallback, this));
}


void GenerateGraspPoses::loadAndProcess(const std::string& path)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *temp_cloud) == -1) 
    {
        RCLCPP_ERROR(this->get_logger(), "Falha ao ler arquivo PCD: %s", path.c_str());
        return;
    }
    processCloud(temp_cloud, cloud);
}

geometry_msgs::msg::PoseArray GenerateGraspPoses::processCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr target, pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment)
{
    auto start = std::chrono::high_resolution_clock::now();
    if (target->empty()) 
    {
        return geometry_msgs::msg::PoseArray();
    }

    if (use_pcd_file == false)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI); 

        float rot_x = dis(gen);
        float rot_y = dis(gen);
        float rot_z = dis(gen);

       
        // RCLCPP_INFO(this->get_logger(), "Modo Offline: Aplicando Rotação Aleatória [R:%.2f, P:%.2f, Y:%.2f]", rot_x, rot_y, rot_z);

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
    
    if (cloud_voxel_size_ > 0.0001) 
    {
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(target);
        sor.setLeafSize(cloud_voxel_size_, cloud_voxel_size_, cloud_voxel_size_);
        sor.filter(*voxel_cloud);
    } 
    else 
    {
        *voxel_cloud = *target;
    }

    if (mean_filter) 
    {
        // Setup Serial (Rápido)
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(voxel_cloud);

        stored_cloud_->points.resize(voxel_cloud->points.size());
        stored_cloud_->width = voxel_cloud->width;
        stored_cloud_->height = voxel_cloud->height;
        stored_cloud_->is_dense = voxel_cloud->is_dense;
        stored_cloud_->header = voxel_cloud->header; 

        int K = mean_filter_k_; 

        tbb::parallel_for(tbb::blocked_range<size_t>(0, voxel_cloud->points.size()),
            [&](const tbb::blocked_range<size_t>& range) 
            {
                std::vector<int> pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);

                for (size_t i = range.begin(); i != range.end(); ++i) 
                {
                    if (kdtree.nearestKSearch(voxel_cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) 
                    {
                        float sum_x = 0, sum_y = 0, sum_z = 0;
                        int valid_pts = 0;

                        for (int j = 0; j < K; ++j) 
                        {
                            const auto& neighbor = voxel_cloud->points[pointIdxNKNSearch[j]];
                            sum_x += neighbor.x;
                            sum_y += neighbor.y;
                            sum_z += neighbor.z;
                            valid_pts++;
                        }

                        stored_cloud_->points[i].x = sum_x / valid_pts;
                        stored_cloud_->points[i].y = sum_y / valid_pts;
                        stored_cloud_->points[i].z = sum_z / valid_pts;
                    }
                    else
                    {
                        stored_cloud_->points[i] = voxel_cloud->points[i];
                    }
                }
            }
        );
    }
    else 
    {
        *stored_cloud_ = *voxel_cloud;
    }
    
    stored_cloud_->header.frame_id = "world";

    pcl::getMinMax3D(*stored_cloud_, min_pt_, max_pt_);
    
    float padding = 0.03; 
    min_pt_.array() -= padding; max_pt_.array() += padding;
    
    

    if (total_orientations_ < 1) 
    {
        total_orientations_ = 1;
    }
    
    std::vector<Eigen::Matrix3f> search_matrices;
    search_matrices.reserve(total_orientations_);

    
    search_matrices.push_back(Eigen::Matrix3f::Identity());

    
    if (total_orientations_ > 1) 
    {
        std::random_device rd;  
        std::mt19937 gen(rd()); 
        
        
        std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI);

        for (int i = 0; i < total_orientations_ - 1; ++i) 
        {
            float roll  = dis(gen);
            float pitch = dis(gen);
            float yaw   = dis(gen);

            
            Eigen::Matrix3f m;
            m = Eigen::AngleAxisf(roll,  Eigen::Vector3f::UnitX())
                * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
                * Eigen::AngleAxisf(yaw,   Eigen::Vector3f::UnitZ());
            
            search_matrices.push_back(m);
        }
    }
    
    // RCLCPP_INFO(this->get_logger(), "Gerando raios para %d orientações 3D distintas.", total_orientations_);
    
    
    all_candidates_ = generateMultiOrientedRays(min_pt_, max_pt_, grid_res_, search_matrices);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // RCLCPP_INFO(this->get_logger(), "Tempo Total ProcessCloud: %ld ms", duration);
    
    geometry_msgs::msg::PoseArray best_grasps = evaluateGrasps(target_environment);

    return best_grasps;
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
            for(float z = -half_z; z < half_z; z += res) {
                add_rotated_ray({half_x, y + res/2, z + res/2}, {-1, 0, 0}, R);
            }

        
        for(float x = -half_x; x < half_x; x += res)
            for(float z = -half_z; z < half_z; z += res) {
                add_rotated_ray({x + res/2, half_y, z + res/2}, {0, -1, 0}, R);
            }

        
        for(float x = -half_x; x < half_x; x += res)
            for(float y = -half_y; y < half_y; y += res) {
                add_rotated_ray({x + res/2, y + res/2, half_z}, {0, 0, -1}, R);
            }
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
    StepAnalysis result;
    result.valid = false;
    result.center = center;
    
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

void GenerateGraspPoses::publishGripperModel()
{
    
    if (!gripper_dense_cloud_ || gripper_dense_cloud_->empty()) {
        RCLCPP_WARN(this->get_logger(), "Tentou publicar modelo da garra, mas está vazio!");
        return;
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*gripper_dense_cloud_, msg);

    
    msg.header.frame_id = "world"; 
    msg.header.stamp = this->now();

    pub_gripper_model_->publish(msg);
    
    // RCLCPP_INFO(this->get_logger(), "Modelo denso da garra publicado em '/debug_gripper_model' (%lu pontos)", gripper_dense_cloud_->size());
}

void GenerateGraspPoses::loadGripperAsOctree()
{
    RCLCPP_INFO(this->get_logger(), "Carregando STL, Densificando e APLICANDO OFFSETS: %s", gripper_mesh_path_.c_str());
    
    
    pcl::PolygonMesh mesh;
    if (pcl::io::loadPolygonFileSTL(gripper_mesh_path_, mesh) == 0) {
        RCLCPP_ERROR(this->get_logger(), "ERRO CRÍTICO: STL não encontrado!");
        return;
    }

    pcl::PointCloud<pcl::PointXYZ> mesh_verts;
    pcl::fromPCLPointCloud2(mesh.cloud, mesh_verts);

    
    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_dense_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    
    float density_dist = 0.002f; 
    
    for (const auto& poly : mesh.polygons) 
    {
        if (poly.vertices.size() < 3) continue;

        Eigen::Vector3f v0 = mesh_verts[poly.vertices[0]].getVector3fMap();
        Eigen::Vector3f v1 = mesh_verts[poly.vertices[1]].getVector3fMap();
        Eigen::Vector3f v2 = mesh_verts[poly.vertices[2]].getVector3fMap();

        float area = 0.5f * ((v1 - v0).cross(v2 - v0)).norm();
        int num_points = std::max(1, (int)(area / (density_dist * density_dist)));
        
        
        raw_dense_cloud->points.push_back(pcl::PointXYZ(v0.x(), v0.y(), v0.z()));
        raw_dense_cloud->points.push_back(pcl::PointXYZ(v1.x(), v1.y(), v1.z()));
        raw_dense_cloud->points.push_back(pcl::PointXYZ(v2.x(), v2.y(), v2.z()));

        
        for (int k = 0; k < num_points; ++k) 
        {
            float r1 = (float)rand() / RAND_MAX;
            float r2 = (float)rand() / RAND_MAX;
            float sq_r1 = std::sqrt(r1);
            Eigen::Vector3f p = (1 - sq_r1) * v0 + (sq_r1 * (1 - r2)) * v1 + (sq_r1 * r2) * v2;
            raw_dense_cloud->points.push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
        }
    }

    
    gripper_dense_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    
    transform.translation() = Eigen::Vector3f(mesh_offset_x_, mesh_offset_y_, mesh_offset_z_);
    
    
    Eigen::Matrix3f rot;
    rot = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ());
    transform.linear() = rot;

    
    transform.scale(gripper_mesh_scale_);

    
    pcl::transformPointCloud(*raw_dense_cloud, *gripper_dense_cloud_, transform);

    
    gripper_octree_.deleteTree();
    gripper_octree_.setInputCloud(gripper_dense_cloud_);
    gripper_octree_.addPointsFromInputCloud();

    RCLCPP_INFO(this->get_logger(), "STL Processado: %lu pontos. Offsets aplicados na memória.", gripper_dense_cloud_->size());
}

bool GenerateGraspPoses::check_collision(const ScoredGrasp& grasp, pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment)
{
    if (!gripper_dense_cloud_ || gripper_dense_cloud_->empty()) return true;

    Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
    Eigen::Quaternionf grasp_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);
    
    
    Eigen::Affine3f tf_world_to_grasp = (Eigen::Translation3f(grasp_pos) * grasp_rot).inverse();

    
    float range_limit_sq = 0.15f * 0.15f; 

    
    for (const auto& pt_world : target_environment->points) 
    {
        
        Eigen::Vector3f p_vec(pt_world.x, pt_world.y, pt_world.z);
        if ((p_vec - grasp_pos).squaredNorm() > range_limit_sq) continue;

        
        Eigen::Vector3f p_local = tf_world_to_grasp * p_vec;
        
        pcl::PointXYZ search_point;
        search_point.x = p_local.x();
        search_point.y = p_local.y();
        search_point.z = p_local.z();

        
        if (gripper_octree_.isVoxelOccupiedAtPoint(search_point)) 
        {
            return false;
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

    // RCLCPP_INFO(this->get_logger(), "Fase 1: Avaliando %lu raios...", all_candidates_.size());

    auto start_total = std::chrono::high_resolution_clock::now();

    
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

    
    struct ThreadTimers {
        double inliers_ms = 0.0;
        double analysis_ms = 0.0;
        double scoring_ms = 0.0;
    };

    tbb::enumerable_thread_specific<std::vector<ScoredGrasp>> local_best_grasps;
    tbb::enumerable_thread_specific<std::vector<geometry_msgs::msg::Pose>> local_hit_candidates;
    tbb::enumerable_thread_specific<ThreadTimers> local_timers; 

    
    tbb::parallel_for(tbb::blocked_range<size_t>(0, all_candidates_.size()),
        [&](const tbb::blocked_range<size_t>& range)
        {
            
            auto& my_scored_grasps = local_best_grasps.local();
            auto& my_hit_candidates = local_hit_candidates.local();
            auto& my_timer = local_timers.local(); 

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

                
                auto t0 = std::chrono::high_resolution_clock::now();

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
                
                
                auto t1 = std::chrono::high_resolution_clock::now();
                my_timer.inliers_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

                if (!hit || (t_max - t_min) < 0.005) continue;
                
                my_hit_candidates.push_back(raw_pose);

                std::vector<StepAnalysis> steps;

                
                auto t2 = std::chrono::high_resolution_clock::now();

                for (float t = t_min; t <= t_max; t += analysis_step_size_) 
                {
                    Eigen::Vector3f center = ray_origin + ray_dir * t;
                    StepAnalysis res = analyzeLocalCylinder(current_inliers_ptr, center, ray_dir, cylinder_radius_, cylinder_height_);
                    if (res.valid) steps.push_back(res);
                }
                
                
                auto t3 = std::chrono::high_resolution_clock::now();
                my_timer.analysis_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

                if (steps.empty()) continue;

                
                auto t4 = std::chrono::high_resolution_clock::now();

                StepAnalysis& entry = steps.front();
                StepAnalysis& exit = steps.back();

                float real_thickness = t_max - t_min;
                if (real_thickness > max_gripper_width_) continue; 

                float current_offset = finger_offset_;
                float total_width_needed = real_thickness + (2.0f * current_offset);

                if (total_width_needed > max_gripper_width_) 
                {
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

                
                auto t5 = std::chrono::high_resolution_clock::now();
                my_timer.scoring_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();
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

    
    double max_inliers_ms = 0.0;
    double max_analysis_ms = 0.0;
    double max_scoring_ms = 0.0;

    for (const auto& t : local_timers) {
        if (t.inliers_ms > max_inliers_ms) max_inliers_ms = t.inliers_ms;
        if (t.analysis_ms > max_analysis_ms) max_analysis_ms = t.analysis_ms;
        if (t.scoring_ms > max_scoring_ms) max_scoring_ms = t.scoring_ms;
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
        if (check_collision(sg, target_environment))
        {
            best_grasps_.push_back(sg);
        }
    }
    auto end_total = std::chrono::high_resolution_clock::now();

    has_best_ = !best_grasps_.empty();
    // RCLCPP_INFO(this->get_logger(), "Encontrados %lu grasps.", best_grasps_.size());
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();

    // RCLCPP_INFO(this->get_logger(), "Tempo Total EvaluateGrasps: %ld ms", duration);
    
    
    // RCLCPP_INFO(this->get_logger(), ">> Breakdown (Max Thread): Inliers: %.2f ms | Analysis: %.2f ms | Scoring: %.2f ms", 
    //     max_inliers_ms, max_analysis_ms, max_scoring_ms);
        
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long max_mem_mb = usage.ru_maxrss / 1024; 
    // RCLCPP_INFO(this->get_logger(), "Memória Máxima Usada (RSS): %ld MB", max_mem_mb);

    geometry_msgs::msg::PoseArray pose_array;

    pose_array.header.frame_id = "world"; 
    pose_array.header.stamp = this->now(); 

    
    for(int i = 0; i < num_best_grasps_; i++)
    {
    
        pose_array.poses.push_back(best_grasps_[i].pose_center);
    }

    return pose_array;
}

void GenerateGraspPoses::timerCallback() 
{
    publishGripperModel();
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


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    
    bool has_dense_model = (gripper_dense_cloud_ && !gripper_dense_cloud_->empty());

    for(size_t i = 0; i < best_grasps_.size(); i++)
    {
        const auto& grasp = best_grasps_[i];
        pose_array_msg.poses.push_back(grasp.pose_center);

        
        uint8_t r=0, g=0, b=0;
        if (i == 0) { r=0; g=255; b=0; }       
        else        { r=100; g=100; b=255; }   

        
        if (has_dense_model)
        {
            
            Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
            Eigen::Quaternionf grasp_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);
            Eigen::Affine3f tf_grasp = Eigen::Translation3f(grasp_pos) * grasp_rot;

            
            for (const auto& pt_local : gripper_dense_cloud_->points)
            {
                
                Eigen::Vector3f p_world = tf_grasp * Eigen::Vector3f(pt_local.x, pt_local.y, pt_local.z);
                
                pcl::PointXYZRGB p_colored;
                p_colored.x = p_world.x();
                p_colored.y = p_world.y();
                p_colored.z = p_world.z();
                p_colored.r = r;
                p_colored.g = g;
                p_colored.b = b;
                
                accumulated_cloud->points.push_back(p_colored);
            }
        }

        
        int base_id = i * 30; 
        
        
        visualization_msgs::msg::Marker txt; 
        txt.header.frame_id="world"; txt.header.stamp=t; txt.ns="txt"; txt.id=base_id+3; txt.type=9; txt.action=0; 
        txt.pose=grasp.pose_center; txt.pose.position.z+=0.05; txt.scale.z=0.03; 
        txt.color.r=1; txt.color.g=1; txt.color.b=1; txt.color.a=1.0;
        char buf[128]; 
        if (i==0) sprintf(buf, "TOP 1\nS:%.2f", grasp.total_score);
        else sprintf(buf, "#%lu", i+1);
        txt.text=buf; 
        ma.markers.push_back(txt);
    }
    
    
    if (!accumulated_cloud->empty()) {
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