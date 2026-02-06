#include "vision/GenerateGraspPoses.hpp" 

#include <rclcpp_components/register_node_macro.hpp>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <sys/resource.h>
#include <limits> 
#include <x86intrin.h>
#include <atomic>
#include <mutex>
#include <numeric>

// Mensagens para Benchmark
#include <std_msgs/msg/float64.hpp>

// TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/blocked_range.h>

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

struct GlobalBenchStats {
    double total_func;
    double loop_tbb;
    double max_inliers;
    double max_analysis;
    double max_scoring;
    double sort;
    double collision;
    int checks;
} g_last_run_stats;

struct BenchmarkPublishers {
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_total;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_loop;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_inliers;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_analysis;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_scoring;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_sort;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_collision;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr n_checks;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr score_max;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr score_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr mem_usage;
};
static std::unique_ptr<BenchmarkPublishers> g_bench_pubs;

double getMemoryUsageMB() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) 
    {
        
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }
    return 0.0;
}

GenerateGraspPoses::GenerateGraspPoses(const rclcpp::NodeOptions & options) 
: Node("generate_grasp_poses", options) 
{
    this->declare_parameter<bool>("use_pcd_file", false);
    this->declare_parameter<std::string>("pcd_path", "/home/momesso/pibic/nuvem.pcd");
    
    this->declare_parameter<std::string>("object_mesh_path", "/home/momesso/pibic/objeto.obj");
    this->declare_parameter<bool>("publish_object_mesh", false);

    this->declare_parameter<std::string>("gripper_mesh_path", "/home/momesso/hand_and_fingers.obj");
    this->declare_parameter<double>("gripper_mesh_scale", 1.0);
    
    this->declare_parameter<std::string>("gripper_glb_path", "/home/momesso/hand_and_fingers.glb");
    this->declare_parameter<bool>("publish_gripper_mesh", false);
    
    this->declare_parameter<double>("mesh_offset_x", 0.025);
    this->declare_parameter<double>("mesh_offset_y", 0.0);
    this->declare_parameter<double>("mesh_offset_z", 0.0);
    
    this->declare_parameter<double>("mesh_rot_roll", 1.57);
    this->declare_parameter<double>("mesh_rot_pitch", 0.0);
    this->declare_parameter<double>("mesh_rot_yaw", 1.57); 

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
    this->declare_parameter<double>("target_score", 0.95);
    
    this->declare_parameter<bool>("use_mean_filter", true); 
    this->declare_parameter<int>("mean_filter_k", 15);

    this->declare_parameter<int>("num_best_grasps", 1);
    this->declare_parameter<double>("rotation_step_deg", 20.0);

    this->declare_parameter<int>("num_random_orientations", 20);

    this->declare_parameter<int>("num_benchmark_runs", 10);
    this->declare_parameter<bool>("enable_ray_animation", true);
    this->declare_parameter<int>("animation_delay_ms", 20);

    use_pcd_file = this->get_parameter("use_pcd_file").as_bool();
    pcd_path_ = this->get_parameter("pcd_path").as_string();
    
    object_mesh_path_ = this->get_parameter("object_mesh_path").as_string();
    publish_object_mesh_ = this->get_parameter("publish_object_mesh").as_bool();

    gripper_mesh_path_ = this->get_parameter("gripper_mesh_path").as_string();
    gripper_mesh_scale_ = static_cast<float>(this->get_parameter("gripper_mesh_scale").as_double());

    gripper_glb_path_ = this->get_parameter("gripper_glb_path").as_string();
    publish_gripper_mesh_ = this->get_parameter("publish_gripper_mesh").as_bool();

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

    target_score_ = static_cast<float>(this->get_parameter("target_score").as_double());

    mean_filter = this->get_parameter("use_mean_filter").as_bool();
    mean_filter_k_ = this->get_parameter("mean_filter_k").as_int();

    num_best_grasps_ = this->get_parameter("num_best_grasps").as_int();
    total_orientations_ = this->get_parameter("num_random_orientations").as_int();
    rotation_step_deg_ = static_cast<float>(this->get_parameter("rotation_step_deg").as_double());
    
    num_benchmark_runs_ = this->get_parameter("num_benchmark_runs").as_int();
    enable_ray_animation_ = this->get_parameter("enable_ray_animation").as_bool();
    animation_delay_ms_ = this->get_parameter("animation_delay_ms").as_int();

    

    rclcpp::QoS qos_profile(10);
    qos_profile.transient_local(); 
    qos_profile.reliable();

    pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("input_cloud", qos_profile);

    pub_rays_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("candidate_rays", qos_profile);
    pub_bbox_  = this->create_publisher<visualization_msgs::msg::Marker>("bounding_box", 10);
    pub_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("best_grasps_markers", 10);
    pub_poses_ = this->create_publisher<geometry_msgs::msg::PoseArray>("best_grasps_poses", 10);
    pub_debug_inliers_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_ray_inliers", 10);
    pub_debug_collision_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_collision_check", 10);
    pub_gripper_model_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_gripper_model", 10);
    pub_gripper_boxes_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("debug_gripper_boxes", 10);
    pub_debug_grasps_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_grasps_cloud", 10);
    debug_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/grasp_debug_rays", 10);
    
    pub_object_mesh_ = this->create_publisher<visualization_msgs::msg::Marker>("debug_object_mesh", qos_profile);

    g_bench_pubs = std::make_unique<BenchmarkPublishers>();
    g_bench_pubs->t_total = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/total_ms", 10);
    g_bench_pubs->t_loop = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/loop_serial_ms", 10);
    
    g_bench_pubs->t_inliers = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/parts/inliers_ms", 10);
    g_bench_pubs->t_analysis = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/parts/analysis_ms", 10);
    g_bench_pubs->t_scoring = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/parts/scoring_ms", 10);
    g_bench_pubs->t_sort = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/parts/sort_ms", 10);
    g_bench_pubs->t_collision = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/parts/collision_ms", 10);
    g_bench_pubs->n_checks = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/stats/collision_checks", 10);
    
    g_bench_pubs->score_max = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/score/max", 10);
    g_bench_pubs->score_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/score/avg", 10);
    
    g_bench_pubs->mem_usage = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/system/memory_mb", 10);

    stored_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    gripper_dense_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>); 
    
    collision_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    extractBoundingBoxesFromOBJ(); 
    
    
    publishGripperModel();
    publishGripperCollisionBoxes();

    RCLCPP_INFO(this->get_logger(), "MODO ARQUIVO: Carregando PCD de %s...", pcd_path_.c_str());

    if(use_pcd_file == true)
    {
        loadAndProcess(pcd_path_);
    }
    
    
    timer_ = this->create_wall_timer(10ms, std::bind(&GenerateGraspPoses::timerCallback, this));
}


void GenerateGraspPoses::loadAndProcess(const std::string& path)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *temp_cloud) == -1) 
    {
        RCLCPP_ERROR(this->get_logger(), "Falha ao ler arquivo PCD: %s", path.c_str());
        return;
    }
    
    GlobalBenchStats acc_stats = {0,0,0,0,0,0,0,0};
    int runs = std::max(1, num_benchmark_runs_);

    RCLCPP_INFO(this->get_logger(), ">>> INICIANDO BENCHMARK LOOP: %d execuções <<<", runs);


    for(int i = 0; i < runs; ++i)
    {
        hit_candidates_.clear();
        best_grasps_.clear();
        
        
        processCloud(temp_cloud, temp_cloud);

        
        auto msg_f64 = [](double val){ std_msgs::msg::Float64 m; m.data = val; return m; };
        
        
        g_bench_pubs->t_total->publish(msg_f64(g_last_run_stats.total_func));
        g_bench_pubs->t_loop->publish(msg_f64(g_last_run_stats.loop_tbb));
        g_bench_pubs->t_inliers->publish(msg_f64(g_last_run_stats.max_inliers));
        g_bench_pubs->t_analysis->publish(msg_f64(g_last_run_stats.max_analysis));
        g_bench_pubs->t_scoring->publish(msg_f64(g_last_run_stats.max_scoring));
        g_bench_pubs->t_sort->publish(msg_f64(g_last_run_stats.sort));
        g_bench_pubs->t_collision->publish(msg_f64(g_last_run_stats.collision));
        g_bench_pubs->n_checks->publish(msg_f64((double)g_last_run_stats.checks));

       
        g_bench_pubs->mem_usage->publish(msg_f64(getMemoryUsageMB()));

        
        double max_score = 0.0;
        double avg_score = 0.0;
        if (!best_grasps_.empty()) {
            max_score = best_grasps_[0].total_score;
            double sum = 0;
            for(const auto& bg : best_grasps_) sum += bg.total_score;
            avg_score = sum / best_grasps_.size();
        }
        g_bench_pubs->score_max->publish(msg_f64(max_score));
        g_bench_pubs->score_avg->publish(msg_f64(avg_score));
    

        acc_stats.total_func += g_last_run_stats.total_func;
        acc_stats.loop_tbb += g_last_run_stats.loop_tbb;
        acc_stats.max_inliers += g_last_run_stats.max_inliers;
        acc_stats.max_analysis += g_last_run_stats.max_analysis;
        acc_stats.max_scoring += g_last_run_stats.max_scoring;
        acc_stats.sort += g_last_run_stats.sort;
        acc_stats.collision += g_last_run_stats.collision;
        acc_stats.checks += g_last_run_stats.checks;
        
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    
    double div = static_cast<double>(runs);
    
    RCLCPP_INFO(this->get_logger(), " ");
    RCLCPP_INFO(this->get_logger(), "============================================================");
    RCLCPP_INFO(this->get_logger(), "======= RESULTADO FINAL: MÉDIAS APÓS %d EXECUÇÕES ========", runs);
    RCLCPP_INFO(this->get_logger(), "============================================================");
    RCLCPP_INFO(this->get_logger(), "Tempo Total Médio:       %.4f ms", acc_stats.total_func / div);
    RCLCPP_INFO(this->get_logger(), "  -> TBB Loop Médio:     %.4f ms", acc_stats.loop_tbb / div);
    RCLCPP_INFO(this->get_logger(), "     |-> Inliers (Max):  %.4f ms", acc_stats.max_inliers / div);
    RCLCPP_INFO(this->get_logger(), "     |-> Analysis (Max): %.4f ms", acc_stats.max_analysis / div);
    RCLCPP_INFO(this->get_logger(), "     |-> Scoring (Max):  %.4f ms", acc_stats.max_scoring / div);
    RCLCPP_INFO(this->get_logger(), "  -> Sort Médio:         %.4f ms", acc_stats.sort / div);
    RCLCPP_INFO(this->get_logger(), "  -> Collision Médio:    %.4f ms (Checks Avg: %.1f)", acc_stats.collision / div, (double)acc_stats.checks / div);
    RCLCPP_INFO(this->get_logger(), "============================================================");
}

geometry_msgs::msg::PoseArray GenerateGraspPoses::processCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr target, pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment)
{
    if (!target || target->empty()) return geometry_msgs::msg::PoseArray();
    
    if (use_pcd_file == true)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI); 
        float rot_x = dis(gen); float rot_y = dis(gen); float rot_z = dis(gen);

        
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*target, centroid);

       
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();

       
        transform.rotate(Eigen::AngleAxisf(rot_x, Eigen::Vector3f::UnitX())); 
        transform.rotate(Eigen::AngleAxisf(rot_y, Eigen::Vector3f::UnitY()));
        transform.rotate(Eigen::AngleAxisf(rot_z, Eigen::Vector3f::UnitZ())); 

        transform.translate(-centroid.head<3>());

        
        pcl::transformPointCloud(*target, *target, transform);

        
        if (publish_object_mesh_)
        {
            visualization_msgs::msg::Marker mesh_marker;
            mesh_marker.header.frame_id = "world";
            mesh_marker.header.stamp = this->now();
            mesh_marker.ns = "object_mesh_aligned";
            mesh_marker.id = 0;
            mesh_marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            mesh_marker.action = visualization_msgs::msg::Marker::ADD;
            mesh_marker.mesh_resource = "file://" + object_mesh_path_; 
            
            
            mesh_marker.scale.x = 1.0;
            mesh_marker.scale.y = 1.0;
            mesh_marker.scale.z = 1.0;
            
            
            Eigen::Vector3f t = transform.translation();
            Eigen::Quaternionf q(transform.rotation());

            mesh_marker.pose.position.x = t.x();
            mesh_marker.pose.position.y = t.y();
            mesh_marker.pose.position.z = t.z();
            
            mesh_marker.pose.orientation.x = q.x();
            mesh_marker.pose.orientation.y = q.y();
            mesh_marker.pose.orientation.z = q.z();
            mesh_marker.pose.orientation.w = q.w();

            
            mesh_marker.color.r = 0.8;
            mesh_marker.color.g = 0.8;
            mesh_marker.color.b = 0.8;
            mesh_marker.color.a = 1.0; 

            pub_object_mesh_->publish(mesh_marker);
        }
        
    }
    else
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(target);
        sor.setMeanK(120); 
        sor.setStddevMulThresh(1.5); 
        sor.filter(*target);
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (use_pcd_file && cloud_voxel_size_ > 0.001f) 
    {
        pcl::VoxelGrid<pcl::PointXYZ> sor_tgt;
        sor_tgt.setInputCloud(target);
        sor_tgt.setLeafSize(cloud_voxel_size_, cloud_voxel_size_, cloud_voxel_size_);
        sor_tgt.filter(*voxel_cloud);
    } 
    else 
    {
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
                            sum_x += neighbor.x; sum_y += neighbor.y; sum_z += neighbor.z; valid_pts++;
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

    publishBest();
    auto t = this->now();
    sensor_msgs::msg::PointCloud2 m; 
    pcl::toROSMsg(*stored_cloud_, m); 
    m.header.stamp = t; 
    m.header.frame_id = "world"; 
    pub_cloud_->publish(m);
    
    all_candidates_ = generateMultiOrientedRays(min_pt_, max_pt_, grid_res_);
    
    std::random_device rd;
    std::mt19937 g(rd());
 
    std::shuffle(all_candidates_.begin(), all_candidates_.end(), g);

    if (target_environment && !target_environment->empty()) 
    {
        
        if (!collision_cloud_) collision_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);

        
        pcl::VoxelGrid<pcl::PointXYZ> sor_env;
        sor_env.setInputCloud(target_environment);
        sor_env.setLeafSize(0.005f, 0.005f, 0.005f); 
        sor_env.filter(*collision_cloud_);

        
        if (collision_cloud_->empty()) 
        {
            *collision_cloud_ = *target_environment;
        }
        collision_kdtree_.setInputCloud(collision_cloud_);
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Ambiente de colisão vazio ou nulo!");
    }
    // ------------------------------------

    return evaluateGrasps(target_environment);
}

std::vector<geometry_msgs::msg::Pose> GenerateGraspPoses::generateMultiOrientedRays(
    const Eigen::Vector4f& min, const Eigen::Vector4f& max, float res) 
{
    std::vector<geometry_msgs::msg::Pose> poses;
    
    Eigen::Vector3f center = (min.head<3>() + max.head<3>()) / 2.0f;
    // 'size' contém a largura total, altura total e profundidade total
    Eigen::Vector3f size = max.head<3>() - min.head<3>();
    
    // Lambda para adicionar raio e seu comprimento específico
    auto add_ray = [&](Eigen::Vector3f local_pos, Eigen::Vector3f direction, float length) 
    {
        geometry_msgs::msg::Pose p;
        p.position.x = local_pos.x() + center.x(); 
        p.position.y = local_pos.y() + center.y(); 
        p.position.z = local_pos.z() + center.z();
        
        Eigen::Quaternionf q; 
        q.setFromTwoVectors(Eigen::Vector3f::UnitX(), direction); 
        p.orientation.x = q.x(); p.orientation.y = q.y(); 
        p.orientation.z = q.z(); p.orientation.w = q.w();
        
        poses.push_back(p);
        ray_lengths.push_back(length); 
    };

    float half_x = size.x() / 2.0f;
    float half_y = size.y() / 2.0f;
    float half_z = size.z() / 2.0f;

    // Face X: Varre Y e Z. O raio viaja em X, então o comprimento é size.x() (Total)
    for(float y = -half_y; y < half_y; y += res)
        for(float z = -half_z; z < half_z; z += res) {
            add_ray({half_x, y, z}, {-1, 0, 0}, size.x());  
            // Se descomentar a linha abaixo, use size.x() também
            add_ray({-half_x, y, z}, {1, 0, 0}, size.x());  
        }

    // Face Y: Varre X e Z. O raio viaja em Y, então o comprimento é size.y() (Total)
    for(float x = -half_x; x < half_x; x += res)
        for(float z = -half_z; z < half_z; z += res) {
            add_ray({x, half_y, z}, {0, -1, 0}, size.y());
            add_ray({x, -half_y, z}, {0, 1, 0}, size.y());
        }

    // Face Z: Varre X e Y. O raio viaja em Z, então o comprimento é size.z() (Total)
    for(float x = -half_x; x < half_x; x += res)
        for(float y = -half_y; y < half_y; y += res) {
            add_ray({x, y, half_z}, {0, 0, -1}, size.z());
            add_ray({x, y, -half_z}, {0, 0, 1}, size.z());
        }

   
    if (pub_rays_) { 
        visualization_msgs::msg::MarkerArray marker_array;

        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        for (size_t i = 0; i < poses.size(); ++i) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "world"; 
            marker.header.stamp = this->now();
            marker.ns = "generated_rays";
            marker.id = i;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose = poses[i];
            
            
            marker.scale.x = ray_lengths[i]; 
            
            
            marker.scale.y = 0.005; 
            marker.scale.z = 0.005; 

            marker.color.r = 1.0;
            marker.color.a = 1.0; 

            marker_array.markers.push_back(marker);
        }
        
        
        pub_rays_->publish(marker_array);
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

        if (i == 0)       { m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; }
        else if (i == 1) { m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; }
        else             { m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0; }
        m.color.a = 0.4; 

        ma.markers.push_back(m);
    }
    pub_gripper_boxes_->publish(ma);
}


bool GenerateGraspPoses::check_collision(ScoredGrasp& grasp, const pcl::KdTreeFLANN<pcl::PointXYZ>& env_kdtree, bool publish_debug, bool try_rotations)
{
    if (gripper_boxes_.empty()) return true;
    if (!env_kdtree.getInputCloud() || env_kdtree.getInputCloud()->empty()) return true;

    static bool tf_initialized = false;
    static Eigen::Affine3f tf_tcp_to_mesh = Eigen::Affine3f::Identity();
    static float max_search_radius = 0.0f;

    if (!tf_initialized) 
    {
        Eigen::Matrix3f rot_geom;
        rot_geom = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
                 * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
                 * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ());
        tf_tcp_to_mesh.linear() = rot_geom;
        tf_tcp_to_mesh.translation() = Eigen::Vector3f(mesh_offset_x_, mesh_offset_y_, mesh_offset_z_);
        
        float max_box_dist = 0.0f;
        for(const auto& box : gripper_boxes_) {
             float dist = box.max_pt.norm(); 
             if (box.min_pt.norm() > dist) dist = box.min_pt.norm();
             if (dist > max_box_dist) max_box_dist = dist;
        }
        float offset_dist = tf_tcp_to_mesh.translation().norm();
        max_search_radius = offset_dist + max_box_dist + 0.05f;
        
        tf_initialized = true;
    }

    Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
    Eigen::Quaternionf original_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);

    std::vector<int> pointIdx;
    std::vector<float> pointSqDist;
    pcl::PointXYZ searchPoint;
    searchPoint.x = grasp_pos.x(); searchPoint.y = grasp_pos.y(); searchPoint.z = grasp_pos.z();

    if (env_kdtree.radiusSearch(searchPoint, max_search_radius, pointIdx, pointSqDist) == 0) {
        return true; 
    }

    const float MARGIN = 0.002f;
    const auto& cloud_points = env_kdtree.getInputCloud()->points;

   
    const int NUM_STEPS = try_rotations ? 18 : 1; 
    const float ANGLE_STEP = (2.0f * M_PI) / 18.0f; 
    
    bool final_collision_state = true; 

    for (int step = 0; step < NUM_STEPS; ++step)
    {
        float current_angle = step * ANGLE_STEP;
        Eigen::Quaternionf rotation_offset(Eigen::AngleAxisf(current_angle, Eigen::Vector3f::UnitY()));

        Eigen::Quaternionf current_rot = original_rot * rotation_offset;

        Eigen::Affine3f tf_world_to_mesh = Eigen::Translation3f(grasp_pos) * current_rot * tf_tcp_to_mesh;
        Eigen::Affine3f tf_mesh_to_world = tf_world_to_mesh.inverse();

        bool collision_in_this_angle = false;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr debug_cloud;
        if (publish_debug && step == 0) {
            debug_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
            debug_cloud->header.frame_id = "world";
            debug_cloud->reserve(pointIdx.size());
        }

        for (int idx : pointIdx)
        {
            const auto& pt = cloud_points[idx];
            
            Eigen::Vector3f p_world(pt.x, pt.y, pt.z);
            Eigen::Vector3f p_local = tf_mesh_to_world * p_world;

            bool point_is_inside = false;

            for (const auto& box : gripper_boxes_)
            {
                if (p_local.x() >= (box.min_pt.x() - MARGIN) && p_local.x() <= (box.max_pt.x() + MARGIN) &&
                    p_local.y() >= (box.min_pt.y() - MARGIN) && p_local.y() <= (box.max_pt.y() + MARGIN) &&
                    p_local.z() >= (box.min_pt.z() - MARGIN) && p_local.z() <= (box.max_pt.z() + MARGIN))
                {
                    point_is_inside = true;
                    collision_in_this_angle = true;
                    break; 
                }
            }

            if (publish_debug && debug_cloud) {
                pcl::PointXYZRGB p_vis;
                p_vis.x = pt.x; p_vis.y = pt.y; p_vis.z = pt.z;
                if (point_is_inside) { p_vis.r = 255; p_vis.g = 0; p_vis.b = 0; }
                else                 { p_vis.r = 0; p_vis.g = 255; p_vis.b = 0; }
                debug_cloud->points.push_back(p_vis);
            }

            if (collision_in_this_angle && !publish_debug) break;
        }

        if (publish_debug && debug_cloud && !debug_cloud->empty() && step == 0)
        {
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*debug_cloud, msg);
            msg.header.stamp = this->now();
            pub_debug_collision_->publish(msg);
        }

        if (!collision_in_this_angle) {
            
            grasp.pose_center.orientation.w = current_rot.w();
            grasp.pose_center.orientation.x = current_rot.x();
            grasp.pose_center.orientation.y = current_rot.y();
            grasp.pose_center.orientation.z = current_rot.z();

            final_collision_state = false; 
            break; 
        }
    }

    return !final_collision_state;
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


struct ThreadStats {
    double inliers_ms = 0.0;     uint64_t inliers_clk = 0;
    double analysis_ms = 0.0;    uint64_t analysis_clk = 0;
    double scoring_ms = 0.0;     uint64_t scoring_clk = 0;
    
    std::vector<ScoredGrasp> local_candidates;
    std::vector<geometry_msgs::msg::Pose> local_raw_poses; 
};

geometry_msgs::msg::PoseArray GenerateGraspPoses::evaluateGrasps(pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment)
{
    
    auto format_clocks = [](uint64_t n) -> std::string 
    {
        std::string s = std::to_string(n);
        int insertPosition = static_cast<int>(s.length()) - 3;
        while (insertPosition > 0) 
        { 
            s.insert(insertPosition, "."); 
            insertPosition -= 3; 
        }
        return s;
    };

    uint64_t c_func_start = __rdtsc();
    auto t_func_start = std::chrono::high_resolution_clock::now();

    hit_candidates_.clear(); 
    
    
    uint64_t c_kdtree_start = __rdtsc();
    auto t_kdtree_start = std::chrono::high_resolution_clock::now();
    
    pcl::KdTreeFLANN<pcl::PointXYZ> env_kdtree;
    
    if (target_environment->empty()) 
    {
        RCLCPP_WARN(this->get_logger(), "Ambiente vazio, ignorando colisão.");
    } 
    else 
    {
        env_kdtree.setInputCloud(target_environment);
    }
    
    auto t_kdtree_end = std::chrono::high_resolution_clock::now();
    double d_kdtree = std::chrono::duration<double, std::milli>(t_kdtree_end - t_kdtree_start).count();
    uint64_t clk_kdtree = __rdtsc() - c_kdtree_start;

    
    uint64_t c_voxel_start = __rdtsc();
    auto t_voxel_start = std::chrono::high_resolution_clock::now();

    float voxel_size = 0.0075f;
    std::unordered_map<long, VoxelBucket> voxel_grid;
    
    
    auto get_key = [&](int x, int y, int z) -> long 
    {
        return ((long)x * 73856093) ^ ((long)y * 19349663) ^ ((long)z * 83492791);
    };
    
   
    for (const auto& pt : stored_cloud_->points) 
    {
        int ix = std::floor(pt.x / voxel_size); 
        int iy = std::floor(pt.y / voxel_size); 
        int iz = std::floor(pt.z / voxel_size);
        
        long key = get_key(ix, iy, iz);
        
        if (voxel_grid.find(key) == voxel_grid.end()) 
        {
            voxel_grid[key].center = Eigen::Vector3f(
                (ix + 0.5f) * voxel_size, 
                (iy + 0.5f) * voxel_size, 
                (iz + 0.5f) * voxel_size
            );
        }
        voxel_grid[key].points.push_back(pt);
    }

    auto t_voxel_end = std::chrono::high_resolution_clock::now();
    double d_voxel = std::chrono::duration<double, std::milli>(t_voxel_end - t_voxel_start).count();
    uint64_t clk_voxel = __rdtsc() - c_voxel_start;

    
    float voxel_radius = (voxel_size * 1.73205f) / 2.0f;
    float voxel_check_threshold = cylinder_radius_ + voxel_radius; 
    float voxel_check_threshold_squared = voxel_check_threshold * voxel_check_threshold;
    float cylinder_radius_sq = cylinder_radius_ * cylinder_radius_;

    
    double acc_inliers_ms = 0.0;   uint64_t acc_inliers_clk = 0;
    double acc_analysis_ms = 0.0;  uint64_t acc_analysis_clk = 0;
    double acc_scoring_ms = 0.0;   uint64_t acc_scoring_clk = 0;

    std::vector<ScoredGrasp> initial_candidates;
    initial_candidates.reserve(all_candidates_.size());
    int perfect_grasps_count = 0;

    RCLCPP_INFO(this->get_logger(), "Iniciando Processamento SERIAL c/ Animação (Candidates: %lu)...", all_candidates_.size());
    
    uint64_t c_loop_start = __rdtsc();
    auto t_loop_start = std::chrono::high_resolution_clock::now();
  
    for (size_t i = 0; i < all_candidates_.size(); ++i) 
    {
        
        if (perfect_grasps_count >= num_best_grasps_) 
        {
            break;
        }

        const auto& raw_pose = all_candidates_[i]; 
        
        Eigen::Quaternionf q_start(raw_pose.orientation.w, raw_pose.orientation.x, raw_pose.orientation.y, raw_pose.orientation.z);
        Eigen::Vector3f ray_origin_start(raw_pose.position.x, raw_pose.position.y, raw_pose.position.z);
        Eigen::Vector3f ray_dir_start = q_start * Eigen::Vector3f::UnitX(); 

        
        uint64_t c0 = __rdtsc(); 
        auto t0 = std::chrono::high_resolution_clock::now();

        float t_min_init = 1e6;
        float t_max_init = -1e6;
        Eigen::Vector3f PIVOT_POINT = {0.0, 0.0, 0.0}; 
        bool hit_init = false;
        pcl::PointCloud<pcl::PointXYZ>::Ptr init_inliers(new pcl::PointCloud<pcl::PointXYZ>);
        
        
        for (const auto& [key, bucket] : voxel_grid) 
        {
            Eigen::Vector3f diff = bucket.center - ray_origin_start;
            
            
            if ((diff.cross(ray_dir_start)).squaredNorm() > voxel_check_threshold_squared) 
            {
                continue;
            }

            
            for (const auto& pt : bucket.points) 
            {
                Eigen::Vector3f p(pt.x, pt.y, pt.z);
                float t = (p - ray_origin_start).dot(ray_dir_start);
                Eigen::Vector3f dist_vec = p - (ray_origin_start + t * ray_dir_start);
                
                if (dist_vec.squaredNorm() < cylinder_radius_sq) 
                {
                    if (t < t_min_init) 
                    {
                        t_min_init = t;
                        PIVOT_POINT = p;
                    }
                    if (t > t_max_init) t_max_init = t;
                    hit_init = true;
                    init_inliers->points.push_back(pt);
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now(); 
        uint64_t c1 = __rdtsc();
        acc_inliers_ms += std::chrono::duration<double, std::milli>(t1 - t0).count(); 
        acc_inliers_clk += (c1 - c0);

        
        float init_thickness = t_max_init - t_min_init;
        if (!hit_init || init_inliers->size() < 3 || 
            init_thickness < 0.0005 || init_thickness > max_gripper_width_) 
        {
            continue;
        }

        
        uint64_t c2_pre = __rdtsc(); 
        auto t2_pre = std::chrono::high_resolution_clock::now();

        Eigen::Vector4f centroid; 
        Eigen::Matrix3f covariance_matrix;
        
        pcl::compute3DCentroid(*init_inliers, centroid);
        pcl::computeCovarianceMatrixNormalized(*init_inliers, centroid, covariance_matrix);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
        
        Eigen::Vector3f pca_normal = eigen_solver.eigenvectors().col(0);
        
     
        if (!pca_normal.allFinite()) 
        {
            pca_normal = -ray_dir_start; 
        }
        else if (pca_normal.dot(ray_dir_start) > 0) 
        {
            pca_normal = -pca_normal;
        }

        // const Eigen::Vector3f PIVOT_POINT = ray_origin_start + (ray_dir_start * t_min_init); 
        const float DISTANCE_TO_PIVOT = t_min_init;                                        
        const Eigen::Vector3f DIR_START = ray_dir_start;                                   
        const Eigen::Vector3f DIR_TARGET = -pca_normal;                                

        auto t2_post = std::chrono::high_resolution_clock::now(); 
        uint64_t c2_post = __rdtsc();
        acc_analysis_ms += std::chrono::duration<double, std::milli>(t2_post - t2_pre).count();
        acc_analysis_clk += (c2_post - c2_pre);

        ScoredGrasp best_iter_grasp;
        best_iter_grasp.total_score = -1.0; 
        bool found_valid_in_optimization = false;
        bool perfect_candidate_found = false;                          

        
        int max_optimization_steps = 12; 
        

        for (int step = 0; step < max_optimization_steps; ++step)
        {
            
            float t_lerp = (float)step / (float)max_optimization_steps; 
            Eigen::Vector3f current_ray_dir = ((1.0f - t_lerp) * DIR_START + t_lerp * DIR_TARGET).normalized();
            Eigen::Vector3f current_ray_origin = PIVOT_POINT - (current_ray_dir * DISTANCE_TO_PIVOT);

            uint64_t c3 = __rdtsc(); 
            auto t3 = std::chrono::high_resolution_clock::now();
            
            
            float t_min = 1e6;
            float t_max = -1e6;
            bool hit = false;
            pcl::PointCloud<pcl::PointXYZ>::Ptr current_inliers_ptr(new pcl::PointCloud<pcl::PointXYZ>);
            
            for (const auto& [key, bucket] : voxel_grid) 
            {
                Eigen::Vector3f diff = bucket.center - current_ray_origin;
                if ((diff.cross(current_ray_dir)).squaredNorm() > voxel_check_threshold_squared) 
                {
                    continue;
                }
                
                for (const auto& pt : bucket.points) 
                {
                    Eigen::Vector3f p(pt.x, pt.y, pt.z);
                    float t = (p - current_ray_origin).dot(current_ray_dir);
                    Eigen::Vector3f dist_vec = p - (current_ray_origin + t * current_ray_dir);
                    
                    if (dist_vec.squaredNorm() < cylinder_radius_sq) 
                    {
                        if (t < t_min) t_min = t;
                        if (t > t_max) t_max = t;
                        hit = true;
                        current_inliers_ptr->points.push_back(pt);
                    }
                }
            }

            auto t4 = std::chrono::high_resolution_clock::now(); 
            uint64_t c4 = __rdtsc();
            acc_inliers_ms += std::chrono::duration<double, std::milli>(t4 - t3).count(); 
            acc_inliers_clk += (c4 - c3);
            
            
            if (!hit || current_inliers_ptr->size() < 5) 
            {
                continue;
            }

            
            if (enable_ray_animation_ && debug_marker_pub_)
            {
                
                auto t_now = this->now();
                visualization_msgs::msg::MarkerArray markers;

                sensor_msgs::msg::PointCloud2 m; 
                pcl::toROSMsg(*stored_cloud_, m); 
                m.header.stamp = t_now; 
                m.header.frame_id = "world"; 
                pub_cloud_->publish(m);

                // Configuração base para NÃO PISCAR (Lifetime = 0)
                visualization_msgs::msg::Marker base_marker;
                base_marker.header.frame_id = "world";
                base_marker.header.stamp = t_now;
                base_marker.action = visualization_msgs::msg::Marker::ADD;
                base_marker.lifetime = rclcpp::Duration::from_seconds(0); 

                // -----------------------------------------------------------------
                // 1. PIVÔ ESTÁTICO (Esfera Ciano no Ponto de Contato)
                // -----------------------------------------------------------------
                visualization_msgs::msg::Marker pivot_mk = base_marker;
                pivot_mk.ns = "debug_anim_pivot"; 
                pivot_mk.id = 1;
                pivot_mk.type = visualization_msgs::msg::Marker::SPHERE;
                
                pivot_mk.pose.position.x = PIVOT_POINT.x(); 
                pivot_mk.pose.position.y = PIVOT_POINT.y(); 
                pivot_mk.pose.position.z = PIVOT_POINT.z();
                
                pivot_mk.pose.orientation.w = 1.0;

                pivot_mk.scale.x = 0.005; 
                pivot_mk.scale.y = 0.005; 
                pivot_mk.scale.z = 0.005;
                
                pivot_mk.color.a = 1.0; pivot_mk.color.r = 0.0; pivot_mk.color.g = 1.0; pivot_mk.color.b = 1.0; 
                markers.markers.push_back(pivot_mk);

                // -----------------------------------------------------------------
                // 2. CILINDROS DE ANÁLISE (Entry & Exit)
                // -----------------------------------------------------------------
                Eigen::Quaternionf q_cyl = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), current_ray_dir);
                Eigen::Vector3f pos_entry = current_ray_origin + current_ray_dir * t_min;
                Eigen::Vector3f pos_exit  = current_ray_origin + current_ray_dir * t_max;

                // --- Cilindro de Entrada (ENTRY) ---
                visualization_msgs::msg::Marker cyl_entry = base_marker;
                cyl_entry.ns = "debug_anim_cyl_entry";
                cyl_entry.id = 6;
                cyl_entry.type = visualization_msgs::msg::Marker::CYLINDER;

                cyl_entry.pose.position.x = pos_entry.x();
                cyl_entry.pose.position.y = pos_entry.y();
                cyl_entry.pose.position.z = pos_entry.z();
                
                cyl_entry.pose.orientation.w = q_cyl.w();
                cyl_entry.pose.orientation.x = q_cyl.x();
                cyl_entry.pose.orientation.y = q_cyl.y();
                cyl_entry.pose.orientation.z = q_cyl.z();

                cyl_entry.scale.x = cylinder_radius_ * 2.0; 
                cyl_entry.scale.y = cylinder_radius_ * 2.0; 
                cyl_entry.scale.z = cylinder_height_;

                cyl_entry.color.a = 0.3; cyl_entry.color.r = 0.0; cyl_entry.color.g = 0.5; cyl_entry.color.b = 1.0;
                markers.markers.push_back(cyl_entry);

                // --- Cilindro de Saída (EXIT) ---
                if ((t_max - t_min) > 0.005) 
                {
                    visualization_msgs::msg::Marker cyl_exit = cyl_entry; 
                    cyl_exit.ns = "debug_anim_cyl_exit";
                    cyl_exit.id = 7;
                    
                    cyl_exit.pose.position.x = pos_exit.x();
                    cyl_exit.pose.position.y = pos_exit.y();
                    cyl_exit.pose.position.z = pos_exit.z();

                    cyl_exit.color.r = 1.0; cyl_exit.color.g = 0.5; cyl_exit.color.b = 0.0;
                    markers.markers.push_back(cyl_exit);
                }

                // -----------------------------------------------------------------
                // [PREPARAÇÃO] CÁLCULO DOS VETORES ORIGINAIS (Necessário para os Markers abaixo)
                // -----------------------------------------------------------------
                // Correção: Calculamos aqui para usar tanto no Cilindro quanto na Seta
                Eigen::Quaternionf q_orig(
                    all_candidates_[i].orientation.w,
                    all_candidates_[i].orientation.x,
                    all_candidates_[i].orientation.y,
                    all_candidates_[i].orientation.z
                );
                Eigen::Vector3f dir_orig = q_orig * Eigen::Vector3f::UnitX();
                Eigen::Vector3f v_start(
                    all_candidates_[i].position.x, 
                    all_candidates_[i].position.y, 
                    all_candidates_[i].position.z
                );

                // -----------------------------------------------------------------
                // 3. CILINDRO DO RAIO ORIGINAL (Envolvendo a seta roxa)
                // -----------------------------------------------------------------
                visualization_msgs::msg::Marker orig_cyl_mk = base_marker;
                orig_cyl_mk.ns = "debug_anim_cyl_original";
                orig_cyl_mk.id = 8; // ID Único
                orig_cyl_mk.type = visualization_msgs::msg::Marker::CYLINDER;

                // Centro = Inicio + (Direção * Metade do Comprimento)
                Eigen::Vector3f center_cyl_orig = v_start + (dir_orig * (ray_lengths[i] / 2.0f));

                orig_cyl_mk.pose.position.x = center_cyl_orig.x();
                orig_cyl_mk.pose.position.y = center_cyl_orig.y();
                orig_cyl_mk.pose.position.z = center_cyl_orig.z();

                // Rotacionar eixo Z (cilindro) para dir_orig
                Eigen::Quaternionf q_cyl_orig = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), dir_orig);

                orig_cyl_mk.pose.orientation.w = q_cyl_orig.w();
                orig_cyl_mk.pose.orientation.x = q_cyl_orig.x();
                orig_cyl_mk.pose.orientation.y = q_cyl_orig.y();
                orig_cyl_mk.pose.orientation.z = q_cyl_orig.z();

                orig_cyl_mk.scale.x = cylinder_radius_ * 2.0f; 
                orig_cyl_mk.scale.y = cylinder_radius_ * 2.0f; 
                orig_cyl_mk.scale.z = ray_lengths[i];

                orig_cyl_mk.color.a = 0.2; orig_cyl_mk.color.r = 1.0; orig_cyl_mk.color.g = 0.0; orig_cyl_mk.color.b = 1.0;
                markers.markers.push_back(orig_cyl_mk);

                // -----------------------------------------------------------------
                // 4. RAIO ORIGINAL (Seta Magenta)
                // -----------------------------------------------------------------
                visualization_msgs::msg::Marker orig_ray_mk = base_marker;
                orig_ray_mk.ns = "debug_anim_ray_original"; 
                orig_ray_mk.id = 5;
                orig_ray_mk.type = visualization_msgs::msg::Marker::ARROW;

                geometry_msgs::msg::Point p_orig_start, p_orig_end;
                p_orig_start.x = v_start.x(); 
                p_orig_start.y = v_start.y(); 
                p_orig_start.z = v_start.z();

                Eigen::Vector3f v_end = v_start + (dir_orig * ray_lengths[i]);

                p_orig_end.x = v_end.x(); 
                p_orig_end.y = v_end.y(); 
                p_orig_end.z = v_end.z();

                orig_ray_mk.points.push_back(p_orig_start); 
                orig_ray_mk.points.push_back(p_orig_end);
                
                orig_ray_mk.scale.x = 0.002; orig_ray_mk.scale.y = 0.004; orig_ray_mk.scale.z = 0.0;  
                orig_ray_mk.color.a = 0.6; orig_ray_mk.color.r = 1.0; orig_ray_mk.color.g = 0.0; orig_ray_mk.color.b = 1.0;
                markers.markers.push_back(orig_ray_mk);

                // -----------------------------------------------------------------
                // 5. RAIO DA GARRA ATUAL (Seta Amarela)
                // -----------------------------------------------------------------
                visualization_msgs::msg::Marker ray_mk = base_marker;
                ray_mk.ns = "debug_anim_ray"; 
                ray_mk.id = 2;
                ray_mk.type = visualization_msgs::msg::Marker::ARROW;

                geometry_msgs::msg::Point p_start, p_end;
                p_start.x = current_ray_origin.x(); 
                p_start.y = current_ray_origin.y(); 
                p_start.z = current_ray_origin.z();

                Eigen::Vector3f visual_end = current_ray_origin + (current_ray_dir * (DISTANCE_TO_PIVOT + 0.05f));
                p_end.x = visual_end.x(); 
                p_end.y = visual_end.y(); 
                p_end.z = visual_end.z();

                ray_mk.points.push_back(p_start); 
                ray_mk.points.push_back(p_end);
                
                ray_mk.scale.x = 0.003; ray_mk.scale.y = 0.006; ray_mk.scale.z = 0.01;  
                ray_mk.color.a = 0.8; ray_mk.color.r = 1.0; ray_mk.color.g = 1.0; ray_mk.color.b = 0.0;
                markers.markers.push_back(ray_mk);

                // -----------------------------------------------------------------
                // 6. NORMAL DA SUPERFÍCIE (Seta Vermelha)
                // -----------------------------------------------------------------
                visualization_msgs::msg::Marker norm_mk = base_marker;
                norm_mk.ns = "debug_anim_normal";
                norm_mk.id = 3;
                norm_mk.type = visualization_msgs::msg::Marker::ARROW;

                geometry_msgs::msg::Point p_piv_geom;
                p_piv_geom.x = PIVOT_POINT.x(); p_piv_geom.y = PIVOT_POINT.y(); p_piv_geom.z = PIVOT_POINT.z();

                Eigen::Vector3f norm_end = PIVOT_POINT + (pca_normal * 0.04f);
                geometry_msgs::msg::Point p_norm_end;
                p_norm_end.x = norm_end.x(); p_norm_end.y = norm_end.y(); p_norm_end.z = norm_end.z();

                norm_mk.points.push_back(p_piv_geom);
                norm_mk.points.push_back(p_norm_end);

                norm_mk.scale.x = 0.002; norm_mk.scale.y = 0.004; norm_mk.scale.z = 0.0;
                norm_mk.color.a = 1.0; norm_mk.color.r = 1.0; norm_mk.color.g = 0.0; norm_mk.color.b = 0.0;
                markers.markers.push_back(norm_mk);

                // -----------------------------------------------------------------
                // 7. INLIERS (Pontos Brancos)
                // -----------------------------------------------------------------
                if (current_inliers_ptr && !current_inliers_ptr->empty()) 
                {
                    visualization_msgs::msg::Marker pts_mk = base_marker;
                    pts_mk.ns = "debug_anim_inliers"; 
                    pts_mk.id = 4;
                    pts_mk.type = visualization_msgs::msg::Marker::POINTS;
                    pts_mk.scale.x = 0.0015; pts_mk.scale.y = 0.0015;
                    pts_mk.color.a = 1.0; pts_mk.color.r = 1.0; pts_mk.color.g = 0.0; pts_mk.color.b = 0.0;

                    pts_mk.points.reserve(current_inliers_ptr->size());
                    for (const auto& p : current_inliers_ptr->points) 
                    {
                        geometry_msgs::msg::Point gp;
                        gp.x = p.x; gp.y = p.y; gp.z = p.z;
                        pts_mk.points.push_back(gp);
                    }
                    markers.markers.push_back(pts_mk);
                }

                debug_marker_pub_->publish(markers);
                
                std::this_thread::sleep_for(std::chrono::milliseconds(animation_delay_ms_));
            }
            
            float real_thickness = t_max - t_min;
            if (real_thickness < 0.002 || real_thickness > max_gripper_width_) 
            {
                continue;
            }

            
            uint64_t c5 = __rdtsc(); 
            auto t5 = std::chrono::high_resolution_clock::now();

            std::vector<StepAnalysis> steps; 
            steps.reserve(2); 
            
            Eigen::Vector3f center_entry = current_ray_origin + current_ray_dir * t_min;
            StepAnalysis res_entry = analyzeLocalCylinder(current_inliers_ptr, center_entry, current_ray_dir, cylinder_radius_, cylinder_height_);
            
            if (res_entry.valid) 
            {
                steps.push_back(res_entry);
            }

            if (real_thickness > 0.001f) 
            { 
                Eigen::Vector3f center_exit = current_ray_origin + current_ray_dir * t_max;
                StepAnalysis res_exit = analyzeLocalCylinder(current_inliers_ptr, center_exit, current_ray_dir, cylinder_radius_, cylinder_height_);
                
                if (res_exit.valid) 
                {
                    steps.push_back(res_exit);
                }
            }
            
            auto t6 = std::chrono::high_resolution_clock::now(); 
            uint64_t c6 = __rdtsc();
            acc_analysis_ms += std::chrono::duration<double, std::milli>(t6 - t5).count();
            acc_analysis_clk += (c6 - c5);

            if (steps.empty()) 
            {
                continue; 
            }

           
            uint64_t c7 = __rdtsc(); 
            auto t7 = std::chrono::high_resolution_clock::now();

            StepAnalysis& entry = steps.front();
            StepAnalysis& exit = steps.back();
            
            float score_ang_entry = 1.0f - (std::min(entry.angle_to_normal_deg, 90.0f) / 90.0f);
            float score_ang_exit  = 1.0f - (std::min(exit.angle_to_normal_deg, 90.0f) / 90.0f);
            
            float score_plan_entry = std::max(0.0f, 1.0f - (entry.curvature * 20.0f)); 
            float score_plan_exit  = std::max(0.0f, 1.0f - (exit.curvature * 20.0f));
            
            float orient_factor_entry = (score_plan_entry > 0.3) ? 1.0f : 0.5f;
            float orient_factor_exit  = (score_plan_exit > 0.3) ? 1.0f : 0.5f;
            
            float score_sym_entry = entry.symmetry_score;
            float score_sym_exit  = exit.symmetry_score;

            double total = (score_ang_entry * weight_orientation_ * orient_factor_entry + score_sym_entry * weight_symmetry_ ) * 0.5 
                         + (score_ang_exit * weight_orientation_ * orient_factor_exit + score_sym_exit * weight_symmetry_ ) * 0.5;

            auto t8 = std::chrono::high_resolution_clock::now(); 
            uint64_t c8 = __rdtsc();
            acc_scoring_ms += std::chrono::duration<double, std::milli>(t8 - t7).count();
            acc_scoring_clk += (c8 - c7);

            
            if (total > best_iter_grasp.total_score) 
            {
                found_valid_in_optimization = true;
                
                float current_offset = finger_offset_; 
                Eigen::Vector3f p_f1 = current_ray_origin + current_ray_dir * (t_min - current_offset);
                Eigen::Vector3f p_f2 = current_ray_origin + current_ray_dir * (t_max + current_offset);
                Eigen::Vector3f center_grasp = (p_f1 + p_f2) / 2.0f;
                
                Eigen::Quaternionf best_q = findBestOrientation(p_f1, p_f2);

                best_iter_grasp.pose_center.position.x = center_grasp.x(); 
                best_iter_grasp.pose_center.position.y = center_grasp.y(); 
                best_iter_grasp.pose_center.position.z = center_grasp.z();
                
                best_iter_grasp.pose_center.orientation.x = best_q.x(); 
                best_iter_grasp.pose_center.orientation.y = best_q.y(); 
                best_iter_grasp.pose_center.orientation.z = best_q.z(); 
                best_iter_grasp.pose_center.orientation.w = best_q.w();
                
                best_iter_grasp.total_score = total;
                best_iter_grasp.raw_ray_dir = current_ray_dir;
            }

            
            if (total >= target_score_) 
            { 
                if (check_collision(best_iter_grasp, collision_kdtree_, true, false)) 
                { 
                    initial_candidates.push_back(best_iter_grasp);
                    hit_candidates_.push_back(raw_pose);
                    
                    perfect_grasps_count++;
                    perfect_candidate_found = true;
                    
                    if (enable_ray_animation_) 
                    {
                        RCLCPP_INFO(this->get_logger(), "GRASP PERFEITO ENCONTRADO! Score: %.4f", total);
                    }
                    break; 
                }
            }
        } 

        if (!perfect_candidate_found && found_valid_in_optimization) 
        {
            initial_candidates.push_back(best_iter_grasp);
            hit_candidates_.push_back(raw_pose);
        }
    }



    auto t_loop_end = std::chrono::high_resolution_clock::now();
    uint64_t c_loop_end = __rdtsc();

    if (initial_candidates.empty()) 
    {
        has_best_ = false; 
        return geometry_msgs::msg::PoseArray();
    }

    
    uint64_t c_sort_start = __rdtsc();
    auto t_sort_start = std::chrono::high_resolution_clock::now();
    
    std::sort(initial_candidates.begin(), initial_candidates.end(), 
        [](const ScoredGrasp& a, const ScoredGrasp& b) 
        { 
            return a.total_score > b.total_score; 
        });
        
    auto t_sort_end = std::chrono::high_resolution_clock::now();
    uint64_t c_sort_end = __rdtsc();

    
    uint64_t c_collision_start = __rdtsc();
    auto t_collision_start = std::chrono::high_resolution_clock::now();

    best_grasps_.clear();
    best_grasps_.reserve(num_best_grasps_); 
    
    int checks_count = 0; 
    
    for (auto& candidate : initial_candidates)
    {
        if (best_grasps_.size() >= (size_t)num_best_grasps_) 
        {
            break;
        }
        
        checks_count++;
        if (check_collision(candidate, collision_kdtree_, false, true)) 
        {
            best_grasps_.push_back(candidate);
        }
        
        
    }
    has_best_ = !best_grasps_.empty();
    
    auto t_collision_end = std::chrono::high_resolution_clock::now();
    uint64_t c_collision_end = __rdtsc();
    
   
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.frame_id = "world"; 
    pose_array.header.stamp = this->now(); 
    
    for(const auto& bg : best_grasps_) 
    {
        pose_array.poses.push_back(bg.pose_center);
    }

    auto t_func_end = std::chrono::high_resolution_clock::now();
    uint64_t c_func_end = __rdtsc();

    double d_func = std::chrono::duration<double, std::milli>(t_func_end - t_func_start).count();
    double d_loop = std::chrono::duration<double, std::milli>(t_loop_end - t_loop_start).count();
    double d_sort = std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count();
    double d_col = std::chrono::duration<double, std::milli>(t_collision_end - t_collision_start).count();

    uint64_t clk_func = c_func_end - c_func_start;
    
    RCLCPP_INFO(this->get_logger(), "================ BENCHMARK SERIAL (ANIMATION MODE) ===============");
    RCLCPP_INFO(this->get_logger(), "Tempo Total Função:      %.4f ms | Clocks: %s", d_func, format_clocks(clk_func).c_str());
    RCLCPP_INFO(this->get_logger(), "  -> Serial Loop:        %.4f ms", d_loop);
    
    RCLCPP_INFO(this->get_logger(), "     |-> Total Inliers:  %.4f ms | Clocks: %s", acc_inliers_ms, format_clocks(acc_inliers_clk).c_str());
    RCLCPP_INFO(this->get_logger(), "     |-> Total Analysis: %.4f ms | Clocks: %s", acc_analysis_ms, format_clocks(acc_analysis_clk).c_str());
    RCLCPP_INFO(this->get_logger(), "     |-> Total Scoring:  %.4f ms | Clocks: %s", acc_scoring_ms, format_clocks(acc_scoring_clk).c_str());
    
    RCLCPP_INFO(this->get_logger(), "  -> Sort:               %.4f ms", d_sort);
    RCLCPP_INFO(this->get_logger(), "  -> Collision Check:    %.4f ms (%d checks)", d_col, checks_count);
    
    size_t num_to_print = std::min((size_t)5, best_grasps_.size());
    if (num_to_print > 0) 
    {
        RCLCPP_INFO(this->get_logger(), ">>> TOP %lu SCORES <<<", num_to_print);
        for (size_t i = 0; i < num_to_print; ++i) 
        {
             RCLCPP_INFO(this->get_logger(), "  #%02lu: Score: %.4f", i+1, best_grasps_[i].total_score);
        }
    } 
    else 
    {
        RCLCPP_WARN(this->get_logger(), ">>> NENHUM GRASP ENCONTRADO <<<");
    }
    RCLCPP_INFO(this->get_logger(), "============================================================");

    g_last_run_stats.total_func = d_func;
    g_last_run_stats.loop_tbb = d_loop; 
    g_last_run_stats.max_inliers = acc_inliers_ms;
    g_last_run_stats.max_analysis = acc_analysis_ms;
    g_last_run_stats.max_scoring = acc_scoring_ms;
    g_last_run_stats.sort = d_sort;
    g_last_run_stats.collision = d_col;
    g_last_run_stats.checks = checks_count;

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

        
        Eigen::Affine3f tf_collision_final = tf_tcp_to_world * tf_geometry_to_tcp;
        Eigen::Quaternionf q_coll(tf_collision_final.rotation());

       
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

            Eigen::Vector3f center_world = tf_collision_final * box.center;
            
            mk.pose.position.x = center_world.x();
            mk.pose.position.y = center_world.y();
            mk.pose.position.z = center_world.z();

            mk.pose.orientation.x = q_coll.x();
            mk.pose.orientation.y = q_coll.y();
            mk.pose.orientation.z = q_coll.z();
            mk.pose.orientation.w = q_coll.w();

            mk.scale.x = box.dimensions.x();
            mk.scale.y = box.dimensions.y();
            mk.scale.z = box.dimensions.z();

            if (i == 0) { mk.color.r = 0.0; mk.color.g = 1.0; mk.color.b = 0.0; mk.color.a = 0.6; }
            else        { mk.color.r = 0.0; mk.color.g = 1.0; mk.color.b = 1.0; mk.color.a = 0.3; }
            
            ma.markers.push_back(mk);
        }

        
        if (publish_gripper_mesh_)
        {
            
            Eigen::Affine3f visual_tf = Eigen::Affine3f::Identity();
            visual_tf.translation() = Eigen::Vector3f(-0.015, mesh_offset_y_, mesh_offset_z_);
            
            Eigen::Matrix3f rot;
            rot = Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX())
                * Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f::UnitY())
                * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());
            visual_tf.linear() = rot;

            
            Eigen::Affine3f tf_mesh_final = tf_tcp_to_world * visual_tf;
            
            Eigen::Vector3f t_mesh = tf_mesh_final.translation();
            Eigen::Quaternionf q_mesh(tf_mesh_final.rotation());

            
            visualization_msgs::msg::Marker mesh_mk;
            mesh_mk.header.frame_id = "world";
            mesh_mk.header.stamp = t;
            
            mesh_mk.ns = "gripper_mesh_visual_" + std::to_string(i);
            mesh_mk.id = i;
            mesh_mk.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            mesh_mk.action = visualization_msgs::msg::Marker::ADD;
            mesh_mk.mesh_resource = "file://" + gripper_glb_path_;
            
            mesh_mk.pose.position.x = t_mesh.x();
            mesh_mk.pose.position.y = t_mesh.y();
            mesh_mk.pose.position.z = t_mesh.z();
            
            mesh_mk.pose.orientation.x = q_mesh.x();
            mesh_mk.pose.orientation.y = q_mesh.y();
            mesh_mk.pose.orientation.z = q_mesh.z();
            mesh_mk.pose.orientation.w = q_mesh.w();

            mesh_mk.scale.x = gripper_mesh_scale_;
            mesh_mk.scale.y = gripper_mesh_scale_;
            mesh_mk.scale.z = gripper_mesh_scale_;

            if (i == 0) 
            { 
                mesh_mk.color.r = 0.0; 
                mesh_mk.color.g = 1.0;
                mesh_mk.color.b = 0.0; 
                mesh_mk.color.a = 0.7; 
            }
            else        
            { 
                mesh_mk.color.r = 0.0; 
                mesh_mk.color.g = 1.0; 
                mesh_mk.color.b = 1.0; 
                mesh_mk.color.a = 0.4; 
            }

            mesh_mk.mesh_use_embedded_materials = true;

            ma.markers.push_back(mesh_mk);
        }
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