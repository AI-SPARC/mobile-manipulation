#include "vision/GenerateGraspPoses.hpp" 
#include "vision/VoxelCollision.hpp"
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
#include <filesystem>
// Mensagens para Benchmark
#include <std_msgs/msg/float64.hpp>

// TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
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
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>

using namespace std::chrono_literals;

namespace vision 
{

struct GlobalBenchStats {
    double total_func;
    double loop_tbb;
    double max_inliers;
    double max_inliers_no_hit;
    double max_analysis;
    double max_scoring;
    double sort;
    double collision;
    double kd_tree;
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

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_total_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_loop_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_inliers_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_analysis_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_scoring_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_sort_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr t_collision_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr n_checks_avg;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr score_avg_run_avg;
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
    this->declare_parameter<std::string>("pcd_path", "/home/momesso/pcds/Wrench.pcd");
    
    this->declare_parameter<std::string>("object_mesh_path", "/home/momesso/pcds/GLB_Foxglove/Wrench.glb");
    this->declare_parameter<bool>("publish_object_mesh", false);

    this->declare_parameter<std::string>("gripper_mesh_path", "/home/momesso/hand_and_fingers.obj");
    this->declare_parameter<double>("gripper_mesh_scale", 1.0);
    
    this->declare_parameter<std::string>("gripper_glb_path", "/home/momesso/pcds/GLB_Foxglove/PandaHand.glb");
    this->declare_parameter<bool>("publish_gripper_mesh", false);
    
    this->declare_parameter<double>("mesh_offset_x", 0.0);
    this->declare_parameter<double>("mesh_offset_y", 0.0);
    this->declare_parameter<double>("mesh_offset_z", 0.025);
    
    this->declare_parameter<double>("mesh_rot_roll", 1.57);
    this->declare_parameter<double>("mesh_rot_pitch", 0.0);
    this->declare_parameter<double>("mesh_rot_yaw", 0.0); 

    this->declare_parameter<double>("grid_res", 0.005);
    this->declare_parameter<double>("cloud_voxel_size", 0.001);
    
    this->declare_parameter<double>("sphere_radius", 0.01); 
    this->declare_parameter<double>("analysis_step_size", 0.01);
    
    this->declare_parameter<double>("max_gripper_width", 0.07); 
    this->declare_parameter<double>("finger_offset", 0.027); 
    
    this->declare_parameter<int>("min_points_per_segment", 2);
    this->declare_parameter<double>("weight_orientation", 0.6); 
    this->declare_parameter<double>("weight_symmetry", 0.4);
    this->declare_parameter<double>("target_score", 10.0);
    
    this->declare_parameter<bool>("use_mean_filter", false); 
    this->declare_parameter<int>("mean_filter_k", 15);

    this->declare_parameter<int>("num_best_grasps", 100);
    this->declare_parameter<double>("rotation_step_deg", 15.0);

    this->declare_parameter<int>("num_random_orientations", 20);

    this->declare_parameter<int>("num_benchmark_runs", 1);
    this->declare_parameter<bool>("enable_ray_animation", false);
    this->declare_parameter<int>("animation_delay_ms", 5000);

    this->declare_parameter<bool>("activate_centroid", false);
    this->declare_parameter<bool>("eval_mode", true);

    use_pcd_file = this->get_parameter("use_pcd_file").as_bool();
    pcd_path_ = this->get_parameter("pcd_path").as_string();
    eval_mode_ = this->get_parameter("eval_mode").as_bool();
    std::string pcd_amb_path_ = "";

    const std::vector<std::string>& args = options.arguments();
    std::vector<std::string> pcd_args;
    for (const auto& arg : args) {
        if (arg.find(".pcd") != std::string::npos) {
            pcd_args.push_back(arg);
        }
    }

    if (pcd_args.size() >= 2) {
        use_pcd_file = true;
        pcd_path_ = pcd_args[0];
        pcd_amb_path_ = pcd_args[1];
    } else if (pcd_args.size() == 1) {
        use_pcd_file = true;
        pcd_path_ = pcd_args[0];
    }

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

    sphere_radius_ = static_cast<float>(this->get_parameter("sphere_radius").as_double());

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

    activate_centroid = this->get_parameter("activate_centroid").as_bool();
    
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
    sphere_debug_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("debug/sphere_sectors", rclcpp::QoS(1).transient_local());
    
    pub_object_mesh_ = this->create_publisher<visualization_msgs::msg::Marker>("debug_object_mesh", qos_profile);

    if(use_pcd_file == true)
    {
        rclcpp::QoS latched_qos(10);
        latched_qos.transient_local();
        latched_qos.reliable();

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
        g_bench_pubs->score_avg_run_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/score/avg_run/avg", latched_qos);
        g_bench_pubs->t_total_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/avg_run/total_ms", 10);
        g_bench_pubs->t_loop_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/avg_run/loop_serial_ms", 10);
        g_bench_pubs->t_inliers_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/avg_run/parts/inliers_ms", 10);
        g_bench_pubs->t_analysis_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/avg_run/parts/analysis_ms", 10);
        g_bench_pubs->t_scoring_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/avg_run/parts/scoring_ms", 10);
        g_bench_pubs->t_sort_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/avg_run/parts/sort_ms", 10);
        g_bench_pubs->t_collision_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/time/avg_run/parts/collision_ms", 10);
        g_bench_pubs->n_checks_avg = this->create_publisher<std_msgs::msg::Float64>("/grasp_bench/stats/avg_run/collision_checks", 10);
    }
    
    stored_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    gripper_dense_cloud_.reset(new pcl::PointCloud<pcl::PointXYZRGB>); 
    
    collision_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    extractBoundingBoxesFromOBJ(); 
    
    publishGripperModel();
    publishGripperCollisionBoxes();

    RCLCPP_INFO(this->get_logger(), "MODO ARQUIVO: Verificando %s...", pcd_path_.c_str());

    if(use_pcd_file == true)
    {
        if (std::filesystem::is_directory(pcd_path_))
        {
            RCLCPP_INFO(this->get_logger(), "Pasta detectada! Iniciando processamento em lote (batch)...");
            
            // Vetores para guardar os resultados médios de cada arquivo PCD processado
            std::vector<double> all_scores, all_totals, all_kds, all_loops;
            std::vector<double> all_inliers, all_analysis, all_scorings;
            std::vector<double> all_sorts, all_cols, all_checks;

            std::vector<std::string> pcd_files;
            for (const auto& entry : std::filesystem::directory_iterator(pcd_path_)) {
                if (entry.path().extension() == ".pcd") {
                    pcd_files.push_back(entry.path().string());
                }
            }
            std::sort(pcd_files.begin(), pcd_files.end());

            for (const auto& file : pcd_files)
            {
                RCLCPP_INFO(this->get_logger(), ">>> Processando PCD: %s", file.c_str());
                std::vector<double> res = loadAndProcess(file, pcd_amb_path_); 
                
                if (!res.empty()) {
                    all_scores.push_back(res[0]);
                    all_totals.push_back(res[1]);
                    all_kds.push_back(res[2]);
                    all_loops.push_back(res[3]);
                    all_inliers.push_back(res[4]);
                    all_analysis.push_back(res[5]);
                    all_scorings.push_back(res[6]);
                    all_sorts.push_back(res[7]);
                    all_cols.push_back(res[8]);
                    all_checks.push_back(res[9]);
                }
            }

            if (!all_totals.empty()) {
                auto print_stats = [&](const std::string& name, std::vector<double>& v, bool is_score_or_check = false) {
                    std::sort(v.begin(), v.end());
                    double min_v = v.front();
                    double max_v = v.back();
                    double median = v.size() % 2 == 0 ? (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2.0 : v[v.size() / 2];
                    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();

                    std::string unit = is_score_or_check ? "" : " ms";
                    RCLCPP_INFO(this->get_logger(), "  -> %-18s | Média: %8.4f%s | Mediana: %8.4f%s | Min: %8.4f%s | Max: %8.4f%s",
                        name.c_str(), mean, unit.c_str(), median, unit.c_str(), min_v, unit.c_str(), max_v, unit.c_str());
                };

                RCLCPP_INFO(this->get_logger(), " ");
                RCLCPP_INFO(this->get_logger(), "=========================================================================================================");
                RCLCPP_INFO(this->get_logger(), "=== BATCH CONCLUÍDO | %lu ARQUIVOS PROCESSADOS (CADA UM COM %d EXECUÇÕES DE BENCHMARK)", all_totals.size(), num_benchmark_runs_);
                RCLCPP_INFO(this->get_logger(), "=========================================================================================================");
                print_stats("Score Geral", all_scores, true);
                print_stats("Tempo Total", all_totals);
                print_stats("KD-Tree", all_kds);
                print_stats("TBB Loop", all_loops);
                print_stats("   |-> Inliers", all_inliers);
                print_stats("   |-> Analysis", all_analysis);
                print_stats("   |-> Scoring", all_scorings);
                print_stats("Sort", all_sorts);
                print_stats("Collision Final", all_cols);
                print_stats("Checks (Qtd)", all_checks, true);
                RCLCPP_INFO(this->get_logger(), "=========================================================================================================");
            }
        }
        else
        {
            
            loadAndProcess(pcd_path_, pcd_amb_path_);
        }
    }

   // Inscrições
    cloud_map_sub_.subscribe(this, "/cloud_map", rmw_qos_profile_sensor_data);
    segmented_cloud_sub_.subscribe(this, "/gsam2/segmented_cloud", rmw_qos_profile_sensor_data);

    // Inicializa o sincronizador
    sync_.reset(new message_filters::Synchronizer<PointCloudSyncPolicy>(
        PointCloudSyncPolicy(10), segmented_cloud_sub_, cloud_map_sub_));

    // Faz o bind com 2 placeholders (um para cada PointCloud)
    sync_->registerCallback(
        std::bind(&GenerateGraspPoses::pointcloud_callback, this, std::placeholders::_1, std::placeholders::_2));
    
    timer_ = this->create_wall_timer(10ms, std::bind(&GenerateGraspPoses::timerCallback, this));
}

void GenerateGraspPoses::pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& segmented_cloud_msg,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_map_msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map_pcl(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);

    try 
    {
        pcl::fromROSMsg(*cloud_map_msg, *cloud_map_pcl);
        pcl::fromROSMsg(*segmented_cloud_msg, *segmented_cloud_pcl);

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "Nuvens sincronizadas! Segmentada: %d pts | Mapa: %d pts. Frame ID: %s", 
            (int)segmented_cloud_pcl->points.size(), 
            (int)cloud_map_pcl->points.size(), 
            segmented_cloud_msg->header.frame_id.c_str());

        if (segmented_cloud_pcl->points.empty()) 
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                "Nuvem segmentada vazia. Ignorando...");
            return;
        }

        // Passa as nuvens para o processamento de Grasp
        processCloud(segmented_cloud_pcl, cloud_map_pcl);
    }
    catch (const std::exception& e) 
    {
        RCLCPP_ERROR(this->get_logger(), "Erro ao converter as nuvens de pontos: %s", e.what());
    }
}
std::vector<double> GenerateGraspPoses::loadAndProcess(const std::string& obj_path, const std::string& amb_path)
{
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr obj_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr amb_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Carrega a nuvem do objeto
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(obj_path, *obj_cloud) == -1) 
    {
        RCLCPP_ERROR(this->get_logger(), "Falha ao ler arquivo PCD do objeto: %s", obj_path.c_str());
        return {}; // Retorna vetor vazio em caso de erro
    }
    // Carrega a nuvem do ambiente (se fornecida e válida), senão usa o próprio objeto como fallback
    if (!amb_path.empty() && pcl::io::loadPCDFile<pcl::PointXYZ>(amb_path, *amb_cloud) != -1) {
        RCLCPP_INFO(this->get_logger(), "Nuvem de ambiente carregada para colisão: %s", amb_path.c_str());
    } else {
        *amb_cloud = *obj_cloud;
    }
    
    GlobalBenchStats acc_stats = {0,0,0,0,0,0,0,0,0,0};
    double acc_score_avg_total = 0.0;
    
    int runs = std::max(1, num_benchmark_runs_);

    RCLCPP_INFO(this->get_logger(), ">>> INICIANDO PROCESSAMENTO: %d execuções <<<", runs);

    for(int i = 0; i < runs; ++i)
    {
        hit_candidates_.clear();
        best_grasps_.clear();
        
        
        processCloud(obj_cloud, amb_cloud);
        
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

        acc_stats.kd_tree += g_last_run_stats.kd_tree;
        acc_stats.total_func += g_last_run_stats.total_func;
        acc_stats.loop_tbb += g_last_run_stats.loop_tbb;
        acc_stats.max_inliers_no_hit += g_last_run_stats.max_inliers_no_hit;
        acc_stats.max_inliers += g_last_run_stats.max_inliers;
        acc_stats.max_analysis += g_last_run_stats.max_analysis;
        acc_stats.max_scoring += g_last_run_stats.max_scoring;
        acc_stats.sort += g_last_run_stats.sort;
        acc_stats.collision += g_last_run_stats.collision;
        acc_stats.checks += g_last_run_stats.checks;

        double current_count = static_cast<double>(i + 1);

        acc_score_avg_total += avg_score;
        
        double current_mean_of_scores = acc_score_avg_total / current_count; 
        g_bench_pubs->score_avg_run_avg->publish(msg_f64(current_mean_of_scores));

        g_bench_pubs->t_total_avg->publish(msg_f64(acc_stats.total_func / current_count));
        g_bench_pubs->t_loop_avg->publish(msg_f64(acc_stats.loop_tbb / current_count));
        g_bench_pubs->t_inliers_avg->publish(msg_f64(acc_stats.max_inliers / current_count));
        g_bench_pubs->t_analysis_avg->publish(msg_f64(acc_stats.max_analysis / current_count));
        g_bench_pubs->t_scoring_avg->publish(msg_f64(acc_stats.max_scoring / current_count));
        g_bench_pubs->t_sort_avg->publish(msg_f64(acc_stats.sort / current_count));
        g_bench_pubs->t_collision_avg->publish(msg_f64(acc_stats.collision / current_count));
        g_bench_pubs->n_checks_avg->publish(msg_f64((double)acc_stats.checks / current_count));
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    

    double div = static_cast<double>(runs);
    std::vector<double> file_results(10, 0.0);
    file_results[0] = acc_score_avg_total / div;
    file_results[1] = acc_stats.total_func / div;
    file_results[2] = acc_stats.kd_tree / div;
    file_results[3] = acc_stats.loop_tbb / div;
    file_results[4] = acc_stats.max_inliers / div;
    file_results[5] = acc_stats.max_analysis / div;
    file_results[6] = acc_stats.max_scoring / div;
    file_results[7] = acc_stats.sort / div;
    file_results[8] = acc_stats.collision / div;
    file_results[9] = static_cast<double>(acc_stats.checks) / div;

    RCLCPP_INFO(this->get_logger(), " ");
    RCLCPP_INFO(this->get_logger(), "============================================================");
    RCLCPP_INFO(this->get_logger(), "======= RESULTADO FINAL: MÉDIAS APÓS %d EXECUÇÕES ========", runs);
    RCLCPP_INFO(this->get_logger(), "============================================================");
    RCLCPP_INFO(this->get_logger(), "Score Médio Geral:       %.4f", file_results[0]);
    RCLCPP_INFO(this->get_logger(), "Tempo Total Médio:       %.4f ms", file_results[1]);
    RCLCPP_INFO(this->get_logger(), "  -> KD-Tree Médio:      %.4f ms", file_results[2]);
    RCLCPP_INFO(this->get_logger(), "  -> TBB Loop Médio:     %.4f ms", file_results[3]);
    RCLCPP_INFO(this->get_logger(), "     |-> Inliers (Max):  %.4f ms", file_results[4]);
    RCLCPP_INFO(this->get_logger(), "     |-> Analysis (Max): %.4f ms", file_results[5]);
    RCLCPP_INFO(this->get_logger(), "     |-> Scoring (Max):  %.4f ms", file_results[6]);
    RCLCPP_INFO(this->get_logger(), "  -> Sort Médio:         %.4f ms", file_results[7]);
    RCLCPP_INFO(this->get_logger(), "  -> Collision Médio:    %.4f ms (Checks Avg: %.1f)", file_results[8], file_results[9]);
    RCLCPP_INFO(this->get_logger(), "  -> Target Score:       %.4f ms", target_score_);
    RCLCPP_INFO(this->get_logger(), "============================================================");

    // Retorna todas as médias pro batch principal agrupar
    return file_results;
}
void GenerateGraspPoses::processCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr target, pcl::PointCloud<pcl::PointXYZ>::Ptr target_environment)
{
    if (!target || target->empty()) return;
    
     if (use_pcd_file == true)
    {
        Eigen::Vector4f temp_centroid;
        pcl::compute3DCentroid(*target, temp_centroid);
        global_centroid = temp_centroid.head<3>();
 
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
 
        if (!eval_mode_)   // ← ADICIONE ESTA LINHA
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(0.0f, 2.0f * M_PI);
            float rot_x = dis(gen); float rot_y = dis(gen); float rot_z = dis(gen);
            transform.rotate(Eigen::AngleAxisf(rot_x, Eigen::Vector3f::UnitX()));
            transform.rotate(Eigen::AngleAxisf(rot_y, Eigen::Vector3f::UnitY()));
            transform.rotate(Eigen::AngleAxisf(rot_z, Eigen::Vector3f::UnitZ()));
        }        

        transform.translate(-global_centroid);
        
        

      

        pcl::transformPointCloud(*target, *target, transform);
        if (target_environment && !target_environment->empty()) 
        {
            pcl::transformPointCloud(*target_environment, *target_environment, transform);
        }
        
     
        if (publish_object_mesh_)
        {
            visualization_msgs::msg::Marker mesh_marker;
            mesh_marker.header.frame_id = "map";
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
    
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    if (use_pcd_file && cloud_voxel_size_ >= 0.001f) 
    {
        float inv_res = 1.0f / cloud_voxel_size_;
        
        // Estrutura de Hash mágica para achar o Voxel em tempo O(1)
        struct VoxelHash {
            std::size_t operator()(const std::tuple<int, int, int>& k) const {
                return (std::get<0>(k) * 73856093) ^ (std::get<1>(k) * 19349663) ^ (std::get<2>(k) * 83492791);
            }
        };

        // Mapa que guarda apenas 1 ponto por Voxel
        std::unordered_map<std::tuple<int, int, int>, pcl::PointXYZ, VoxelHash> grid;
        grid.reserve(target->size()); // Previne realocação de memória (Ganha muita velocidade)

        for (const auto& pt : target->points) {
            int ix = std::floor(pt.x * inv_res);
            int iy = std::floor(pt.y * inv_res);
            int iz = std::floor(pt.z * inv_res);
            
            // Salva o ponto no grid. Se já tiver um ponto lá, ele sobrescreve (muito mais rápido que tirar média)
            grid[{ix, iy, iz}] = pt; 
        }

        // Passa os pontos filtrados para a nuvem final
        voxel_cloud->reserve(grid.size());
        for (const auto& kv : grid) {
            voxel_cloud->points.push_back(kv.second);
        }
        
        voxel_cloud->width = voxel_cloud->points.size();
        voxel_cloud->height = 1;
        voxel_cloud->is_dense = true;
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
    
    stored_cloud_->header.frame_id = "map";
   Eigen::Vector4f temp_min, temp_max;

    // 2. Extrai os limites da nuvem
    pcl::getMinMax3D(*stored_cloud_, temp_min, temp_max);

    // 3. Salva apenas o X, Y, Z nas variáveis seguras da sua classe
    min_pt_ = temp_min.head<3>();
    max_pt_ = temp_max.head<3>();

    // 4. Aplica o padding (escreva isso APENAS UMA VEZ)
    float padding = 0.01f;
    min_pt_.array() -= padding; 
    max_pt_.array() += padding;

    publishBest();
    auto t = this->now();
    sensor_msgs::msg::PointCloud2 m; 
    pcl::toROSMsg(*stored_cloud_, m); 
    m.header.stamp = t; 
    m.header.frame_id = "map"; 
    pub_cloud_->publish(m);
    
    all_candidates_ = generateMultiOrientedRays(min_pt_.head<3>(), max_pt_.head<3>(), grid_res_);
    
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
        voxel_checker_.build(collision_cloud_, 0.003f); 
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Ambiente de colisão vazio ou nulo!");
    }
    
    evaluateGrasps(target_environment);
    return;
}


std::vector<geometry_msgs::msg::Pose> GenerateGraspPoses::generateMultiOrientedRays(
    const Eigen::Vector3f& min_global, const Eigen::Vector3f& max_global, float res) 
{
    std::vector<geometry_msgs::msg::Pose> poses;
    ray_lengths.clear(); 

    // O comprimento do raio agora é fixo e igual à abertura máxima da garra
    float ray_len = max_gripper_width_;
    float inv_res = 1.0f / res;

    std::set<std::tuple<int, int, int>> occupied_voxels;

    for (const auto& pt : stored_cloud_->points)
    {
        int ix = std::floor(pt.x * inv_res);
        int iy = std::floor(pt.y * inv_res);
        int iz = std::floor(pt.z * inv_res);
        occupied_voxels.insert({ix, iy, iz});
    }

    auto add_pose = [&](float x, float y, float z, Eigen::Vector3f dir) 
    {
        geometry_msgs::msg::Pose p;
        
        // A lambda já faz o recuo! Se dir = {-1, 0, 0}, então:
        // p.position.x = x - (-1 * res) = x + res.
        // Ou seja, o raio nasce 1 voxel para fora e aponta para dentro.
        p.position.x = x - (dir.x() * res);
        p.position.y = y - (dir.y() * res);
        p.position.z = z - (dir.z() * res);
        
        Eigen::Quaternionf q; 
        q.setFromTwoVectors(Eigen::Vector3f::UnitZ(), dir); 
        p.orientation.x = q.x(); p.orientation.y = q.y(); 
        p.orientation.z = q.z(); p.orientation.w = q.w();
        
        poses.push_back(p);
        ray_lengths.push_back(ray_len);
    };

    // --- SETUP DOS MARKERS DOS VOXELS ---
    visualization_msgs::msg::Marker internal_voxels_marker;
    internal_voxels_marker.header.frame_id = "map";
    internal_voxels_marker.header.stamp = this->now();
    internal_voxels_marker.ns = "voxels_internos";
    internal_voxels_marker.id = 0;
    internal_voxels_marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
    internal_voxels_marker.action = visualization_msgs::msg::Marker::ADD;
    internal_voxels_marker.scale.x = res; 
    internal_voxels_marker.scale.y = res; 
    internal_voxels_marker.scale.z = res;
    internal_voxels_marker.color.r = 0.8f; 
    internal_voxels_marker.color.g = 0.8f; 
    internal_voxels_marker.color.b = 0.8f; 
    internal_voxels_marker.color.a = 0.25f; // Cinza transparente (interno)

    visualization_msgs::msg::Marker surface_voxels_marker;
    surface_voxels_marker.header.frame_id = "map";
    surface_voxels_marker.header.stamp = this->now();
    surface_voxels_marker.ns = "voxels_superficie_vermelhos";
    surface_voxels_marker.id = 1;
    surface_voxels_marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
    surface_voxels_marker.action = visualization_msgs::msg::Marker::ADD;
    surface_voxels_marker.scale.x = res; 
    surface_voxels_marker.scale.y = res; 
    surface_voxels_marker.scale.z = res;
    surface_voxels_marker.color.r = 1.0f; 
    surface_voxels_marker.color.g = 0.0f; 
    surface_voxels_marker.color.b = 0.0f; 
    surface_voxels_marker.color.a = 0.9f; // Vermelho sólido (superfície)

    // 2. GERAÇÃO DE RAIOS E CLASSIFICAÇÃO DOS VOXELS
    for (const auto& voxel : occupied_voxels) 
    {
        int ix = std::get<0>(voxel);
        int iy = std::get<1>(voxel);
        int iz = std::get<2>(voxel);

        // Centro do voxel atual
        float cx = (ix + 0.5f) * res;
        float cy = (iy + 0.5f) * res;
        float cz = (iz + 0.5f) * res;

        bool is_surface = false;

        // --- EIXO X ---
        // Passamos APENAS cx, cy, cz para o add_pose, porque a lambda já calcula o offset de recuo
        if (occupied_voxels.find({ix + 1, iy, iz}) == occupied_voxels.end()) {
            add_pose(cx, cy, cz, {-1, 0, 0}); 
            is_surface = true;
        }
        if (occupied_voxels.find({ix - 1, iy, iz}) == occupied_voxels.end()) {
            add_pose(cx, cy, cz, {1, 0, 0});  
            is_surface = true;
        }

        // --- EIXO Y ---
        if (occupied_voxels.find({ix, iy + 1, iz}) == occupied_voxels.end()) {
            add_pose(cx, cy, cz, {0, -1, 0}); 
            is_surface = true;
        }
        if (occupied_voxels.find({ix, iy - 1, iz}) == occupied_voxels.end()) {
            add_pose(cx, cy, cz, {0, 1, 0});  
            is_surface = true;
        }

        // --- EIXO Z ---
        if (occupied_voxels.find({ix, iy, iz + 1}) == occupied_voxels.end()) {
            add_pose(cx, cy, cz, {0, 0, -1}); 
            is_surface = true;
        }
        if (occupied_voxels.find({ix, iy, iz - 1}) == occupied_voxels.end()) {
            add_pose(cx, cy, cz, {0, 0, 1});  
            is_surface = true;
        }

        geometry_msgs::msg::Point pt;
        pt.x = cx; pt.y = cy; pt.z = cz;

        if (is_surface) {
            surface_voxels_marker.points.push_back(pt);
        } else {
            internal_voxels_marker.points.push_back(pt);
        }
    }

    // 3. VISUALIZAÇÃO
    // 3. VISUALIZAÇÃO
    if (pub_rays_) { 
        visualization_msgs::msg::MarkerArray marker_array;
        
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        if (!internal_voxels_marker.points.empty()) 
            marker_array.markers.push_back(internal_voxels_marker);
        if (!surface_voxels_marker.points.empty())  
            marker_array.markers.push_back(surface_voxels_marker);

        int ray_id = 2;
        for (size_t i = 0; i < poses.size(); ++i) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map"; 
            marker.header.stamp = this->now();
            marker.ns = "adaptive_rays";
            marker.id = ray_id++;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // Força a origem do Marker para 0,0,0 (usaremos os Points para ditar a posição e rotação)
            marker.pose.orientation.w = 1.0;

            // Ponto Inicial (Origem do raio recuada pelo tamanho do voxel)
            geometry_msgs::msg::Point p_start;
            p_start.x = poses[i].position.x;
            p_start.y = poses[i].position.y;
            p_start.z = poses[i].position.z;

            // Recria o vetor de direção a partir do Quaternion armazenado
            Eigen::Quaterniond q(poses[i].orientation.w, poses[i].orientation.x, poses[i].orientation.y, poses[i].orientation.z);
            Eigen::Vector3d ray_dir = q * Eigen::Vector3d::UnitZ();

            // Ponto Final (Origem + Direção * Comprimento do raio)
            geometry_msgs::msg::Point p_end;
            p_end.x = p_start.x + (ray_dir.x() * ray_lengths[i]);
            p_end.y = p_start.y + (ray_dir.y() * ray_lengths[i]);
            p_end.z = p_start.z + (ray_dir.z() * ray_lengths[i]);

            marker.points.push_back(p_start);
            marker.points.push_back(p_end);
            
            // Quando usamos "points" na ARROW:
            // scale.x é a espessura da haste, scale.y é a largura da ponta, scale.z é o tamanho da ponta
            marker.scale.x = 0.0015; 
            marker.scale.y = 0.003; 
            marker.scale.z = 0.003; 

            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 1.0; 
            marker.color.a = 0.6; 

            marker_array.markers.push_back(marker);
        }
        
        pub_rays_->publish(marker_array);
    }
    return poses;
}


StepAnalysis GenerateGraspPoses::analyzeLocalSphere(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& ray_dir)
{
    StepAnalysis result;
    result.valid = false;
    result.center = center;

    const size_t N = cloud->size();
    if (!cloud || N <= min_points_per_segment_) return result;
    result.point_count = N;

    const pcl::PointXYZ* __restrict__ pts = cloud->points.data();

    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        sx += pts[i].x; sy += pts[i].y; sz += pts[i].z;
    }
    const float inv_N = 1.0f / static_cast<float>(N);
    const float mx = sx * inv_N, my = sy * inv_N, mz = sz * inv_N;

    float acc_xx = 0.0f, acc_xy = 0.0f, acc_xz = 0.0f;
    float acc_yy = 0.0f, acc_yz = 0.0f, acc_zz = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        const float dx = pts[i].x - mx, dy = pts[i].y - my, dz = pts[i].z - mz;
        acc_xx += dx*dx; acc_xy += dx*dy; acc_xz += dx*dz;
        acc_yy += dy*dy; acc_yz += dy*dz; acc_zz += dz*dz;
    }

    Eigen::Matrix3f cov;
    cov(0,0)=acc_xx*inv_N; cov(0,1)=acc_xy*inv_N; cov(0,2)=acc_xz*inv_N;
    cov(1,0)=cov(0,1);     cov(1,1)=acc_yy*inv_N; cov(1,2)=acc_yz*inv_N;
    cov(2,0)=cov(0,2);     cov(2,1)=cov(1,2);     cov(2,2)=acc_zz*inv_N;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov, Eigen::ComputeEigenvectors);
    const Eigen::Matrix3f& ev = solver.eigenvectors();

    const Eigen::Vector3f& vals = solver.eigenvalues();
    const float eig_sum = vals[0] + vals[1] + vals[2];
    result.curvature = (eig_sum > 1e-6f) ? std::abs(vals[0]) / eig_sum : 0.0f;

    Eigen::Vector3f pca_normal = ev.col(0);
    if (pca_normal.dot(ray_dir) > 0.0f) pca_normal = -pca_normal;
    result.normal_vector = pca_normal;

    const float t1x=ev(0,1), t1y=ev(1,1), t1z=ev(2,1);
    const float t2x=ev(0,2), t2y=ev(1,2), t2z=ev(2,2);
    const float ox=center.x(), oy=center.y(), oz=center.z();

    // Limite radial: metade do raio
    const float safe_r = (sphere_radius_ > 1e-4f) ? sphere_radius_ : 0.02f;
    const float r2_half = (safe_r * 0.5f) * (safe_r * 0.5f);

    constexpr int NUM_SECTORS = 16; // 8 angulares × 2 anéis radiais
    int sector_counts[NUM_SECTORS] = {0};

    for (size_t i = 0; i < N; ++i) {
        const float dx = pts[i].x - ox, dy = pts[i].y - oy, dz = pts[i].z - oz;
        const float u = dx*t1x + dy*t1y + dz*t1z;
        const float v = dx*t2x + dy*t2y + dz*t2z;
        const float r2 = u*u + v*v;
        const int ring = (r2 >= r2_half);  // bit 3: 0 = interno (centro→meio), 1 = externo (meio→raio)
        const int sector = (u >= 0.0f) | ((v >= 0.0f) << 1) | ((u*u >= v*v) << 2) | (ring << 3);
        sector_counts[sector]++;
    }

    const int threshold = static_cast<int>(min_points_per_segment_);
    int populated = 0;
    for (int k = 0; k < NUM_SECTORS; ++k)
        populated += (sector_counts[k] >= threshold);
    result.symmetry_score = static_cast<float>(populated) / static_cast<float>(NUM_SECTORS);

    const float dot_val = std::min(std::abs(ray_dir.dot(result.normal_vector)), 1.0f);
    result.angle_to_normal_deg = std::acos(dot_val) * 57.2957795f;

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

    // MATRIZ ÚNICA: Aplicada usando os parâmetros do ROS.
    Eigen::Affine3f visual_tf = Eigen::Affine3f::Identity();
    Eigen::Matrix3f rot_geom;
    rot_geom = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
             * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ())
             * Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX());
    visual_tf.linear() = rot_geom;
    visual_tf.translation() = Eigen::Vector3f(mesh_offset_x_, mesh_offset_y_, mesh_offset_z_);

    std::vector<std::tuple<uint8_t, uint8_t, uint8_t>> colors = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255}, {255, 0, 255}
    };

    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];
        float min_x = std::numeric_limits<float>::max(); float min_y = std::numeric_limits<float>::max(); float min_z = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest(); float max_y = std::numeric_limits<float>::lowest(); float max_z = std::numeric_limits<float>::lowest();

        auto color = colors[i % colors.size()];
        uint8_t r = std::get<0>(color); uint8_t g = std::get<1>(color); uint8_t b = std::get<2>(color);

        for (unsigned int v = 0; v < mesh->mNumVertices; v++) 
        {
            Eigen::Vector3f p_scaled(
                mesh->mVertices[v].x * gripper_mesh_scale_,
                mesh->mVertices[v].y * gripper_mesh_scale_,
                mesh->mVertices[v].z * gripper_mesh_scale_
            );

            // ASSANDO A TRANSFORMAÇÃO NO VÉRTICE. A CAIXA JÁ NASCE CORRETA.
            Eigen::Vector3f p_local = visual_tf * p_scaled;

            if (p_local.x() < min_x) min_x = p_local.x();
            if (p_local.y() < min_y) min_y = p_local.y();
            if (p_local.z() < min_z) min_z = p_local.z();
            if (p_local.x() > max_x) max_x = p_local.x();
            if (p_local.y() > max_y) max_y = p_local.y();
            if (p_local.z() > max_z) max_z = p_local.z();
            
            pcl::PointXYZRGB pt_rgb;
            pt_rgb.x = p_local.x(); pt_rgb.y = p_local.y(); pt_rgb.z = p_local.z();
            pt_rgb.r = r; pt_rgb.g = g; pt_rgb.b = b;
            gripper_dense_cloud_->points.push_back(pt_rgb);
        }

        LocalBox box;
        float margin = 0.005f;
        box.min_pt = Eigen::Vector3f(min_x - margin, min_y - margin, min_z - margin);
        box.max_pt = Eigen::Vector3f(max_x + margin, max_y + margin, max_z + margin);
        
        box.center = (box.min_pt + box.max_pt) / 2.0f;
        box.dimensions = box.max_pt - box.min_pt;

        gripper_boxes_.push_back(box);
    }

    
    
    gripper_dense_cloud_->width = gripper_dense_cloud_->points.size();
    gripper_dense_cloud_->height = 1;
    gripper_dense_cloud_->is_dense = true;


    // =====================================================================
    // LOG DE DEBUG: PRINTAR A POSE DAS 3 CAIXAS (CENTRO E DIMENSÕES)
    // =====================================================================
    RCLCPP_INFO(this->get_logger(), " ");
    RCLCPP_INFO(this->get_logger(), "=========================================================");
    RCLCPP_INFO(this->get_logger(), "======= DEBUG: COORDENADAS REAIS DAS CAIXAS BASE ========");
    RCLCPP_INFO(this->get_logger(), "=========================================================");
    
    for (size_t b = 0; b < gripper_boxes_.size(); b++)
    {
        const auto& box = gripper_boxes_[b];
        std::string nome_caixa = (b == 0) ? "Base/Palma" : (b == 1 ? "Dedo 1" : "Dedo 2");
        
        RCLCPP_INFO(this->get_logger(), 
            "[%s] -> POSIÇÃO (Centro): X: %.4f | Y: %.4f | Z: %.4f", 
            nome_caixa.c_str(), box.center.x(), box.center.y(), box.center.z()
        );
        
        RCLCPP_INFO(this->get_logger(), 
            "[%s] -> DIMENSÕES       : X: %.4f | Y: %.4f | Z: %.4f", 
            nome_caixa.c_str(), box.dimensions.x(), box.dimensions.y(), box.dimensions.z()
        );
        RCLCPP_INFO(this->get_logger(), "---------------------------------------------------------");
    }
    RCLCPP_INFO(this->get_logger(), "As caixas acima usam ORIENTAÇÃO IDENTIDADE (x:0, y:0, z:0, w:1)");
    RCLCPP_INFO(this->get_logger(), "=========================================================");   
}

void GenerateGraspPoses::publishGripperModel()
{
    if (!gripper_dense_cloud_ || gripper_dense_cloud_->empty()) return;
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*gripper_dense_cloud_, msg);
    msg.header.frame_id = "map"; 
    msg.header.stamp = this->now();
    pub_gripper_model_->publish(msg);
}

void GenerateGraspPoses::publishGripperCollisionBoxes()
{
    if (gripper_boxes_.empty()) return;

    visualization_msgs::msg::MarkerArray ma;
    auto t = this->now();

    for(size_t i = 0; i < gripper_boxes_.size(); i++)
    {
        const auto& box = gripper_boxes_[i];

        visualization_msgs::msg::Marker m;
        m.header.frame_id = "map";
        m.header.stamp = t;
        m.ns = "gripper_collision_boxes_base";
        m.id = i;
        m.type = visualization_msgs::msg::Marker::CUBE;
        m.action = visualization_msgs::msg::Marker::ADD;

        // USA O CENTRO PURO (Ele já tem a matriz assada dentro dele)
        m.pose.position.x = box.center.x();
        m.pose.position.y = box.center.y();
        m.pose.position.z = box.center.z();

        // ROTAÇÃO ZERO (A caixa já foi alinhada na extração)
        m.pose.orientation.x = 0.0;
        m.pose.orientation.y = 0.0;
        m.pose.orientation.z = 0.0;
        m.pose.orientation.w = 1.0;

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

bool GenerateGraspPoses::check_collision(ScoredGrasp& grasp, 
    const pcl::KdTreeFLANN<pcl::PointXYZ>& /*env_kdtree*/, 
    bool /*publish_debug*/, bool try_rotations)
{
    if (gripper_boxes_.empty()) return true;
    if (!voxel_checker_.isReady()) return true;

    Eigen::Vector3f grasp_pos(
        grasp.pose_center.position.x, 
        grasp.pose_center.position.y, 
        grasp.pose_center.position.z);
    Eigen::Quaternionf original_rot(
        grasp.pose_center.orientation.w, 
        grasp.pose_center.orientation.x, 
        grasp.pose_center.orientation.y, 
        grasp.pose_center.orientation.z);

    const int NUM_STEPS = try_rotations ? 18 : 1;
    const float ANGLE_STEP = (2.0f * M_PI) / 18.0f;
    const float MARGIN = 0.003f;

    for (int step = 0; step < NUM_STEPS; ++step)
    {
        float angle = step * ANGLE_STEP;
        Eigen::Quaternionf rot_offset(
            Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitY()));
        Eigen::Quaternionf current_rot = original_rot * rot_offset;

        Eigen::Affine3f tf_tcp_to_world = 
            Eigen::Translation3f(grasp_pos) * current_rot;
        Eigen::Affine3f tf_world_to_tcp = tf_tcp_to_world.inverse();

        if (!voxel_checker_.gripperCollides(
                tf_tcp_to_world, tf_world_to_tcp, 
                gripper_boxes_, MARGIN))
        {
            // Sem colisão neste ângulo → atualiza orientação e retorna válido
            grasp.pose_center.orientation.w = current_rot.w();
            grasp.pose_center.orientation.x = current_rot.x();
            grasp.pose_center.orientation.y = current_rot.y();
            grasp.pose_center.orientation.z = current_rot.z();
            return true;
        }
    }
    return false; // Todas as rotações colidem
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
        // O eixo X passa a ser o produto vetorial entre Y (dedos) e Z (approach)
        Eigen::Vector3f candidate_x = finger_axis.cross(candidate_approach);

        Eigen::Matrix3f rot_mat;
        rot_mat.col(0) = candidate_x;        // Eixo X (perpendicular)
        rot_mat.col(1) = finger_axis;        // Eixo Y (eixo de abertura dos dedos)
        rot_mat.col(2) = candidate_approach; // Eixo Z (direção de approach / end-effector) 

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
   
    struct VoxelBucket {
        std::vector<pcl::PointXYZ> points;
        Eigen::Vector3f center;
        bool has_points = false; 
        bool is_near_surface = false;
        std::vector<uint32_t> nearby_filled_indices; 
    };

    auto format_clocks = [](uint64_t n) -> std::string {
        std::string s = std::to_string(n);
        int insertPosition = static_cast<int>(s.length()) - 3;
        while (insertPosition > 0) { 
            s.insert(insertPosition, "."); 
            insertPosition -= 3; 
        }
        return s;
    };

    
    auto t_func_start = std::chrono::high_resolution_clock::now();
    uint64_t c_func_start = __rdtsc();

    hit_candidates_.clear(); 
    
 
    auto t_kdtree_start = std::chrono::high_resolution_clock::now();
    

    pcl::KdTreeFLANN<pcl::PointXYZ> env_kdtree;
    if (target_environment->empty()) {
        RCLCPP_WARN(this->get_logger(), "Ambiente vazio, ignorando colisão.");
    } else {
        env_kdtree.setInputCloud(target_environment);
    }
    
    auto t_kdtree_end = std::chrono::high_resolution_clock::now();
    double d_kdtree = std::chrono::duration<double, std::milli>(t_kdtree_end - t_kdtree_start).count();
    
    auto t_voxel_start = std::chrono::high_resolution_clock::now();

   
    Eigen::Vector3f min_bound(1e9, 1e9, 1e9);
    Eigen::Vector3f max_bound(-1e9, -1e9, -1e9);

    for (const auto& pt : stored_cloud_->points) {
        if (pt.x < min_bound.x()) min_bound.x() = pt.x;
        if (pt.y < min_bound.y()) min_bound.y() = pt.y;
        if (pt.z < min_bound.z()) min_bound.z() = pt.z;
        if (pt.x > max_bound.x()) max_bound.x() = pt.x;
        if (pt.y > max_bound.y()) max_bound.y() = pt.y;
        if (pt.z > max_bound.z()) max_bound.z() = pt.z;
    }

    float voxel_size = 0.01f; 
    float padding = voxel_size * 2.0f;
    min_bound -= Eigen::Vector3f(padding, padding, padding);
    max_bound += Eigen::Vector3f(padding, padding, padding);

    float inv_voxel_size = 1.0f / voxel_size;
    int dim_x = std::ceil((max_bound.x() - min_bound.x()) * inv_voxel_size);
    int dim_y = std::ceil((max_bound.y() - min_bound.y()) * inv_voxel_size);
    int dim_z = std::ceil((max_bound.z() - min_bound.z()) * inv_voxel_size);

    int stride_y = dim_x;
    int stride_z = dim_x * dim_y;
    size_t total_voxels = (size_t)dim_x * dim_y * dim_z;

    std::vector<VoxelBucket> linear_grid(total_voxels);
    
    // 3.3. Popular o Grid
    for (const auto& pt : stored_cloud_->points) 
    {
        int ix = std::floor((pt.x - min_bound.x()) * inv_voxel_size); 
        int iy = std::floor((pt.y - min_bound.y()) * inv_voxel_size); 
        int iz = std::floor((pt.z - min_bound.z()) * inv_voxel_size);
        
        if (ix >= 0 && ix < dim_x && iy >= 0 && iy < dim_y && iz >= 0 && iz < dim_z)
        {
            size_t flat_idx = ix + iy * stride_y + iz * stride_z;
            VoxelBucket& bucket = linear_grid[flat_idx];
            
            if (!bucket.has_points) {
                bucket.center = Eigen::Vector3f(
                    min_bound.x() + (ix + 0.5f) * voxel_size,
                    min_bound.y() + (iy + 0.5f) * voxel_size,
                    min_bound.z() + (iz + 0.5f) * voxel_size
                );
                bucket.has_points = true;
                bucket.points.reserve(16); 
                bucket.is_near_surface = true;
                bucket.nearby_filled_indices.push_back(flat_idx); 
            }
            bucket.points.push_back(pt);
        }
    }

    int expansion_rad = std::ceil(sphere_radius_ * inv_voxel_size);
    int expansion_rad_sq = expansion_rad * expansion_rad;

    std::vector<std::tuple<int, int, int>> sphere_offsets;
    sphere_offsets.reserve((2*expansion_rad+1)*(2*expansion_rad+1)*(2*expansion_rad+1));

    for(int dx = -expansion_rad; dx <= expansion_rad; ++dx) {
        for(int dy = -expansion_rad; dy <= expansion_rad; ++dy) {
            for(int dz = -expansion_rad; dz <= expansion_rad; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                if (dx*dx + dy*dy + dz*dz <= expansion_rad_sq) {
                    sphere_offsets.emplace_back(dx, dy, dz);
                }
            }
        }
    }

    for (int iz = 0; iz < dim_z; ++iz) {
        for (int iy = 0; iy < dim_y; ++iy) {
            for (int ix = 0; ix < dim_x; ++ix) {
                size_t current_idx = ix + iy * stride_y + iz * stride_z;
                if (linear_grid[current_idx].has_points) {
                    for (const auto& offset : sphere_offsets) {
                        int nx = ix + std::get<0>(offset);
                        int ny = iy + std::get<1>(offset);
                        int nz = iz + std::get<2>(offset);

                        if (nx >= 0 && nx < dim_x && ny >= 0 && ny < dim_y && nz >= 0 && nz < dim_z) {
                            size_t neighbor_idx = nx + ny * stride_y + nz * stride_z;
                            linear_grid[neighbor_idx].is_near_surface = true;
                            linear_grid[neighbor_idx].nearby_filled_indices.push_back(current_idx);
                        }
                    }
                }
            }
        }
    }

    auto t_voxel_end = std::chrono::high_resolution_clock::now();
    
   
    struct ThreadLocalData {
        std::vector<uint32_t> visited_timestamps;
        uint32_t current_query_id = 0;

        std::vector<const VoxelBucket*> voxels_no_caminho;
        pcl::PointCloud<pcl::PointXYZ>::Ptr init_inliers;
        pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_entry;
        pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_exit;

        std::vector<ScoredGrasp> local_candidates;
        std::vector<geometry_msgs::msg::Pose> local_hits;

        double acc_inliers_ms = 0.0;
        double acc_analysis_ms = 0.0;
        double acc_scoring_ms = 0.0;

        ThreadLocalData(size_t grid_size) : visited_timestamps(grid_size, 0) {
            voxels_no_caminho.reserve(100);
            init_inliers.reset(new pcl::PointCloud<pcl::PointXYZ>); init_inliers->reserve(100);
            inliers_entry.reset(new pcl::PointCloud<pcl::PointXYZ>); inliers_entry->reserve(500);
            inliers_exit.reset(new pcl::PointCloud<pcl::PointXYZ>);  inliers_exit->reserve(500);
        }
    };

    tbb::enumerable_thread_specific<ThreadLocalData> tls([&]() {
        return ThreadLocalData(total_voxels);
    });

    std::atomic<int> atomic_perfect_count{0};

    float voxel_radius = (voxel_size * 1.73205f) / 2.0f;
    float voxel_check_threshold = sphere_radius_ + voxel_radius; 
    float voxel_check_threshold_squared = voxel_check_threshold * voxel_check_threshold;
    float cylinder_radius_sq = sphere_radius_ * sphere_radius_;
    float max_scan_dist = max_gripper_width_ * 1.1f;
    int r_int = std::ceil(sphere_radius_ * inv_voxel_size);
    int r_sq = r_int * r_int;

    const float voxel_extent = (voxel_size * 1.73205f) * 0.5f; 
    const float safe_radius_sq = (sphere_radius_ - voxel_extent) * (sphere_radius_ - voxel_extent);
    const bool use_fast_check = (sphere_radius_ > voxel_extent); 

    RCLCPP_INFO(this->get_logger(), "Iniciando Processamento PARALELO (Candidates: %lu)...", all_candidates_.size());
    auto t_loop_start = std::chrono::high_resolution_clock::now();

    tbb::global_control global_thread_limit(tbb::global_control::max_allowed_parallelism, 16);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, all_candidates_.size()), 
        [&](const tbb::blocked_range<size_t>& range) 
    {
        
        ThreadLocalData& local = tls.local();

        for (size_t i = range.begin(); i != range.end(); ++i) 
        {
            
            if (atomic_perfect_count.load() >= num_best_grasps_) {
                continue; 
            }

            const auto& raw_pose = all_candidates_[i]; 
            
           
            Eigen::Quaternionf q_start(raw_pose.orientation.w, raw_pose.orientation.x, raw_pose.orientation.y, raw_pose.orientation.z);
            Eigen::Vector3f ray_origin_start(raw_pose.position.x, raw_pose.position.y, raw_pose.position.z);
            Eigen::Vector3f ray_dir_start = q_start * Eigen::Vector3f::UnitZ();

            auto t0 = std::chrono::high_resolution_clock::now();

            
            float t_min_init = 1e6f;
            float t_max_init = -1e6f;
            Eigen::Vector3f PIVOT_POINT = {0.0f, 0.0f, 0.0f}; 
            bool hit_init = false;
            local.init_inliers->clear();
            local.voxels_no_caminho.clear();
            bool continuar = false;

            
            Eigen::Vector3f dir = ray_dir_start;
            if (std::abs(dir.x()) < 1e-9f) dir.x() = 1e-9f;
            if (std::abs(dir.y()) < 1e-9f) dir.y() = 1e-9f;
            if (std::abs(dir.z()) < 1e-9f) dir.z() = 1e-9f;

            Eigen::Vector3f relative_origin = ray_origin_start - min_bound;
            int cur_x = std::floor(relative_origin.x() * inv_voxel_size);
            int cur_y = std::floor(relative_origin.y() * inv_voxel_size);
            int cur_z = std::floor(relative_origin.z() * inv_voxel_size);

            int step_x = (dir.x() > 0) ? 1 : -1;
            int step_y = (dir.y() > 0) ? 1 : -1;
            int step_z = (dir.z() > 0) ? 1 : -1;

            float tMaxX = ((cur_x + (step_x > 0 ? 1 : 0)) * voxel_size - relative_origin.x()) / dir.x();
            float tMaxY = ((cur_y + (step_y > 0 ? 1 : 0)) * voxel_size - relative_origin.y()) / dir.y();
            float tMaxZ = ((cur_z + (step_z > 0 ? 1 : 0)) * voxel_size - relative_origin.z()) / dir.z();

            float tDeltaX = std::abs(voxel_size / dir.x());
            float tDeltaY = std::abs(voxel_size / dir.y());
            float tDeltaZ = std::abs(voxel_size / dir.z());

     
            local.current_query_id++;
            if (local.current_query_id == 0) {
                std::fill(local.visited_timestamps.begin(), local.visited_timestamps.end(), 0);
                local.current_query_id = 1;
            }

            
            while (true)
            {
                if (std::min({tMaxX, tMaxY, tMaxZ}) - std::min({tDeltaX, tDeltaY, tDeltaZ}) > max_scan_dist) break;

                for (int dx = -r_int; dx <= r_int; ++dx) {
                    for (int dy = -r_int; dy <= r_int; ++dy) {
                        for (int dz = -r_int; dz <= r_int; ++dz) {
                            if (dx*dx + dy*dy + dz*dz > r_sq) continue;

                            int nx = cur_x + dx;
                            int ny = cur_y + dy;
                            int nz = cur_z + dz;

                            if (nx >= 0 && nx < dim_x && ny >= 0 && ny < dim_y && nz >= 0 && nz < dim_z) {
                                size_t flat_idx = nx + ny * stride_y + nz * stride_z;
                                if (local.visited_timestamps[flat_idx] != local.current_query_id) {
                                    local.visited_timestamps[flat_idx] = local.current_query_id;
                                    const auto& bucket = linear_grid[flat_idx];
                                    if (bucket.has_points) {
                                        local.voxels_no_caminho.push_back(&bucket);
                                    }
                                }
                            }
                        }
                    }
                }

                if (tMaxX < tMaxY) {
                    if (tMaxX < tMaxZ) { cur_x += step_x; tMaxX += tDeltaX; } else { cur_z += step_z; tMaxZ += tDeltaZ; }
                } else {
                    if (tMaxY < tMaxZ) { cur_y += step_y; tMaxY += tDeltaY; } else { cur_z += step_z; tMaxZ += tDeltaZ; }
                }
            }

            
            for (const auto* bucket_ptr : local.voxels_no_caminho) 
            {
                const auto& bucket = *bucket_ptr;
                Eigen::Vector3f diff = bucket.center - ray_origin_start;
                if ((diff.cross(ray_dir_start)).squaredNorm() > voxel_check_threshold_squared) continue;

                for (const auto& pt : bucket.points) {
                    Eigen::Vector3f p(pt.x, pt.y, pt.z);
                    float t = (p - ray_origin_start).dot(ray_dir_start);
                    Eigen::Vector3f dist_vec = p - (ray_origin_start + t * ray_dir_start);
                    
                    if (dist_vec.squaredNorm() < cylinder_radius_sq) {
                        if (t < t_min_init) { t_min_init = t; PIVOT_POINT = p; }
                        if (t > t_max_init) { t_max_init = t; }
                        hit_init = true;
                        local.init_inliers->points.push_back(pt);
                    }
                }
                if(t_max_init - t_min_init >= max_gripper_width_) { continuar = true; break; }
            }
            if(continuar) continue;

            auto t1 = std::chrono::high_resolution_clock::now(); 
            local.acc_inliers_ms += std::chrono::duration<double, std::milli>(t1 - t0).count(); 

            float init_thickness = t_max_init - t_min_init;
            if (!hit_init || local.init_inliers->size() < 3 || 
                init_thickness < 0.0005 || init_thickness > max_gripper_width_) continue;

           
            auto t2_pre = std::chrono::high_resolution_clock::now();
            Eigen::Vector4f centroid; 
            Eigen::Matrix3f covariance_matrix;
            pcl::compute3DCentroid(*local.init_inliers, centroid);
            pcl::computeCovarianceMatrixNormalized(*local.init_inliers, centroid, covariance_matrix);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
            Eigen::Vector3f pca_normal = eigen_solver.eigenvectors().col(0);
            
            if (!pca_normal.allFinite()) pca_normal = -ray_dir_start; 
            else if (pca_normal.dot(ray_dir_start) > 0) pca_normal = -pca_normal;

            const float DISTANCE_TO_PIVOT = t_min_init;                                        
            const Eigen::Vector3f DIR_START = ray_dir_start;                                   
            const Eigen::Vector3f DIR_TARGET = -pca_normal;     
            
            float dot_prod = DIR_START.dot(DIR_TARGET);
            
            
            if (dot_prod > 1.0f) dot_prod = 1.0f;
            if (dot_prod < -1.0f) dot_prod = -1.0f;

            float angle_diff_rad = std::acos(dot_prod);
            float angle_diff_deg = angle_diff_rad * (180.0f / M_PI);

          
            // RCLCPP_INFO(this->get_logger(), "[Cand %lu] Angle Diff: %.2f deg", i, angle_diff_deg);

            

            auto t2_post = std::chrono::high_resolution_clock::now(); 
            local.acc_analysis_ms += std::chrono::duration<double, std::milli>(t2_post - t2_pre).count();

            
            ScoredGrasp best_iter_grasp;
            best_iter_grasp.total_score = -1.0; 
            bool found_valid_in_optimization = false;
            bool perfect_candidate_found = false;                          
            int max_optimization_steps = 10; 

            for (int step = 0; step < max_optimization_steps; ++step)
            {
                auto t3 = std::chrono::high_resolution_clock::now();
                float t_lerp = (float)step / (float)max_optimization_steps; 
                Eigen::Vector3f current_ray_dir = ((1.0f - t_lerp) * DIR_START + t_lerp * DIR_TARGET).normalized();
                Eigen::Vector3f current_ray_origin = PIVOT_POINT - (current_ray_dir * DISTANCE_TO_PIVOT);

                local.voxels_no_caminho.clear();
                
               
                {
                    Eigen::Vector3f rel_pivot = PIVOT_POINT - min_bound;
                    int pv_x = std::floor(rel_pivot.x() * inv_voxel_size);
                    int pv_y = std::floor(rel_pivot.y() * inv_voxel_size);
                    int pv_z = std::floor(rel_pivot.z() * inv_voxel_size);
                    if (pv_x >= 0 && pv_x < dim_x && pv_y >= 0 && pv_y < dim_y && pv_z >= 0 && pv_z < dim_z) {
                        size_t pivot_idx = pv_x + pv_y * stride_y + pv_z * stride_z;
                        local.voxels_no_caminho.push_back(&linear_grid[pivot_idx]);
                    }
                }

               
                Eigen::Vector3f ray_end_pos = current_ray_origin + (current_ray_dir * max_scan_dist);
                Eigen::Vector3f traverse_dir = -current_ray_dir; 
                if (std::abs(traverse_dir.x()) < 1e-9f) traverse_dir.x() = 1e-9f;
                if (std::abs(traverse_dir.y()) < 1e-9f) traverse_dir.y() = 1e-9f;
                if (std::abs(traverse_dir.z()) < 1e-9f) traverse_dir.z() = 1e-9f;

                Eigen::Vector3f rel_start = ray_end_pos - min_bound;
                int cur_x = std::floor(rel_start.x() * inv_voxel_size);
                int cur_y = std::floor(rel_start.y() * inv_voxel_size);
                int cur_z = std::floor(rel_start.z() * inv_voxel_size);

                int step_x = (traverse_dir.x() > 0) ? 1 : -1;
                int step_y = (traverse_dir.y() > 0) ? 1 : -1;
                int step_z = (traverse_dir.z() > 0) ? 1 : -1;

                float tMaxX = ((cur_x + (step_x > 0 ? 1 : 0)) * voxel_size - rel_start.x()) / traverse_dir.x();
                float tMaxY = ((cur_y + (step_y > 0 ? 1 : 0)) * voxel_size - rel_start.y()) / traverse_dir.y();
                float tMaxZ = ((cur_z + (step_z > 0 ? 1 : 0)) * voxel_size - rel_start.z()) / traverse_dir.z();

                float tDeltaX = std::abs(voxel_size / traverse_dir.x());
                float tDeltaY = std::abs(voxel_size / traverse_dir.y());
                float tDeltaZ = std::abs(voxel_size / traverse_dir.z());

                bool voxel_hit_any = false;
                
                while (true)
                {
                    float t_curr = std::min({tMaxX, tMaxY, tMaxZ}) - std::min({tDeltaX, tDeltaY, tDeltaZ});
                    if (t_curr > max_scan_dist) break; 
                    
                    if (cur_x >= 0 && cur_x < dim_x && cur_y >= 0 && cur_y < dim_y && cur_z >= 0 && cur_z < dim_z)
                    {
                        size_t current_flat_idx = cur_x + cur_y * stride_y + cur_z * stride_z;
                        const auto& bucket_curr = linear_grid[current_flat_idx];

                        if (bucket_curr.is_near_surface && bucket_curr.has_points) {
                            local.voxels_no_caminho.push_back(&bucket_curr);
                            voxel_hit_any = true;
                            break; 
                        }
                    } 

                    if (tMaxX < tMaxY) {
                        if (tMaxX < tMaxZ) { cur_x += step_x; tMaxX += tDeltaX; } else { cur_z += step_z; tMaxZ += tDeltaZ; }
                    } else {
                        if (tMaxY < tMaxZ) { cur_y += step_y; tMaxY += tDeltaY; } else { cur_z += step_z; tMaxZ += tDeltaZ; }
                    }
                }

                if (!voxel_hit_any && local.voxels_no_caminho.size() <= 1) continue; 

                
                local.inliers_entry->clear();
                local.inliers_exit->clear();

                const auto* bucket_min = local.voxels_no_caminho.front();
                float exact_t_min = 1e9f;
                bool points_found = false;
                float c_ox = current_ray_origin.x(), c_oy = current_ray_origin.y(), c_oz = current_ray_origin.z();
                float c_dx = current_ray_dir.x(), c_dy = current_ray_dir.y(), c_dz = current_ray_dir.z();

                for(const auto& pt : bucket_min->points) {
                    float vx = pt.x - c_ox, vy = pt.y - c_oy, vz = pt.z - c_oz;
                    float t = vx * c_dx + vy * c_dy + vz * c_dz;
                    if (t < exact_t_min) { exact_t_min = t; points_found = true; }
                }
                if (!points_found) continue; 

                const auto* bucket_max = local.voxels_no_caminho.back();
                float exact_t_max = -1e9f;
                bool exit_found = false; 
                for(const auto& pt : bucket_max->points) {
                    float vx = pt.x - c_ox, vy = pt.y - c_oy, vz = pt.z - c_oz;
                    float t = vx * c_dx + vy * c_dy + vz * c_dz;
                    if (t > exact_t_max) { exact_t_max = t; exit_found = true; }
                }
                if (!exit_found && exact_t_max < -1e8f) exact_t_max = exact_t_min; 

               
                Eigen::Vector3f center_entry = current_ray_origin + current_ray_dir * exact_t_min;
                Eigen::Vector3f center_exit  = current_ray_origin + current_ray_dir * exact_t_max;

                auto collect_spherical = [&](const Eigen::Vector3f& center, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out) 
                {
                    
                    int ix = std::floor((center.x() - min_bound.x()) * inv_voxel_size);
                    int iy = std::floor((center.y() - min_bound.y()) * inv_voxel_size);
                    int iz = std::floor((center.z() - min_bound.z()) * inv_voxel_size);

                    if (ix >= 0 && ix < dim_x && iy >= 0 && iy < dim_y && iz >= 0 && iz < dim_z) {
                        size_t idx = ix + iy * stride_y + iz * stride_z;
                        const auto& center_bucket = linear_grid[idx];

                        if (!center_bucket.nearby_filled_indices.empty()) {
                            
                            
                            size_t total_candidates = 0;
                            for (const auto& neighbor_idx : center_bucket.nearby_filled_indices) {
                                total_candidates += linear_grid[neighbor_idx].points.size();
                            }

                            
                            cloud_out->points.reserve(cloud_out->points.size() + total_candidates);
                           

                            
                            const float cx = center.x();
                            const float cy = center.y();
                            const float cz = center.z();

                            for (const auto& neighbor_idx : center_bucket.nearby_filled_indices) 
                            {
                                const auto& neighbor_bucket = linear_grid[neighbor_idx];

                                for (const auto& pt : neighbor_bucket.points) 
                                {
                                    float dx = pt.x - cx;
                                    float dy = pt.y - cy;
                                    float dz = pt.z - cz;

                                    float dist_sq = (dx * dx) + (dy * dy) + (dz * dz);

                                    if (dist_sq <= cylinder_radius_sq) 
                                    {
                                        cloud_out->points.push_back(pt);
                                    }
                                }
                            }
                        }
                    }
                };

                collect_spherical(center_entry, local.inliers_entry);
                if ((exact_t_max - exact_t_min) > 0.005f) collect_spherical(center_exit, local.inliers_exit);

                auto t4 = std::chrono::high_resolution_clock::now();
                local.acc_inliers_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();

                float real_thickness = exact_t_max - exact_t_min;
                if (!points_found || local.inliers_entry->size() < 5 || local.inliers_exit->size() < 5 || real_thickness >= max_gripper_width_) continue; 

                if (enable_ray_animation_ && debug_marker_pub_)
                {
                    std::lock_guard<std::mutex> lock(toma);
                    auto t_now = this->now();
                    visualization_msgs::msg::MarkerArray markers;

                    // Limpa marcadores da iteração anterior
                    visualization_msgs::msg::Marker clear_marker;
                    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
                    markers.markers.push_back(clear_marker);

                    // Publica a nuvem original de contexto
                    sensor_msgs::msg::PointCloud2 m; 
                    pcl::toROSMsg(*stored_cloud_, m); 
                    m.header.stamp = t_now; 
                    m.header.frame_id = "map"; 
                    pub_cloud_->publish(m);

                    visualization_msgs::msg::Marker base_marker;
                    base_marker.header.frame_id = "map";
                    base_marker.header.stamp = t_now;
                    base_marker.action = visualization_msgs::msg::Marker::ADD;
                    base_marker.pose.orientation.w = 1.0;
                    base_marker.lifetime = rclcpp::Duration::from_seconds(0); 

                    StepAnalysis debug_entry = analyzeLocalSphere(local.inliers_entry, center_entry, current_ray_dir);
                    StepAnalysis debug_exit;
                    if (real_thickness > 0.001f) {
                        debug_exit = analyzeLocalSphere(local.inliers_exit, center_exit, current_ray_dir);
                    }

                    // 1. ORIGEM DO RAIO (Esfera Magenta)
                    visualization_msgs::msg::Marker origin_mk = base_marker;
                    origin_mk.ns = "anim_ray_origin"; origin_mk.id = 1;
                    origin_mk.type = visualization_msgs::msg::Marker::SPHERE;
                    origin_mk.pose.position.x = current_ray_origin.x(); 
                    origin_mk.pose.position.y = current_ray_origin.y(); 
                    origin_mk.pose.position.z = current_ray_origin.z();
                    origin_mk.scale.x = 0.005; origin_mk.scale.y = 0.005; origin_mk.scale.z = 0.005;
                    origin_mk.color.a = 1.0; origin_mk.color.r = 1.0; origin_mk.color.g = 0.0; origin_mk.color.b = 1.0; 
                    markers.markers.push_back(origin_mk);

                    // 2. RAIO PRINCIPAL (Seta Amarela Atravessando)
                    visualization_msgs::msg::Marker ray_mk = base_marker;
                    ray_mk.ns = "anim_ray"; ray_mk.id = 2;
                    ray_mk.type = visualization_msgs::msg::Marker::ARROW;
                    geometry_msgs::msg::Point p_start, p_end;
                    p_start.x = current_ray_origin.x(); 
                    p_start.y = current_ray_origin.y(); 
                    p_start.z = current_ray_origin.z();
                    // O raio vai um pouco além do ponto de saída para mostrar o vazamento
                    Eigen::Vector3f visual_end = current_ray_origin + (current_ray_dir * (exact_t_max + 0.03f)); 
                    p_end.x = visual_end.x(); p_end.y = visual_end.y(); p_end.z = visual_end.z();
                    ray_mk.points.push_back(p_start); ray_mk.points.push_back(p_end);
                    ray_mk.scale.x = 0.002; ray_mk.scale.y = 0.005; ray_mk.scale.z = 0.005;  
                    ray_mk.color.a = 1.0; ray_mk.color.r = 1.0; ray_mk.color.g = 1.0; ray_mk.color.b = 0.2;
                    markers.markers.push_back(ray_mk);

                    // --- FUNÇÃO AUXILIAR PARA RENDERIZAR O PONTO DE CONTATO (ESTILO DEBUG) ---
                    auto drawContact = [&](const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                           const Eigen::Vector3f& center, const StepAnalysis& analysis,
                                           const std::string& prefix, int base_id, 
                                           float r, float g, float b, float norm_r, float norm_g, float norm_b) 
                    {
                        // A. Pontos Inliers (Cubos 3D para melhor visibilidade)
                        if (cloud && !cloud->empty()) {
                            visualization_msgs::msg::Marker pts_mk = base_marker;
                            pts_mk.ns = prefix + "_inliers"; pts_mk.id = base_id + 1;
                            pts_mk.type = visualization_msgs::msg::Marker::CUBE_LIST;
                            pts_mk.scale.x = 0.0011; pts_mk.scale.y = 0.0011; pts_mk.scale.z = 0.0011;
                            pts_mk.color.a = 1.0; pts_mk.color.r = r; pts_mk.color.g = g; pts_mk.color.b = b;
                            pts_mk.points.reserve(cloud->size());
                            for (const auto& p : cloud->points) {
                                geometry_msgs::msg::Point gp; gp.x = p.x; gp.y = p.y; gp.z = p.z;
                                pts_mk.points.push_back(gp);
                            }
                            markers.markers.push_back(pts_mk);
                        }

                        // B. Esfera Translúcida de Busca
                        visualization_msgs::msg::Marker sphere_mk = base_marker;
                        sphere_mk.ns = prefix + "_sphere"; sphere_mk.id = base_id + 2;
                        sphere_mk.type = visualization_msgs::msg::Marker::SPHERE;
                        sphere_mk.pose.position.x = center.x(); sphere_mk.pose.position.y = center.y(); sphere_mk.pose.position.z = center.z();
                        sphere_mk.scale.x = sphere_radius_ * 2.0; sphere_mk.scale.y = sphere_radius_ * 2.0; sphere_mk.scale.z = sphere_radius_ * 2.0;
                        sphere_mk.color.a = 0.12; sphere_mk.color.r = r; sphere_mk.color.g = g; sphere_mk.color.b = b;
                        markers.markers.push_back(sphere_mk);

                        // C. Ponto de Contato Perfeito (Esfera Sólida Central)
                        visualization_msgs::msg::Marker contact_mk = base_marker;
                        contact_mk.ns = prefix + "_contact"; contact_mk.id = base_id + 3;
                        contact_mk.type = visualization_msgs::msg::Marker::SPHERE;
                        contact_mk.pose = sphere_mk.pose;
                        contact_mk.scale.x = 0.004; contact_mk.scale.y = 0.004; contact_mk.scale.z = 0.004;
                        contact_mk.color.a = 1.0; contact_mk.color.r = norm_r; contact_mk.color.g = norm_g; contact_mk.color.b = norm_b;
                        markers.markers.push_back(contact_mk);

                        if (analysis.valid) {
                            // D. Disco PCA (Cilindro de Vidro)
                            visualization_msgs::msg::Marker disc_mk = base_marker;
                            disc_mk.ns = prefix + "_pca_disc"; disc_mk.id = base_id + 4;
                            disc_mk.type = visualization_msgs::msg::Marker::CYLINDER;
                            disc_mk.pose.position = sphere_mk.pose.position;
                            Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), analysis.normal_vector);
                            disc_mk.pose.orientation.x = q.x(); disc_mk.pose.orientation.y = q.y(); 
                            disc_mk.pose.orientation.z = q.z(); disc_mk.pose.orientation.w = q.w();
                            disc_mk.scale.x = sphere_radius_ * 2.0; disc_mk.scale.y = sphere_radius_ * 2.0; disc_mk.scale.z = 0.0005; // Fino como papel
                            disc_mk.color.a = 0.35; disc_mk.color.r = r; disc_mk.color.g = g; disc_mk.color.b = b;
                            markers.markers.push_back(disc_mk);

                            // E. Seta Normal Local
                            visualization_msgs::msg::Marker norm_mk = base_marker;
                            norm_mk.ns = prefix + "_normal"; norm_mk.id = base_id + 5;
                            norm_mk.type = visualization_msgs::msg::Marker::ARROW;
                            geometry_msgs::msg::Point n0, n1;
                            n0.x = center.x(); n0.y = center.y(); n0.z = center.z();
                            Eigen::Vector3f n_end = center + (analysis.normal_vector * (sphere_radius_ * 1.5f));
                            n1.x = n_end.x(); n1.y = n_end.y(); n1.z = n_end.z();
                            norm_mk.points.push_back(n0); norm_mk.points.push_back(n1);
                            norm_mk.scale.x = 0.002; norm_mk.scale.y = 0.005; norm_mk.scale.z = 0.005;
                            norm_mk.color.a = 1.0; norm_mk.color.r = norm_r; norm_mk.color.g = norm_g; norm_mk.color.b = norm_b; 
                            markers.markers.push_back(norm_mk);
                        }
                    };

                    // 3. Desenhar Zona de Entrada (Verde Claro)
                    drawContact(local.inliers_entry, center_entry, debug_entry, "anim_entry", 100, 
                                0.1f, 0.9f, 0.4f,  // Cor da Nuvem/Disco (Verde)
                                0.0f, 1.0f, 0.0f); // Cor Normal/Contato

                    // 4. Desenhar Zona de Saída (Laranja)
                    if (real_thickness > 0.001f) {
                        drawContact(local.inliers_exit, center_exit, debug_exit, "anim_exit", 200, 
                                    1.0f, 0.5f, 0.1f,  // Cor da Nuvem/Disco (Laranja)
                                    1.0f, 0.4f, 0.0f); // Cor Normal/Contato
                    }

                    // 5. Voxels do Caminho do Raio (Apenas para contexto de colisão)
                   

                    debug_marker_pub_->publish(markers);
                    std::this_thread::sleep_for(std::chrono::milliseconds(animation_delay_ms_));
                }


            
                
                auto t5 = std::chrono::high_resolution_clock::now();
                std::vector<StepAnalysis> steps; steps.reserve(2); 
                
                StepAnalysis res_entry = analyzeLocalSphere(local.inliers_entry, center_entry, current_ray_dir);
                if (res_entry.valid) steps.push_back(res_entry);

                if (real_thickness > 0.001f) { 
                    StepAnalysis res_exit = analyzeLocalSphere(local.inliers_exit, center_exit, current_ray_dir);
                    if (res_exit.valid) steps.push_back(res_exit);
                }
                
                auto t6 = std::chrono::high_resolution_clock::now(); 
                local.acc_analysis_ms += std::chrono::duration<double, std::milli>(t6 - t5).count();

                if (steps.empty()) continue; 

                auto t7 = std::chrono::high_resolution_clock::now();
                StepAnalysis& entry = steps.front();
                StepAnalysis& exit = steps.back();
                
                float score_ang_entry = 1.0f - (std::min(entry.angle_to_normal_deg, 90.0f) / 90.0f);
                float score_ang_exit  = 1.0f - (std::min(exit.angle_to_normal_deg, 90.0f) / 90.0f);
                float score_plan_entry = std::max(0.0f, 1.0f - (entry.curvature * 20.0f)); 
                float score_plan_exit  = std::max(0.0f, 1.0f - (exit.curvature * 20.0f));

                float score_sym_entry = entry.symmetry_score;
                float score_sym_exit  = exit.symmetry_score;

                double total = (score_ang_entry * weight_orientation_ + score_sym_entry * weight_symmetry_ ) * 0.5 
                            + (score_ang_exit * weight_orientation_  + score_sym_exit * weight_symmetry_ ) * 0.5;

                auto t8 = std::chrono::high_resolution_clock::now(); 
                local.acc_scoring_ms += std::chrono::duration<double, std::milli>(t8 - t7).count();

                if (total > best_iter_grasp.total_score) 
                {
                    found_valid_in_optimization = true;
                    float current_offset = finger_offset_; 
                    Eigen::Vector3f p_f1 = current_ray_origin + current_ray_dir * (exact_t_min - current_offset);
                    Eigen::Vector3f p_f2 = current_ray_origin + current_ray_dir * (exact_t_max + current_offset);
                    Eigen::Vector3f center_grasp = (p_f1 + p_f2) / 2.0f;
                    Eigen::Quaternionf best_q = findBestOrientation(p_f1, p_f2);

                    float total_with_bonus = total;
                    
                    if(activate_centroid == true)
                    {
                        Eigen::Vector3f centroid_3d = global_centroid.head<3>();
                    
                    
                        float dist_to_centroid = (center_grasp - centroid_3d).norm();
                        
                    
                        float max_expected_dist = (max_pt_.head<3>() - min_pt_.head<3>()).norm() / 2.0f;
                        
                        
                        if (max_expected_dist < 0.001f) max_expected_dist = 0.001f;

                        
                        float clamped_dist = std::min(dist_to_centroid, max_expected_dist);
                        
                    
                        float proximity_bonus = 1.0f - (clamped_dist / max_expected_dist);
                        
                        
                        float weight_proximity = 0.5f; 
                        total_with_bonus = total + (proximity_bonus * weight_proximity);
                    }
                   
                    
                    

                    best_iter_grasp.pose_center.position.x = center_grasp.x(); 
                    best_iter_grasp.pose_center.position.y = center_grasp.y(); 
                    best_iter_grasp.pose_center.position.z = center_grasp.z();
                    best_iter_grasp.pose_center.orientation.x = best_q.x(); 
                    best_iter_grasp.pose_center.orientation.y = best_q.y(); 
                    best_iter_grasp.pose_center.orientation.z = best_q.z(); 
                    best_iter_grasp.pose_center.orientation.w = best_q.w();
                    
                    best_iter_grasp.total_score_without_bonus = total;
                    best_iter_grasp.total_score = total_with_bonus; 
                    best_iter_grasp.score_orientation_entry = score_ang_entry;
                    best_iter_grasp.score_orientation_exit = score_ang_exit;
                    best_iter_grasp.score_symmetry_entry = score_sym_entry;
                    best_iter_grasp.score_symmetry_exit = score_sym_exit;
                    best_iter_grasp.score_symmetry = (score_sym_entry + score_sym_exit) * 0.5f;
                    best_iter_grasp.score_orientation = (score_ang_entry + score_ang_exit) * 0.5f;
                    best_iter_grasp.raw_ray_dir = current_ray_dir;
                    best_iter_grasp.debug_ray_origin = current_ray_origin;
                    best_iter_grasp.debug_ray_dir_final = current_ray_dir;
                    best_iter_grasp.debug_center_entry = center_entry;
                    best_iter_grasp.debug_center_exit = center_exit;
                    best_iter_grasp.debug_t_min = exact_t_min;
                    best_iter_grasp.debug_t_max = exact_t_max;
                }

                if(activate_centroid == false)
                {
                    if (best_iter_grasp.total_score >= target_score_)
                    { 
                        
                        if (check_collision(best_iter_grasp, collision_kdtree_, false, true)) 
                        { 
                            local.local_candidates.push_back(best_iter_grasp);
                            local.local_hits.push_back(raw_pose);
                            
                            atomic_perfect_count++;
                            perfect_candidate_found = true;
                            break; 
                        }
                    }
                }
                
            } 

            if (!perfect_candidate_found && found_valid_in_optimization) 
            {
                //if (best_iter_grasp.total_score >= target_score_) 
                //{
                    
                //}
                    local.local_candidates.push_back(best_iter_grasp);
                    local.local_hits.push_back(raw_pose);    
                
            }
        }
    }, tbb::auto_partitioner());

    auto t_loop_end = std::chrono::high_resolution_clock::now();

  
    std::vector<ScoredGrasp> initial_candidates;
    initial_candidates.reserve(all_candidates_.size());
    
    double acc_inliers_ms = 0.0;
    double acc_analysis_ms = 0.0;
    double acc_scoring_ms = 0.0;

    for (const auto& local : tls) {
        initial_candidates.insert(initial_candidates.end(), local.local_candidates.begin(), local.local_candidates.end());
        hit_candidates_.insert(hit_candidates_.end(), local.local_hits.begin(), local.local_hits.end());
        
        acc_inliers_ms += local.acc_inliers_ms;
        acc_analysis_ms += local.acc_analysis_ms;
        acc_scoring_ms += local.acc_scoring_ms;
    }
    
    int perfect_grasps_count = atomic_perfect_count.load();

    if (initial_candidates.empty()) 
    {
        has_best_ = false; 
        return geometry_msgs::msg::PoseArray();
    }

    
    uint64_t c_sort_start = __rdtsc();
    auto t_sort_start = std::chrono::high_resolution_clock::now();
    
    std::sort(initial_candidates.begin(), initial_candidates.end(), 
        [](const ScoredGrasp& a, const ScoredGrasp& b) { return a.total_score > b.total_score; });
        
    auto t_sort_end = std::chrono::high_resolution_clock::now();

    auto t_collision_start = std::chrono::high_resolution_clock::now();

    using CandidateType = vision::ScoredGrasp;

    tbb::enumerable_thread_specific<std::vector<CandidateType>> local_best_grasps;

    std::atomic<int> checks_count{0};
    std::atomic<int> total_found{0};
    tbb::task_group_context ctx;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, initial_candidates.size()), 
        [&](const tbb::blocked_range<size_t>& r) {
            
            auto& my_local_grasps = local_best_grasps.local();
            
            for (size_t i = r.begin(); i != r.end(); ++i) {
                
                if (ctx.is_group_execution_cancelled()) break;

                checks_count.fetch_add(1, std::memory_order_relaxed);
                auto& candidate = initial_candidates[i];

                if (check_collision(candidate, collision_kdtree_, false, true)) {
                    
                    
                    my_local_grasps.push_back(candidate);
                    
                    
                    int current_total = total_found.fetch_add(1, std::memory_order_relaxed) + 1;
                    
                   
                    if (current_total >= num_best_grasps_) {
                        ctx.cancel_group_execution();
                        break;
                    }
                }
            }
        }, ctx);

  
    best_grasps_.clear();
    best_grasps_.reserve(num_best_grasps_);

    for (auto& local_vec : local_best_grasps) {
        for (auto& candidate : local_vec) {
            if (best_grasps_.size() < (size_t)num_best_grasps_) 
            {
                best_grasps_.push_back(std::move(candidate)); 
            } 
            else 
            {
                break; 
            }
        }
        if (best_grasps_.size() >= (size_t)num_best_grasps_) break;
    }

    has_best_ = !best_grasps_.empty();

    auto t_collision_end = std::chrono::high_resolution_clock::now();



    /*
    
          auto t_collision_start = std::chrono::high_resolution_clock::now();
    best_grasps_.clear();
    best_grasps_.reserve(num_best_grasps_); 
    
    int checks_count = 0; 

    
    for (auto& candidate : initial_candidates)
    {
        if (best_grasps_.size() >= (size_t)num_best_grasps_) break;
        
        checks_count++;
        if (check_collision(candidate, collision_kdtree_, false, true)) {
            best_grasps_.push_back(candidate);
        }
    }


    has_best_ = !best_grasps_.empty();
    
    auto t_collision_end = std::chrono::high_resolution_clock::now(); 




    */
    
    

   geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.frame_id = "map"; 
    pose_array.header.stamp = this->now(); 
    
    for(const auto& bg : best_grasps_) {
        pose_array.poses.push_back(bg.pose_center);
    }

    auto t_func_end = std::chrono::high_resolution_clock::now();
    uint64_t c_func_end = __rdtsc();

    double d_func = std::chrono::duration<double, std::milli>(t_func_end - t_func_start).count();
    double d_loop = std::chrono::duration<double, std::milli>(t_loop_end - t_loop_start).count();
    double d_sort = std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count();
    double d_col = std::chrono::duration<double, std::milli>(t_collision_end - t_collision_start).count();
    uint64_t clk_func = c_func_end - c_func_start;

    
    double total_cpu_work = acc_inliers_ms + acc_analysis_ms + acc_scoring_ms;
    
 
    double wall_ratio = (total_cpu_work > 1e-6) ? (d_loop / total_cpu_work) : 0.0;

    double wall_inliers = acc_inliers_ms * wall_ratio;
    double wall_analysis = acc_analysis_ms * wall_ratio;
    double wall_scoring = acc_scoring_ms * wall_ratio;
    
    double pct_inliers = (total_cpu_work > 1e-6) ? (acc_inliers_ms / total_cpu_work * 100.0) : 0.0;
    double pct_analysis = (total_cpu_work > 1e-6) ? (acc_analysis_ms / total_cpu_work * 100.0) : 0.0;

   
    double est_speedup = (d_loop > 1e-6) ? (total_cpu_work / d_loop) : 0.0;

    RCLCPP_INFO(this->get_logger(), "================ BENCHMARK TBB PARALLEL (AJUSTADO) ===============");
    RCLCPP_INFO(this->get_logger(), "Total de grasps:         %lu", best_grasps_.size());
    RCLCPP_INFO(this->get_logger(), "Tempo Total Função:      %.4f ms", d_func);
    RCLCPP_INFO(this->get_logger(), "  -> Parallel Loop:      %.4f ms (Speedup Aprox: %.1fx)", d_loop, est_speedup);
    
    RCLCPP_INFO(this->get_logger(), "     |-> Inliers:        ~%.4f ms (%5.1f%% load)", wall_inliers, pct_inliers);
    RCLCPP_INFO(this->get_logger(), "     |-> Analysis:       ~%.4f ms (%5.1f%% load)", wall_analysis, pct_analysis);
    RCLCPP_INFO(this->get_logger(), "     |-> Scoring:        ~%.4f ms", wall_scoring);
    
    RCLCPP_INFO(this->get_logger(), "  -> Sort:               %.4f ms", d_sort);
    RCLCPP_INFO(this->get_logger(), "  -> Collision Final:    %.4f ms (%d checks)", d_col, checks_count.load());
    
    size_t num_to_print = best_grasps_.size();
   if (num_to_print > 0) {
        RCLCPP_INFO(this->get_logger(), ">>> TOP %lu SCORES <<<", num_to_print);
        for (size_t i = 0; i < num_to_print; ++i) {
             RCLCPP_INFO(this->get_logger(), 
                "  #%02lu: Tot: %.4f | SymIn: %.4f | SymOut: %.4f | OriIn: %.4f | OriOut: %.4f", 
                i+1, 
                best_grasps_[i].total_score,
                best_grasps_[i].score_symmetry_entry, 
                best_grasps_[i].score_symmetry_exit,
                best_grasps_[i].score_orientation_entry, 
                best_grasps_[i].score_orientation_exit   
             );
        }
    } else {
        RCLCPP_WARN(this->get_logger(), ">>> NENHUM GRASP ENCONTRADO <<<");
    }
    RCLCPP_INFO(this->get_logger(), "==================================================================");

    for (const auto& bg : best_grasps_) 
    {
        printf("Posição: X=%.8f, Y=%.8f, Z=%.8f\n",
            bg.pose_center.position.x,
            bg.pose_center.position.y,
            bg.pose_center.position.z);
        printf("Orientação: X=%.8f, Y=%.8f, Z=%.8f, W=%.8f\n",
            bg.pose_center.orientation.x,
            bg.pose_center.orientation.y,
            bg.pose_center.orientation.z,
            bg.pose_center.orientation.w);
    }
    fflush(stdout);
    g_last_run_stats.kd_tree = d_kdtree;
    g_last_run_stats.total_func = d_func;
    g_last_run_stats.loop_tbb = d_loop;
    g_last_run_stats.max_inliers = wall_inliers; 
    g_last_run_stats.max_analysis = wall_analysis; 
    g_last_run_stats.max_scoring = wall_scoring;
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
    m.header.stamp = t; m.header.frame_id = "map"; 
    pub_cloud_->publish(m);
    
    visualization_msgs::msg::Marker bbox_marker;
    bbox_marker.header.frame_id = "map"; bbox_marker.header.stamp = t;
    bbox_marker.ns = "bbox"; bbox_marker.id = 0;
    bbox_marker.type = visualization_msgs::msg::Marker::CUBE; bbox_marker.action = 0;
    bbox_marker.pose.position.x = (min_pt_[0] + max_pt_[0]) / 2.0;
    bbox_marker.pose.position.y = (min_pt_[1] + max_pt_[1]) / 2.0;
    bbox_marker.pose.position.z = (min_pt_[2] + max_pt_[2]) / 2.0;
    bbox_marker.pose.orientation.w = 1.0;
    bbox_marker.scale.x = max_pt_[0] - min_pt_[0]; bbox_marker.scale.y = max_pt_[1] - min_pt_[1]; bbox_marker.scale.z = max_pt_[2] - min_pt_[2];
    bbox_marker.color.r = 0.8; bbox_marker.color.g = 0.8; bbox_marker.color.b = 0.8; bbox_marker.color.a = 0.2; 
    pub_bbox_->publish(bbox_marker);

    if(has_best_) 
    {
        publishBest();
        publishBestGraspDebug(best_grasps_[0]);
    } 

    
    
    
}


void GenerateGraspPoses::publishBest() 
{
    
    visualization_msgs::msg::MarkerArray ma; 
    geometry_msgs::msg::PoseArray pose_array_msg;
    auto t = this->now();
    
    pose_array_msg.header.frame_id = "map";
    pose_array_msg.header.stamp = t;
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // ESTA MATRIZ É USADA EXCLUSIVAMENTE PARA O ARQUIVO .GLB DO RVIZ!
    Eigen::Affine3f visual_tf = Eigen::Affine3f::Identity();
    Eigen::Matrix3f rot_geom;
    rot_geom = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
             * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ());
    visual_tf.linear() = rot_geom;
    visual_tf.translation() = Eigen::Vector3f(mesh_offset_x_, mesh_offset_y_, mesh_offset_z_);

    for(size_t i = 0; i < best_grasps_.size(); i++)
    {
        const auto& grasp = best_grasps_[i];
        pose_array_msg.poses.push_back(grasp.pose_center);
        
        Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
        Eigen::Quaternionf grasp_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);
        
        // POSE DO TCP PURO!
        Eigen::Affine3f tf_tcp_to_world = Eigen::Translation3f(grasp_pos) * grasp_rot;
        Eigen::Quaternionf q_tcp(tf_tcp_to_world.rotation());

        // 1. DESENHA AS CAIXAS (As caixas já têm o offset embutido. Usa apenas a pose do TCP)
        for(size_t b = 0; b < gripper_boxes_.size(); b++)
        {
            const auto& box = gripper_boxes_[b];
            
            visualization_msgs::msg::Marker mk;
            mk.header.frame_id = "map";
            mk.header.stamp = t;
            mk.ns = "debug_boxes_grasp_" + std::to_string(i); 
            mk.id = b;
            mk.type = visualization_msgs::msg::Marker::CUBE;
            mk.action = visualization_msgs::msg::Marker::ADD;

            // MULTIPLICA DIRETAMENTE PELO TCP! SEM DUPLA TRANSFORMAÇÃO!
            Eigen::Vector3f center_world = tf_tcp_to_world * box.center;

          
            
            mk.pose.position.x = center_world.x();
            mk.pose.position.y = center_world.y();
            mk.pose.position.z = center_world.z();

            mk.pose.orientation.x = q_tcp.x();
            mk.pose.orientation.y = q_tcp.y();
            mk.pose.orientation.z = q_tcp.z();
            mk.pose.orientation.w = q_tcp.w();

            mk.scale.x = box.dimensions.x();
            mk.scale.y = box.dimensions.y();
            mk.scale.z = box.dimensions.z();

            if (i == 0) { mk.color.r = 0.0; mk.color.g = 1.0; mk.color.b = 0.0; mk.color.a = 0.6; }
            else        { mk.color.r = 0.0; mk.color.g = 1.0; mk.color.b = 1.0; mk.color.a = 0.3; }
            
            ma.markers.push_back(mk);
        }
        
        
        // 2. DESENHA A MALHA (.GLB)
        if (publish_gripper_mesh_)
        {
            // ==============================================================
            // AJUSTE FINO EXCLUSIVO PARA O OBJ (.GLB)
            // Esses valores transladam a malha visual em relação ao TCP,
            // corrigindo a origem errada do arquivo 3D sem afetar a física.
            // ==============================================================
            float mesh_ajuste_x = 0.0f; // Mude aqui para corrigir o desvio em X
            float mesh_ajuste_y = 0.0f; // Mude aqui para corrigir o desvio em Y
            float mesh_ajuste_z = 0.0f; // Mude aqui para corrigir o desvio em Z

            Eigen::Affine3f visual_tf = Eigen::Affine3f::Identity();
            Eigen::Matrix3f rot_geom;
            
            // Mantém a rotação de ouro (incluindo o giro de 180 graus em X)
            rot_geom = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
                     * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
                     * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ());
            
            visual_tf.linear() = rot_geom;
            
            // Aplica os parâmetros padrão + os ajustes manuais
            visual_tf.translation() = Eigen::Vector3f(
                mesh_offset_x_, 
                mesh_offset_y_ ,
                -0.015 
            );

            // O arquivo que o RViz lê do HD é cru, então ele precisa do offset + a pose do TCP
            Eigen::Affine3f tf_mesh_final = tf_tcp_to_world * visual_tf;
            
            Eigen::Vector3f t_mesh = tf_mesh_final.translation();
            Eigen::Quaternionf q_mesh(tf_mesh_final.rotation());
            
            visualization_msgs::msg::Marker mesh_mk;
            mesh_mk.header.frame_id = "map";
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

            if (i == 0) { mesh_mk.color.r = 0.0; mesh_mk.color.g = 1.0; mesh_mk.color.b = 0.0; mesh_mk.color.a = 0.7; }
            else        { mesh_mk.color.r = 0.0; mesh_mk.color.g = 1.0; mesh_mk.color.b = 1.0; mesh_mk.color.a = 0.4; }

            mesh_mk.mesh_use_embedded_materials = true;
            ma.markers.push_back(mesh_mk);
        }
    }
    
    if (!accumulated_cloud->empty()) 
    {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*accumulated_cloud, cloud_msg);
        cloud_msg.header.frame_id = "map";
        cloud_msg.header.stamp = t;
        pub_debug_grasps_cloud_->publish(cloud_msg);
    }

    pub_markers_->publish(ma);
    pub_poses_->publish(pose_array_msg);
    
    
    
}



void GenerateGraspPoses::publishBestGraspDebug(const ScoredGrasp& best)
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(stored_cloud_);

    auto collectSphere = [&](const Eigen::Vector3f& center) -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointXYZ query;
        query.x = center.x(); query.y = center.y(); query.z = center.z();
        std::vector<int> indices;
        std::vector<float> dists;
        kdtree.radiusSearch(query, sphere_radius_, indices, dists);
        cloud->points.reserve(indices.size());
        for (int idx : indices)
            cloud->points.push_back(stored_cloud_->points[idx]);
        return cloud;
    };

    const Eigen::Vector3f& ray_dir = best.debug_ray_dir_final;
    const Eigen::Vector3f& ray_origin = best.debug_ray_origin;
    const Eigen::Vector3f& center_entry = best.debug_center_entry;
    const Eigen::Vector3f& center_exit = best.debug_center_exit;
    const float t_min = best.debug_t_min;
    const float t_max = best.debug_t_max;
    const float thickness = t_max - t_min;

    auto cloud_entry = collectSphere(center_entry);
    auto cloud_exit  = collectSphere(center_exit);

    StepAnalysis res_entry = analyzeLocalSphere(cloud_entry, center_entry, ray_dir);
    StepAnalysis res_exit;
    if (thickness > 0.001f)
        res_exit = analyzeLocalSphere(cloud_exit, center_exit, ray_dir);

    auto getPCAAxes = [&](const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                          const Eigen::Vector3f& center, const Eigen::Vector3f& rdir,
                          Eigen::Vector3f& normal_out, Eigen::Vector3f& t1_out, Eigen::Vector3f& t2_out,
                          std::vector<float>& us_out, std::vector<float>& vs_out,
                          std::vector<int>& sectors_out, bool* sector_valid_out, int* sector_counts_out)
    {
        const size_t N = cloud->size();
        const pcl::PointXYZ* pts = cloud->points.data();
        float sx=0,sy=0,sz=0;
        for (size_t i=0;i<N;++i){sx+=pts[i].x;sy+=pts[i].y;sz+=pts[i].z;}
        float inv_N=1.0f/static_cast<float>(N);
        float mx=sx*inv_N, my=sy*inv_N, mz=sz*inv_N;
        float axx=0,axy=0,axz=0,ayy=0,ayz=0,azz=0;
        for(size_t i=0;i<N;++i){
            float dx=pts[i].x-mx,dy=pts[i].y-my,dz=pts[i].z-mz;
            axx+=dx*dx;axy+=dx*dy;axz+=dx*dz;ayy+=dy*dy;ayz+=dy*dz;azz+=dz*dz;
        }
        Eigen::Matrix3f cov;
        cov(0,0)=axx*inv_N;cov(0,1)=axy*inv_N;cov(0,2)=axz*inv_N;
        cov(1,0)=cov(0,1);cov(1,1)=ayy*inv_N;cov(1,2)=ayz*inv_N;
        cov(2,0)=cov(0,2);cov(2,1)=cov(1,2);cov(2,2)=azz*inv_N;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> sol(cov, Eigen::ComputeEigenvectors);
        normal_out = sol.eigenvectors().col(0);
        if (normal_out.dot(rdir) > 0.0f) normal_out = -normal_out;
        t1_out = sol.eigenvectors().col(1);
        t2_out = sol.eigenvectors().col(2);

        const float ox=center.x(), oy=center.y(), oz=center.z();
        const int threshold = static_cast<int>(min_points_per_segment_);
        const float safe_r_loc = (sphere_radius_ > 1e-4f) ? sphere_radius_ : 0.02f;
        const float r2_half = (safe_r_loc * 0.5f) * (safe_r_loc * 0.5f);
        for(int k=0;k<16;++k) sector_counts_out[k]=0;
        us_out.resize(N); vs_out.resize(N); sectors_out.resize(N);
        for(size_t i=0;i<N;++i){
            float dx=pts[i].x-ox, dy=pts[i].y-oy, dz=pts[i].z-oz;
            float u=dx*t1_out.x()+dy*t1_out.y()+dz*t1_out.z();
            float v=dx*t2_out.x()+dy*t2_out.y()+dz*t2_out.z();
            float r2 = u*u + v*v;
            int ring = (r2 >= r2_half) ? 1 : 0;
            int sec=(u>=0.0f)|((v>=0.0f)<<1)|((u*u>=v*v)<<2)|(ring<<3);
            us_out[i]=u; vs_out[i]=v;
            sectors_out[i]=sec;
            sector_counts_out[sec]++;
        }
        for(int k=0;k<16;++k)
            sector_valid_out[k]=(sector_counts_out[k]>=threshold);
    };

    // ========== Configuração visual ==========
    constexpr double CUBE_SIZE = 0.0011;  // Tamanho dos cubos de simetria

    // Cores/estilos por esfera
    struct SphereStyle {
        float normal_r, normal_g, normal_b;   // seta normal
        float sphere_r, sphere_g, sphere_b;   // esfera transparente + cubos + círculo
        float contact_r, contact_g, contact_b; // ponto de contato
    };

    const SphereStyle style_entry = {0.0f, 1.0f, 0.4f,   // normal: verde claro
                                     0.1f, 0.9f, 0.4f,   // esfera/cubos/círculo: verde
                                     0.0f, 1.0f, 0.0f};  // contato: verde puro

    const SphereStyle style_exit  = {1.0f, 0.4f, 0.0f,   // normal: laranja
                                     1.0f, 0.5f, 0.1f,   // esfera/cubos/círculo: laranja
                                     1.0f, 0.5f, 0.0f};  // contato: laranja

    visualization_msgs::msg::MarkerArray ma;
    auto stamp = this->now();
    const std::string frame = "map";
    int id = 0;
    const float safe_r = (sphere_radius_ > 1e-4f) ? sphere_radius_ : 0.02f;

    auto addSphereDebug = [&](const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                              const Eigen::Vector3f& center, const Eigen::Vector3f& rdir,
                              const std::string& prefix, const SphereStyle& style)
    {
        if (!cloud || cloud->empty()) return;
        const size_t N = cloud->size();

        Eigen::Vector3f normal, t1, t2;
        std::vector<float> us, vs;
        std::vector<int> sectors;
        bool sector_valid[16];
        int sector_counts[16];
        getPCAAxes(cloud, center, rdir, normal, t1, t2, us, vs, sectors, sector_valid, sector_counts);

        float ox = center.x(), oy = center.y(), oz = center.z();

        // Helper: ponto no plano PCA a partir de coords (u,v), com leve offset ao longo da normal
        auto planePoint = [&](float u, float v, float normal_offset) {
            geometry_msgs::msg::Point p;
            p.x = ox + u*t1.x() + v*t2.x() + normal.x()*normal_offset;
            p.y = oy + u*t1.y() + v*t2.y() + normal.y()*normal_offset;
            p.z = oz + u*t1.z() + v*t2.z() + normal.z()*normal_offset;
            return p;
        };

        // --- 1. Cubos de simetria: COR DISTINTA POR SETOR (16 setores) ---
        static constexpr float SEC16[16][3] = {
            // Anel INTERNO (centro → meio raio) — cores QUENTES
            {1.00f, 0.00f, 0.00f}, {1.00f, 0.50f, 0.00f}, {1.00f, 1.00f, 0.00f}, {0.60f, 1.00f, 0.00f},
            {1.00f, 0.00f, 0.50f}, {1.00f, 0.30f, 0.30f}, {0.80f, 0.60f, 0.00f}, {1.00f, 0.00f, 1.00f},
            // Anel EXTERNO (meio → raio completo) — cores FRIAS
            {0.00f, 0.40f, 1.00f}, {0.00f, 0.80f, 1.00f}, {0.00f, 0.90f, 0.60f}, {0.50f, 0.00f, 1.00f},
            {0.30f, 0.30f, 1.00f}, {0.00f, 1.00f, 0.80f}, {0.60f, 0.40f, 1.00f}, {0.00f, 0.60f, 0.80f},
        };
        for (int k = 0; k < 16; ++k) {
            if (sector_counts[k] == 0) continue;
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_sec"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::CUBE_LIST;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.scale.x = CUBE_SIZE; m.scale.y = CUBE_SIZE; m.scale.z = CUBE_SIZE;
            m.pose.orientation.w = 1.0;
            m.color.r = SEC16[k][0]; m.color.g = SEC16[k][1]; m.color.b = SEC16[k][2];
            m.color.a = sector_valid[k] ? 1.0f : 0.35f;
            m.points.reserve(sector_counts[k]);
            for (size_t i = 0; i < N; ++i) {
                if (sectors[i] != k) continue;
                m.points.push_back(planePoint(us[i], vs[i], 0.0f));
            }
            if (!m.points.empty()) ma.markers.push_back(std::move(m));
        }

        // --- 2. Disco do plano PCA em DUAS ZONAS RADIAIS ---
        Eigen::Quaternionf q_disc = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), normal);

        // Zona EXTERNA (meio raio → raio completo): cor da esfera
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_plane_outer"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::CYLINDER;
            m.action = visualization_msgs::msg::Marker::ADD;
            Eigen::Vector3f off_out = normal * (-0.0015f);
            m.pose.position.x = ox + off_out.x();
            m.pose.position.y = oy + off_out.y();
            m.pose.position.z = oz + off_out.z();
            m.pose.orientation.x = q_disc.x(); m.pose.orientation.y = q_disc.y();
            m.pose.orientation.z = q_disc.z(); m.pose.orientation.w = q_disc.w();
            m.scale.x = safe_r * 2.0; m.scale.y = safe_r * 2.0; m.scale.z = 0.0005;
            m.color.r = style.sphere_r; m.color.g = style.sphere_g; m.color.b = style.sphere_b;
            m.color.a = 0.35f;
            ma.markers.push_back(std::move(m));
        }

        // Zona INTERNA (centro → meio raio): cor clareada
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_plane_inner"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::CYLINDER;
            m.action = visualization_msgs::msg::Marker::ADD;
            Eigen::Vector3f off_in = normal * (-0.0007f);
            m.pose.position.x = ox + off_in.x();
            m.pose.position.y = oy + off_in.y();
            m.pose.position.z = oz + off_in.z();
            m.pose.orientation.x = q_disc.x(); m.pose.orientation.y = q_disc.y();
            m.pose.orientation.z = q_disc.z(); m.pose.orientation.w = q_disc.w();
            m.scale.x = safe_r * 1.0; m.scale.y = safe_r * 1.0; m.scale.z = 0.0005;
            m.color.r = std::min(1.0f, style.sphere_r + 0.5f);
            m.color.g = std::min(1.0f, style.sphere_g + 0.5f);
            m.color.b = std::min(1.0f, style.sphere_b + 0.5f);
            m.color.a = 0.55f;
            ma.markers.push_back(std::move(m));
        }

        // --- 2.5. LINHAS DIVISÓRIAS dos setores sobre o disco ---
        // Offset acima dos discos (que estão em -0.0007/-0.0015) para ficarem visíveis,
        // mas abaixo dos cubos (que estão em 0.0). Usamos -0.0003.
        const float LINE_Z = -0.0003f;

        // 2.5a. Linhas radiais — 4 diâmetros = 8 fatias angulares de 45°
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_div_radial"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::LINE_LIST;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.scale.x = 0.0006; // espessura da linha
            m.pose.orientation.w = 1.0;
            m.color.r = 0.05f; m.color.g = 0.05f; m.color.b = 0.05f; m.color.a = 0.9f; // quase preto

            // 8 raios a cada 45° (0, 45, 90, ... 315) => 4 diâmetros completos
            for (int s = 0; s < 8; ++s) {
                float ang = s * (M_PI / 4.0f); // 45° em rad
                float cu = std::cos(ang) * safe_r;
                float cv = std::sin(ang) * safe_r;
                m.points.push_back(planePoint(0.0f, 0.0f, LINE_Z)); // centro
                m.points.push_back(planePoint(cu, cv, LINE_Z));     // borda
            }
            ma.markers.push_back(std::move(m));
        }

        // 2.5b. Círculo de fronteira radial (meio raio) — separa anel interno do externo
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_div_ring_mid"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::LINE_STRIP;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.scale.x = 0.0007;
            m.pose.orientation.w = 1.0;
            m.color.r = 0.05f; m.color.g = 0.05f; m.color.b = 0.05f; m.color.a = 0.9f;
            constexpr int RES = 96;
            float mid_r = safe_r * 0.5f;
            m.points.reserve(RES + 1);
            for (int j = 0; j <= RES; ++j) {
                float ang = 2.0f * M_PI * j / RES;
                m.points.push_back(planePoint(std::cos(ang) * mid_r, std::sin(ang) * mid_r, LINE_Z));
            }
            ma.markers.push_back(std::move(m));
        }

        // 2.5c. Círculo de borda externa (raio completo) — contorno nítido do disco
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_div_ring_outer"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::LINE_STRIP;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.scale.x = 0.0009; // borda externa um pouco mais grossa
            m.pose.orientation.w = 1.0;
            m.color.r = 0.05f; m.color.g = 0.05f; m.color.b = 0.05f; m.color.a = 1.0f;
            constexpr int RES = 128;
            m.points.reserve(RES + 1);
            for (int j = 0; j <= RES; ++j) {
                float ang = 2.0f * M_PI * j / RES;
                m.points.push_back(planePoint(std::cos(ang) * safe_r, std::sin(ang) * safe_r, LINE_Z));
            }
            ma.markers.push_back(std::move(m));
        }

        // --- 4. Esfera Principal Suave ---
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_sphere"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = ox; m.pose.position.y = oy; m.pose.position.z = oz;
            m.pose.orientation.w = 1.0;
            m.scale.x = safe_r * 2.0; m.scale.y = safe_r * 2.0; m.scale.z = safe_r * 2.0;
            m.color.r = style.sphere_r; m.color.g = style.sphere_g; m.color.b = style.sphere_b;
            m.color.a = 0.12f;
            ma.markers.push_back(std::move(m));
        }

        // --- 5. Normal PCA ---
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_normal"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::ARROW;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.scale.x = 0.002; m.scale.y = 0.005; m.scale.z = 0.005;
            m.pose.orientation.w = 1.0;
            m.color.r = style.normal_r; m.color.g = style.normal_g; m.color.b = style.normal_b;
            m.color.a = 1.0f;
            geometry_msgs::msg::Point p0, p1;
            float arrow_len = safe_r * 1.5f;
            p0.x = ox; p0.y = oy; p0.z = oz;
            p1.x = ox + normal.x()*arrow_len;
            p1.y = oy + normal.y()*arrow_len;
            p1.z = oz + normal.z()*arrow_len;
            m.points.push_back(p0); m.points.push_back(p1);
            ma.markers.push_back(std::move(m));
        }

        // --- 6. Centro de Contato ---
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_contact"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = ox; m.pose.position.y = oy; m.pose.position.z = oz;
            m.pose.orientation.w = 1.0;
            m.scale.x = 0.004; m.scale.y = 0.004; m.scale.z = 0.004;
            m.color.r = style.contact_r; m.color.g = style.contact_g; m.color.b = style.contact_b;
            m.color.a = 1.0f;
            ma.markers.push_back(std::move(m));
        }

        // --- 2.5. Pontos nas posições 3D ORIGINAIS (vermelho) ---
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = prefix + "_pts_original_3d"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::CUBE_LIST;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.scale.x = CUBE_SIZE; m.scale.y = CUBE_SIZE; m.scale.z = CUBE_SIZE;
            m.pose.orientation.w = 1.0;
            m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 0.0f; m.color.a = 1.0f;
            const pcl::PointXYZ* pts = cloud->points.data();
            m.points.reserve(N);
            for (size_t i = 0; i < N; ++i) {
                geometry_msgs::msg::Point p;
                p.x = pts[i].x; p.y = pts[i].y; p.z = pts[i].z;
                m.points.push_back(p);
            }
            ma.markers.push_back(std::move(m));
        }


         {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = frame; m.header.stamp = stamp;
            m.ns = "voxel_cloud"; m.id = id++;
            m.type = visualization_msgs::msg::Marker::CUBE_LIST;
            m.action = visualization_msgs::msg::Marker::ADD;
            // Tamanho do cubo = tamanho do voxel usado na nuvem (cloud_voxel_size_)
            const double vsize = (cloud_voxel_size_ >= 0.001f)
                                ? static_cast<double>(cloud_voxel_size_) : 0.001;
            m.scale.x = vsize; m.scale.y = vsize; m.scale.z = vsize;
            m.pose.orientation.w = 1.0;
            m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 1.0f; m.color.a = 1.0f; // cinza
            m.points.reserve(stored_cloud_->points.size());
            for (const auto& pt : stored_cloud_->points) {
                geometry_msgs::msg::Point p;
                p.x = pt.x; p.y = pt.y; p.z = pt.z;
                m.points.push_back(p);
            }
            ma.markers.push_back(std::move(m));
        }
    };

    addSphereDebug(cloud_entry, center_entry, ray_dir, "best_entry", style_entry);
    if (thickness > 0.001f)
        addSphereDebug(cloud_exit, center_exit, ray_dir, "best_exit", style_exit);

    // ===== Raio completo (amarelo) =====
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame; m.header.stamp = stamp;
        m.ns = "best_ray"; m.id = id++;
        m.type = visualization_msgs::msg::Marker::ARROW;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.002; m.scale.y = 0.005; m.scale.z = 0.005;
        m.pose.orientation.w = 1.0;
        m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 0.2f; m.color.a = 1.0f;
        geometry_msgs::msg::Point p0, p1;
        Eigen::Vector3f visual_end = ray_origin + ray_dir * (t_max + 0.03f);
        p0.x = ray_origin.x(); p0.y = ray_origin.y(); p0.z = ray_origin.z();
        p1.x = visual_end.x(); p1.y = visual_end.y(); p1.z = visual_end.z();
        m.points.push_back(p0); m.points.push_back(p1);
        ma.markers.push_back(std::move(m));
    }

    // ===== Origem do raio (magenta) =====
    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = frame; m.header.stamp = stamp;
        m.ns = "best_ray_origin"; m.id = id++;
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = ray_origin.x(); m.pose.position.y = ray_origin.y(); m.pose.position.z = ray_origin.z();
        m.pose.orientation.w = 1.0;
        m.scale.x = 0.005; m.scale.y = 0.005; m.scale.z = 0.005;
        m.color.r = 1.0f; m.color.g = 0.0f; m.color.b = 1.0f; m.color.a = 1.0f;
        ma.markers.push_back(std::move(m));
    }

    sphere_debug_pub_->publish(ma);

    {
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud_entry, msg);
        msg.header.frame_id = frame; msg.header.stamp = stamp;
        pub_debug_inliers_->publish(msg);
    }

    
}
} // namespace vision

RCLCPP_COMPONENTS_REGISTER_NODE(vision::GenerateGraspPoses)