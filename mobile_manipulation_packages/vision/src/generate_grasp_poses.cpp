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
    this->declare_parameter<bool>("publish_object_mesh", true);

    this->declare_parameter<std::string>("gripper_mesh_path", "/home/momesso/hand_and_fingers.obj");
    this->declare_parameter<double>("gripper_mesh_scale", 1.0);
    
    this->declare_parameter<std::string>("gripper_glb_path", "/home/momesso/pcds/GLB_Foxglove/PandaHand.glb");
    this->declare_parameter<bool>("publish_gripper_mesh", true);
    
    this->declare_parameter<double>("mesh_offset_x", 0.025);
    this->declare_parameter<double>("mesh_offset_y", 0.0);
    this->declare_parameter<double>("mesh_offset_z", 0.0);
    
    this->declare_parameter<double>("mesh_rot_roll", 1.57);
    this->declare_parameter<double>("mesh_rot_pitch", 0.0);
    this->declare_parameter<double>("mesh_rot_yaw", 1.57); 

    this->declare_parameter<double>("grid_res", 0.005);
    this->declare_parameter<double>("cloud_voxel_size", 0.001);
    
    this->declare_parameter<double>("cylinder_radius", 0.025); 
    this->declare_parameter<double>("cylinder_height", 0.005);
    this->declare_parameter<double>("analysis_step_size", 0.01);
    
    this->declare_parameter<double>("max_gripper_width", 0.07); 
    this->declare_parameter<double>("finger_offset", 0.027); 
    
    this->declare_parameter<int>("min_points_per_segment", 2);
    this->declare_parameter<double>("weight_orientation", 0.75); 
    this->declare_parameter<double>("weight_symmetry", 0.25);
    this->declare_parameter<double>("target_score", 10.0);
    
    this->declare_parameter<bool>("use_mean_filter", true); 
    this->declare_parameter<int>("mean_filter_k", 15);

    this->declare_parameter<int>("num_best_grasps", 100);
    this->declare_parameter<double>("rotation_step_deg", 55.0);

    this->declare_parameter<int>("num_random_orientations", 1);

    this->declare_parameter<int>("num_benchmark_runs", 1);
    this->declare_parameter<bool>("enable_ray_animation", false);
    this->declare_parameter<int>("animation_delay_ms", 5000);

    this->declare_parameter<bool>("activate_centroid", true);
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
        collision_kdtree_.setInputCloud(collision_cloud_);
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
    // (Talvez um pouquinho maior para garantir margem, ex: 1.1x, mas você pediu max_gripper_width_)
    float ray_len = max_gripper_width_;

    struct VoxelLimits {
        float min_val = 1e9f;
        float max_val = -1e9f;
        bool active = false;
    };

    // Grids de Projeção (Mapas Esparsos)
    std::map<int, std::map<int, VoxelLimits>> grid_YZ; // Para raios X
    std::map<int, std::map<int, VoxelLimits>> grid_XZ; // Para raios Y
    std::map<int, std::map<int, VoxelLimits>> grid_XY; // Para raios Z

    float inv_res = 1.0f / res;

   std::set<std::tuple<int, int, int>> occupied_voxels;

    for (const auto& pt : stored_cloud_->points)
    {
        int ix = std::floor(pt.x * inv_res);
        int iy = std::floor(pt.y * inv_res);
        int iz = std::floor(pt.z * inv_res);
        occupied_voxels.insert({ix, iy, iz});
    }

    // Lambda original mantido
    auto add_pose = [&](float x, float y, float z, Eigen::Vector3f dir) 
    {
        geometry_msgs::msg::Pose p;
        p.position.x = x; p.position.y = y; p.position.z = z;
        
        Eigen::Quaternionf q; 
        q.setFromTwoVectors(Eigen::Vector3f::UnitZ(), dir); 
        p.orientation.x = q.x(); p.orientation.y = q.y(); 
        p.orientation.z = q.z(); p.orientation.w = q.w();
        
        poses.push_back(p);
        ray_lengths.push_back(ray_len);
    };

    // 2. GERAÇÃO DE RAIOS (Apenas nas faces expostas / casca externa)
    for (const auto& voxel : occupied_voxels) 
    {
        int ix = std::get<0>(voxel);
        int iy = std::get<1>(voxel);
        int iz = std::get<2>(voxel);

        // Centro do voxel atual
        float cx = (ix + 0.5f) * res;
        float cy = (iy + 0.5f) * res;
        float cz = (iz + 0.5f) * res;

        // --- EIXO X ---
        // Se o voxel vizinho em +X está VAZIO, esta face é casca externa!
        if (occupied_voxels.find({ix + 1, iy, iz}) == occupied_voxels.end()) {
            add_pose(cx + res, cy, cz, {-1, 0, 0}); // Raio vem da direita pra esquerda
        }
        // Se o voxel vizinho em -X está VAZIO
        if (occupied_voxels.find({ix - 1, iy, iz}) == occupied_voxels.end()) {
            add_pose(cx - res, cy, cz, {1, 0, 0});  // Raio vem da esquerda pra direita
        }

        // --- EIXO Y ---
        if (occupied_voxels.find({ix, iy + 1, iz}) == occupied_voxels.end()) {
            add_pose(cx, cy + res, cz, {0, -1, 0}); // Vem de cima pra baixo
        }
        if (occupied_voxels.find({ix, iy - 1, iz}) == occupied_voxels.end()) {
            add_pose(cx, cy - res, cz, {0, 1, 0});  // Vem de baixo pra cima
        }

        // --- EIXO Z ---
        if (occupied_voxels.find({ix, iy, iz + 1}) == occupied_voxels.end()) {
            add_pose(cx, cy, cz + res, {0, 0, -1}); // Vem da frente pra trás
        }
        if (occupied_voxels.find({ix, iy, iz - 1}) == occupied_voxels.end()) {
            add_pose(cx, cy, cz - res, {0, 0, 1});  // Vem de trás pra frente
        }
    }

    // 3. VISUALIZAÇÃO
    if (pub_rays_) { 
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        for (size_t i = 0; i < poses.size(); ++i) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map"; 
            marker.header.stamp = this->now();
            marker.ns = "adaptive_rays";
            marker.id = i;
            marker.type = visualization_msgs::msg::Marker::ARROW;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose = poses[i];
            
            marker.scale.x = ray_lengths[i]; // Comprimento exato da garra
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

StepAnalysis GenerateGraspPoses::analyzeLocalCylinder(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const Eigen::Vector3f& center,
    const Eigen::Vector3f& ray_dir)
{
    StepAnalysis result; 
    result.valid = false; 
    result.center = center;

    size_t N = cloud->size();
    if (!cloud || N <= min_points_per_segment_) return result;
    result.point_count = N;

   
    const pcl::PointXYZ* points_ptr = cloud->points.data();
    
    // Média
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        sx += points_ptr[i].x; sy += points_ptr[i].y; sz += points_ptr[i].z;
    }
    float inv_N = 1.0f / static_cast<float>(N);
    float mean_x = sx * inv_N; float mean_y = sy * inv_N; float mean_z = sz * inv_N;
    Eigen::Vector3f centroid(mean_x, mean_y, mean_z);

    // Covariância
    float acc_xx = 0.0f, acc_xy = 0.0f, acc_xz = 0.0f;
    float acc_yy = 0.0f, acc_yz = 0.0f, acc_zz = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        float dx = points_ptr[i].x - mean_x; float dy = points_ptr[i].y - mean_y; float dz = points_ptr[i].z - mean_z;
        acc_xx += dx * dx; acc_xy += dx * dy; acc_xz += dx * dz;
        acc_yy += dy * dy; acc_yz += dy * dz; acc_zz += dz * dz;
    }

    Eigen::Matrix3f covariance_matrix;
    covariance_matrix(0, 0) = acc_xx * inv_N; covariance_matrix(0, 1) = acc_xy * inv_N; covariance_matrix(0, 2) = acc_xz * inv_N;
    covariance_matrix(1, 0) = covariance_matrix(0, 1); covariance_matrix(1, 1) = acc_yy * inv_N; covariance_matrix(1, 2) = acc_yz * inv_N;
    covariance_matrix(2, 0) = covariance_matrix(0, 2); covariance_matrix(2, 1) = covariance_matrix(1, 2); covariance_matrix(2, 2) = acc_zz * inv_N;

  
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
    
   
    Eigen::Vector3f axis_candidate_A = eigen_solver.eigenvectors().col(1); 
    Eigen::Vector3f axis_candidate_B = eigen_solver.eigenvectors().col(2); 

  
    float dot_A = std::abs(ray_dir.dot(axis_candidate_A));
    float dot_B = std::abs(ray_dir.dot(axis_candidate_B));

    Eigen::Vector3f cylinder_axis;
    if (dot_A < dot_B) {
        cylinder_axis = axis_candidate_A; 
    } else {
        cylinder_axis = axis_candidate_B; 
    }

   
    float ray_on_axis = ray_dir.dot(cylinder_axis);
    Eigen::Vector3f normal_geom = ray_dir - (ray_on_axis * cylinder_axis);
    
    if (normal_geom.squaredNorm() > 1e-6f) {
        normal_geom.normalize();
        normal_geom = -normal_geom; 
    } else {
        normal_geom = -ray_dir; 
    }

    result.normal_vector = normal_geom;

   
    Eigen::Vector3f values = eigen_solver.eigenvalues();
    float sum = values.sum();
    result.curvature = (sum > 1e-6f) ? std::abs(values[0]) / sum : 0.0f; 

  
    Eigen::Vector3f diff = centroid - center;
    float parallel_comp = diff.dot(ray_dir);
    Eigen::Vector3f radial_offset_vec = diff - (parallel_comp * ray_dir);
    float radial_offset = radial_offset_vec.norm();

    float safe_radius = (cylinder_radius_ > 1e-4f) ? cylinder_radius_ : 0.02f;
    float linear_score = 1.0f - (radial_offset / safe_radius);
    result.symmetry_score = std::max(0.0f, linear_score);

   
    float dot = std::abs(ray_dir.dot(result.normal_vector));
    if (dot > 1.0f) dot = 1.0f; 
    result.angle_to_normal_deg = std::acos(dot) * 57.2957795f; 
    
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


bool GenerateGraspPoses::check_collision(ScoredGrasp& grasp, const pcl::KdTreeFLANN<pcl::PointXYZ>& env_kdtree, bool publish_debug, bool try_rotations)
{
    if (gripper_boxes_.empty()) return true;
    if (!env_kdtree.getInputCloud() || env_kdtree.getInputCloud()->empty()) return true;

    static bool radius_initialized = false;
    static float max_search_radius = 0.0f;

    if (!radius_initialized) 
    {
        float max_box_dist = 0.0f;
        for(const auto& box : gripper_boxes_) {
             float mx = std::max(std::abs(box.max_pt.x()), std::abs(box.min_pt.x()));
             float my = std::max(std::abs(box.max_pt.y()), std::abs(box.min_pt.y()));
             float mz = std::max(std::abs(box.max_pt.z()), std::abs(box.min_pt.z()));
             float dist = std::sqrt(mx*mx + my*my + mz*mz);
             if (dist > max_box_dist) max_box_dist = dist;
        }
        max_search_radius = max_box_dist + 0.04f; 
        radius_initialized = true;
    }

    Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
    Eigen::Quaternionf original_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);

    std::vector<int> pointIdx;
    std::vector<float> pointSqDist;
    pcl::PointXYZ searchPoint(grasp_pos.x(), grasp_pos.y(), grasp_pos.z());

    if (env_kdtree.radiusSearch(searchPoint, max_search_radius, pointIdx, pointSqDist) == 0) {
        if (publish_debug) 
        {
            RCLCPP_WARN(this->get_logger(), "ALERTA CRÍTICO: A KD-Tree encontrou 0 pontos perto da garra! As nuvens estão em coordenadas separadas!");
            return false;
        }
        return true; 
    }

    const float MARGIN = 0.003f; // Margem de segurança de 2mm
    const auto& cloud_points = env_kdtree.getInputCloud()->points;
    const int NUM_STEPS = try_rotations ? 18 : 1; 
    const float ANGLE_STEP = (2.0f * M_PI) / 18.0f; 
    
    bool final_collision_state = true; 

    for (int step = 0; step < NUM_STEPS; ++step)
    {
        float current_angle = step * ANGLE_STEP;
        Eigen::Quaternionf rotation_offset(Eigen::AngleAxisf(current_angle, Eigen::Vector3f::UnitY()));
        Eigen::Quaternionf current_rot = original_rot * rotation_offset;

        // MATRIZ DEFINITIVA DA FÍSICA! (Crua, Baseada unicamente no TCP)
        Eigen::Affine3f tf_tcp_to_world = Eigen::Translation3f(grasp_pos) * current_rot;
        Eigen::Affine3f tf_world_to_tcp = tf_tcp_to_world.inverse();
        Eigen::Quaternionf q_tcp_world(tf_tcp_to_world.rotation());

        // =========================================================================
        // PUBLICAÇÃO DO ESTADO DE DEBUG (Eixos + Caixas na Posição Real do Teste)
        // =========================================================================
        if (publish_debug && step == 0) {
            visualization_msgs::msg::MarkerArray debug_ma;
            auto t_now = this->now();

            visualization_msgs::msg::Marker center_mk;
            center_mk.header.frame_id = "map";
            center_mk.header.stamp = t_now;
            center_mk.ns = "check_collision_center";
            center_mk.id = 0;
            center_mk.type = visualization_msgs::msg::Marker::SPHERE;
            center_mk.action = visualization_msgs::msg::Marker::ADD;
            center_mk.pose.position.x = grasp_pos.x();
            center_mk.pose.position.y = grasp_pos.y();
            center_mk.pose.position.z = grasp_pos.z();
            center_mk.scale.x = 0.015; center_mk.scale.y = 0.015; center_mk.scale.z = 0.015;
            center_mk.color.r = 1.0; center_mk.color.g = 1.0; center_mk.color.b = 0.0; center_mk.color.a = 1.0;
            debug_ma.markers.push_back(center_mk);

            auto create_axis_arrow = [&](int id, Eigen::Vector3f end_local, float r, float g, float b) {
                visualization_msgs::msg::Marker arrow_mk;
                arrow_mk.header.frame_id = "map";
                arrow_mk.header.stamp = t_now;
                arrow_mk.ns = "check_collision_axes";
                arrow_mk.id = id;
                arrow_mk.type = visualization_msgs::msg::Marker::ARROW;
                arrow_mk.action = visualization_msgs::msg::Marker::ADD;
                
                arrow_mk.pose.position.x = grasp_pos.x();
                arrow_mk.pose.position.y = grasp_pos.y();
                arrow_mk.pose.position.z = grasp_pos.z();
                arrow_mk.pose.orientation.x = q_tcp_world.x();
                arrow_mk.pose.orientation.y = q_tcp_world.y();
                arrow_mk.pose.orientation.z = q_tcp_world.z();
                arrow_mk.pose.orientation.w = q_tcp_world.w();
                
                geometry_msgs::msg::Point p1, p2;
                p1.x = 0.0; p1.y = 0.0; p1.z = 0.0; 
                p2.x = end_local.x(); p2.y = end_local.y(); p2.z = end_local.z(); 
                
                arrow_mk.points.push_back(p1);
                arrow_mk.points.push_back(p2);
                
                arrow_mk.scale.x = 0.004; arrow_mk.scale.y = 0.008; arrow_mk.scale.z = 0.0; 
                arrow_mk.color.r = r; arrow_mk.color.g = g; arrow_mk.color.b = b; arrow_mk.color.a = 1.0;
                return arrow_mk;
            };

            float axis_len = 0.06f; 
            debug_ma.markers.push_back(create_axis_arrow(1, Eigen::Vector3f(axis_len, 0.0, 0.0), 1.0, 0.0, 0.0)); // X (Vermelho)
            debug_ma.markers.push_back(create_axis_arrow(2, Eigen::Vector3f(0.0, axis_len, 0.0), 0.0, 1.0, 0.0)); // Y (Verde)
            debug_ma.markers.push_back(create_axis_arrow(3, Eigen::Vector3f(0.0, 0.0, axis_len), 0.0, 0.0, 1.0)); // Z (Azul)

            for(size_t b = 0; b < gripper_boxes_.size(); b++) {
                visualization_msgs::msg::Marker box_mk;
                box_mk.header.frame_id = "map";
                box_mk.header.stamp = t_now;
                box_mk.ns = "check_collision_boxes";
                box_mk.id = b;
                box_mk.type = visualization_msgs::msg::Marker::CUBE;
                box_mk.action = visualization_msgs::msg::Marker::ADD;

                Eigen::Vector3f cw = tf_tcp_to_world * gripper_boxes_[b].center;
                box_mk.pose.position.x = cw.x();
                box_mk.pose.position.y = cw.y();
                box_mk.pose.position.z = cw.z();

                box_mk.pose.orientation.x = q_tcp_world.x();
                box_mk.pose.orientation.y = q_tcp_world.y();
                box_mk.pose.orientation.z = q_tcp_world.z();
                box_mk.pose.orientation.w = q_tcp_world.w();

                box_mk.scale.x = gripper_boxes_[b].dimensions.x();
                box_mk.scale.y = gripper_boxes_[b].dimensions.y();
                box_mk.scale.z = gripper_boxes_[b].dimensions.z();

                box_mk.color.r = 1.0; box_mk.color.g = 0.0; box_mk.color.b = 1.0; 
                box_mk.color.a = 0.6;
                debug_ma.markers.push_back(box_mk);
            }

            if (debug_marker_pub_) {
                debug_marker_pub_->publish(debug_ma);
            }
        }

        bool collision_in_this_angle = false;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr debug_cloud;
        
        if (publish_debug && step == 0) {
            debug_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
            debug_cloud->header.frame_id = "map";
            debug_cloud->reserve(pointIdx.size());
        }

        for (int idx : pointIdx)
        {
            const auto& pt = cloud_points[idx];
            Eigen::Vector3f p_world(pt.x, pt.y, pt.z);
            
            // Joga o ponto de volta para dentro do referencial da garra (Matemática pura, sem visual_tf lixo)
            Eigen::Vector3f p_local = tf_world_to_tcp * p_world;

            bool point_is_inside = false;
            for (const auto& box : gripper_boxes_)
            {
                // SE O PONTO INVADIR A CAIXA UM MILÍMETRO QUE SEJA, A CHAPA ESQUENTA
                if (p_local.x() >= (box.min_pt.x() - MARGIN) && p_local.x() <= (box.max_pt.x() + MARGIN) &&
                    p_local.y() >= (box.min_pt.y() - MARGIN) && p_local.y() <= (box.max_pt.y() + MARGIN) &&
                    p_local.z() >= (box.min_pt.z() - MARGIN) && p_local.z() <= (box.max_pt.z() + MARGIN))
                {
                    point_is_inside = true;
                    collision_in_this_angle = true;
                    break; // Ponto mortífero detectado
                }
            }

            if (publish_debug && debug_cloud) {
                pcl::PointXYZRGB p_vis;
                p_vis.x = pt.x; p_vis.y = pt.y; p_vis.z = pt.z;
                if (point_is_inside) { p_vis.r = 255; p_vis.g = 0; p_vis.b = 0; } // Vermelho = Colisão Feroz
                else                 { p_vis.r = 0; p_vis.g = 255; p_vis.b = 0; } // Verde = Seguro
                debug_cloud->points.push_back(p_vis);
            }

            // Otimização: Se achou colisão e não é debug, mata o teste desse ângulo na hora
            if (collision_in_this_angle && !publish_debug) break;
        }

        if (publish_debug && debug_cloud && !debug_cloud->empty() && step == 0) {
            sensor_msgs::msg::PointCloud2 msg;
            pcl::toROSMsg(*debug_cloud, msg);
            msg.header.stamp = this->now();
            pub_debug_collision_->publish(msg);
        }

        // Se passar ileso pela caixa de colisão, achamos nosso campeão
        if (!collision_in_this_angle) {
            grasp.pose_center.orientation.w = current_rot.w();
            grasp.pose_center.orientation.x = current_rot.x();
            grasp.pose_center.orientation.y = current_rot.y();
            grasp.pose_center.orientation.z = current_rot.z();
            final_collision_state = false; 
            break; 
        }
    }

    return !final_collision_state; // Retorna true (válido) se não houver colisão
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

    int expansion_rad = std::ceil(cylinder_radius_ * inv_voxel_size);
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
    float voxel_check_threshold = cylinder_radius_ + voxel_radius; 
    float voxel_check_threshold_squared = voxel_check_threshold * voxel_check_threshold;
    float cylinder_radius_sq = cylinder_radius_ * cylinder_radius_;
    float max_scan_dist = max_gripper_width_ * 1.1f;
    int r_int = std::ceil(cylinder_radius_ * inv_voxel_size);
    int r_sq = r_int * r_int;

    const float voxel_extent = (voxel_size * 1.73205f) * 0.5f; 
    const float safe_radius_sq = (cylinder_radius_ - voxel_extent) * (cylinder_radius_ - voxel_extent);
    const bool use_fast_check = (cylinder_radius_ > voxel_extent); 

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

                    
                    sensor_msgs::msg::PointCloud2 m; 
                    pcl::toROSMsg(*stored_cloud_, m); 
                    m.header.stamp = t_now; 
                    m.header.frame_id = "map"; 
                    pub_cloud_->publish(m);

                    
                    visualization_msgs::msg::Marker base_marker;
                    base_marker.header.frame_id = "map";
                    base_marker.header.stamp = t_now;
                    base_marker.action = visualization_msgs::msg::Marker::ADD;
                    base_marker.lifetime = rclcpp::Duration::from_seconds(0); 

                 
                    StepAnalysis debug_entry = analyzeLocalCylinder(local.inliers_entry, center_entry, current_ray_dir);
                    StepAnalysis debug_exit;
                    if (real_thickness > 0.001f) {
                        debug_exit = analyzeLocalCylinder(local.inliers_exit, center_exit, current_ray_dir);
                    }

                    
                    visualization_msgs::msg::Marker pivot_mk = base_marker;
                    pivot_mk.ns = "debug_anim_pivot"; 
                    pivot_mk.id = 1;
                    pivot_mk.type = visualization_msgs::msg::Marker::SPHERE;
                    pivot_mk.pose.position.x = PIVOT_POINT.x(); 
                    pivot_mk.pose.position.y = PIVOT_POINT.y(); 
                    pivot_mk.pose.position.z = PIVOT_POINT.z();
                    pivot_mk.pose.orientation.w = 1.0;
                    pivot_mk.scale.x = 0.005; pivot_mk.scale.y = 0.005; pivot_mk.scale.z = 0.005;
                    pivot_mk.color.a = 1.0; pivot_mk.color.r = 0.0; pivot_mk.color.g = 1.0; pivot_mk.color.b = 1.0; 
                    markers.markers.push_back(pivot_mk);

                    visualization_msgs::msg::Marker origin_mk = base_marker;
                    origin_mk.ns = "current_ray_origin"; 
                    origin_mk.id = 1;
                    origin_mk.type = visualization_msgs::msg::Marker::SPHERE;
                    origin_mk.pose.position.x = current_ray_origin.x(); 
                    origin_mk.pose.position.y = current_ray_origin.y(); 
                    origin_mk.pose.position.z = current_ray_origin.z();
                    origin_mk.pose.orientation.w = 1.0;
                    origin_mk.scale.x = 0.005; origin_mk.scale.y = 0.005; origin_mk.scale.z = 0.005;
                    origin_mk.color.a = 1.0; origin_mk.color.r = 1.0; origin_mk.color.g = 0.0; origin_mk.color.b = 1.0; 
                    markers.markers.push_back(origin_mk);

                   
                    Eigen::Vector3f pos_entry = current_ray_origin + current_ray_dir * exact_t_min;
                    Eigen::Vector3f pos_exit  = current_ray_origin + current_ray_dir * exact_t_max;

                    visualization_msgs::msg::Marker sphere_entry = base_marker;
                    sphere_entry.ns = "debug_anim_sphere_entry";
                    sphere_entry.id = 6;
                    sphere_entry.type = visualization_msgs::msg::Marker::SPHERE;
                    sphere_entry.pose.position.x = pos_entry.x();
                    sphere_entry.pose.position.y = pos_entry.y();
                    sphere_entry.pose.position.z = pos_entry.z();
                    sphere_entry.pose.orientation.w = 1.0;
                    sphere_entry.scale.x = cylinder_radius_ * 2.0; 
                    sphere_entry.scale.y = cylinder_radius_ * 2.0; 
                    sphere_entry.scale.z = cylinder_radius_ * 2.0;
                    sphere_entry.color.a = 0.3; sphere_entry.color.r = 0.0; sphere_entry.color.g = 0.5; sphere_entry.color.b = 1.0;
                    markers.markers.push_back(sphere_entry);

                    if ((exact_t_max - exact_t_min) > 0.005) 
                    {
                        visualization_msgs::msg::Marker sphere_exit = sphere_entry; 
                        sphere_exit.ns = "debug_anim_sphere_exit";
                        sphere_exit.id = 7;
                        sphere_exit.pose.position.x = pos_exit.x();
                        sphere_exit.pose.position.y = pos_exit.y();
                        sphere_exit.pose.position.z = pos_exit.z();
                        sphere_exit.color.r = 1.0; sphere_exit.color.g = 0.5; sphere_exit.color.b = 0.0;
                        markers.markers.push_back(sphere_exit);
                    }

                   
                    visualization_msgs::msg::Marker ray_mk = base_marker;
                    ray_mk.ns = "debug_anim_ray"; 
                    ray_mk.id = 2;
                    ray_mk.type = visualization_msgs::msg::Marker::ARROW;
                    
                    geometry_msgs::msg::Point p_start, p_end;
                    p_start.x = current_ray_origin.x(); p_start.y = current_ray_origin.y(); p_start.z = current_ray_origin.z();
                    Eigen::Vector3f visual_end = current_ray_origin + (current_ray_dir * (DISTANCE_TO_PIVOT + 0.05f));
                    p_end.x = visual_end.x(); p_end.y = visual_end.y(); p_end.z = visual_end.z();

                    ray_mk.points.push_back(p_start); 
                    ray_mk.points.push_back(p_end);
                    ray_mk.scale.x = 0.003; ray_mk.scale.y = 0.006; ray_mk.scale.z = 0.01;  
                    ray_mk.color.a = 0.8; ray_mk.color.r = 1.0; ray_mk.color.g = 1.0; ray_mk.color.b = 0.0;
                    markers.markers.push_back(ray_mk);

                   
                    if (debug_entry.valid) {
                        visualization_msgs::msg::Marker norm_entry = base_marker;
                        norm_entry.ns = "debug_anim_normal_entry";
                        norm_entry.id = 3;
                        norm_entry.type = visualization_msgs::msg::Marker::ARROW;

                        geometry_msgs::msg::Point p_center, p_norm;
                        p_center.x = debug_entry.center.x(); 
                        p_center.y = debug_entry.center.y(); 
                        p_center.z = debug_entry.center.z();

                        Eigen::Vector3f n_end = debug_entry.center + (debug_entry.normal_vector * 0.04f);
                        p_norm.x = n_end.x(); p_norm.y = n_end.y(); p_norm.z = n_end.z();

                        norm_entry.points.push_back(p_center);
                        norm_entry.points.push_back(p_norm);
                        norm_entry.scale.x = 0.002; norm_entry.scale.y = 0.004; norm_entry.scale.z = 0.0;
                        norm_entry.color.a = 1.0; norm_entry.color.r = 1.0; norm_entry.color.g = 0.0; norm_entry.color.b = 0.0; 
                        markers.markers.push_back(norm_entry);
                    }

                   
                    if (debug_exit.valid) {
                        visualization_msgs::msg::Marker norm_exit = base_marker;
                        norm_exit.ns = "debug_anim_normal_exit";
                        norm_exit.id = 4;
                        norm_exit.type = visualization_msgs::msg::Marker::ARROW;

                        geometry_msgs::msg::Point p_center, p_norm;
                        p_center.x = debug_exit.center.x(); 
                        p_center.y = debug_exit.center.y(); 
                        p_center.z = debug_exit.center.z();

                        Eigen::Vector3f n_end = debug_exit.center + (debug_exit.normal_vector * 0.04f);
                        p_norm.x = n_end.x(); p_norm.y = n_end.y(); p_norm.z = n_end.z();

                        norm_exit.points.push_back(p_center);
                        norm_exit.points.push_back(p_norm);
                        norm_exit.scale.x = 0.002; norm_exit.scale.y = 0.004; norm_exit.scale.z = 0.0;
                        norm_exit.color.a = 1.0; norm_exit.color.r = 1.0; norm_exit.color.g = 0.5; norm_exit.color.b = 0.0;
                        markers.markers.push_back(norm_exit);
                    }

                  
                    
                    if (local.inliers_entry && !local.inliers_entry->empty()) 
                    {
                        visualization_msgs::msg::Marker entry_mk = base_marker;
                        entry_mk.ns = "debug_anim_inliers_entry"; 
                        entry_mk.id = 4;
                        entry_mk.type = visualization_msgs::msg::Marker::POINTS;
                        entry_mk.scale.x = 0.003; entry_mk.scale.y = 0.003;
                        entry_mk.color.a = 1.0; entry_mk.color.r = 0.0; entry_mk.color.g = 1.0; entry_mk.color.b = 0.0; 

                        entry_mk.points.reserve(local.inliers_entry->size());
                        for (const auto& p : local.inliers_entry->points) {
                            geometry_msgs::msg::Point gp;
                            gp.x = p.x; gp.y = p.y; gp.z = p.z;
                            entry_mk.points.push_back(gp);
                        }
                        markers.markers.push_back(entry_mk);
                    }

                    if (local.inliers_exit && !local.inliers_exit->empty()) 
                    {
                        visualization_msgs::msg::Marker exit_mk = base_marker;
                        exit_mk.ns = "debug_anim_inliers_exit"; 
                        exit_mk.id = 5;
                        exit_mk.type = visualization_msgs::msg::Marker::POINTS;
                        exit_mk.scale.x = 0.003; exit_mk.scale.y = 0.003;
                        exit_mk.color.a = 1.0; exit_mk.color.r = 1.0; exit_mk.color.g = 0.0; exit_mk.color.b = 1.0; // Roxo

                        exit_mk.points.reserve(local.inliers_exit->size());
                        for (const auto& p : local.inliers_exit->points) {
                            geometry_msgs::msg::Point gp;
                            gp.x = p.x; gp.y = p.y; gp.z = p.z;
                            exit_mk.points.push_back(gp);
                        }
                        markers.markers.push_back(exit_mk);
                    }

                    
                    if (!local.voxels_no_caminho.empty())
                    {
                        visualization_msgs::msg::Marker vox_mk = base_marker;
                        vox_mk.ns = "debug_anim_voxels_path";
                        vox_mk.id = 9;
                        vox_mk.type = visualization_msgs::msg::Marker::CUBE_LIST; 
                        vox_mk.scale.x = 0.0075; vox_mk.scale.y = 0.0075; vox_mk.scale.z = 0.0075;
                        vox_mk.color.r = 0.0; vox_mk.color.g = 1.0; vox_mk.color.b = 1.0; vox_mk.color.a = 0.15;

                        vox_mk.points.reserve(local.voxels_no_caminho.size());
                        for (const auto* bucket_ptr : local.voxels_no_caminho) {
                            if (bucket_ptr) {
                                geometry_msgs::msg::Point p;
                                p.x = bucket_ptr->center.x(); p.y = bucket_ptr->center.y(); p.z = bucket_ptr->center.z();
                                vox_mk.points.push_back(p);
                            }
                        }
                        markers.markers.push_back(vox_mk);
                    }

                    debug_marker_pub_->publish(markers);
                    std::this_thread::sleep_for(std::chrono::milliseconds(animation_delay_ms_));
                }


                
                auto t5 = std::chrono::high_resolution_clock::now();
                std::vector<StepAnalysis> steps; steps.reserve(2); 
                
                StepAnalysis res_entry = analyzeLocalCylinder(local.inliers_entry, center_entry, current_ray_dir);
                if (res_entry.valid) steps.push_back(res_entry);

                if (real_thickness > 0.001f) { 
                    StepAnalysis res_exit = analyzeLocalCylinder(local.inliers_exit, center_exit, current_ray_dir);
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
    
    size_t num_to_print = std::max((size_t)5, best_grasps_.size());
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
} // namespace vision

RCLCPP_COMPONENTS_REGISTER_NODE(vision::GenerateGraspPoses)