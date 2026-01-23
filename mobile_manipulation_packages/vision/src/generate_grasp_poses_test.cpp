#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>
#include <map>
#include <atomic>
// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <pcl/search/kdtree.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std::chrono_literals;

struct StepAnalysis {
    bool valid;
    int point_count;
    float angle_to_normal_deg; 
    Eigen::Vector3f normal_vector; 
    float symmetry_score;
    float curvature; 
    Eigen::Vector3f center;
};

struct ScoredGrasp {
    geometry_msgs::msg::Pose pose_center;
    geometry_msgs::msg::Pose pose_finger1;
    geometry_msgs::msg::Pose pose_finger2;
    
    // Armazena a geometria da estrutura para desenhar depois
    Eigen::Vector3f struct_finger1_back;
    Eigen::Vector3f struct_finger2_back;
    
    double total_score;
    double entry_angle;
    double exit_angle;
    double entry_planarity;
    double exit_planarity;
    Eigen::Vector3f entry_normal;

    // Auxiliares para recálculo na Fase 2
    Eigen::Vector3f raw_ray_dir;
    Eigen::Vector3f raw_p_f1;
    Eigen::Vector3f raw_p_f2;
};

class BestGraspFinder : public rclcpp::Node
{
public:
    BestGraspFinder() : Node("best_grasp_finder") 
    {
        // --- Parametros existentes ---
        this->declare_parameter<float>("grid_res", 0.02);
        this->declare_parameter<float>("cloud_voxel_size", 0.003);
        this->declare_parameter<float>("cylinder_radius", 0.015); 
        this->declare_parameter<float>("cylinder_height", 0.015);
        this->declare_parameter<float>("analysis_step_size", 0.01);
        this->declare_parameter<float>("max_gripper_width", 0.04); 
        this->declare_parameter<float>("finger_offset", 0.03); 
        this->declare_parameter<float>("gripper_finger_depth", 0.08); 
        this->declare_parameter<int>("gripper_collision_threshold", 5); 
        this->declare_parameter<float>("gripper_structure_thickness", 0.005); 
        this->declare_parameter<int>("num_collision_checks", 10);
        this->declare_parameter<int>("min_points_per_segment", 6);
        this->declare_parameter<float>("weight_orientation", 0.6); 
        this->declare_parameter<float>("weight_symmetry", 0.2);
        this->declare_parameter<float>("weight_planarity", 0.2);
        this->declare_parameter<bool>("use_mls_smoothing", false); 
        this->declare_parameter<float>("mls_radius", 0.03);
        this->declare_parameter<int>("num_best_grasps", 5);

        // --- NOVO: Parametro do Topico ---
        this->declare_parameter<std::string>("cloud_topic", "/camera/depth/color/points");

        // Publishers
        pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("processed_cloud", 10);
        pub_rays_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("candidate_rays", 10);
        pub_bbox_  = this->create_publisher<visualization_msgs::msg::Marker>("bounding_box", 10);
        pub_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("best_grasps_markers", 10);
        pub_poses_ = this->create_publisher<geometry_msgs::msg::PoseArray>("best_grasps_poses", 10);
        pub_center_oriented_poses_ = this->create_publisher<geometry_msgs::msg::PoseArray>("best_grasps_center_oriented", 10);

        stored_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        
        // --- CONFIGURAÇÃO DO SUBSCRIBER E QOS ---
        // SensorDataQoS é 'Best Effort' e 'Volatile', ideal para stream de dados.
        // keep_last(1) garante que se o processamento for lento, pegamos apenas o mais novo, descartando o buffer antigo.
        rclcpp::QoS qos_profile = rclcpp::SensorDataQoS();
        qos_profile.keep_last(1);

     
        // Inicializa flag de controle de thread
        is_processing_ = false;

        sub_cloud_Input_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "mapped_object",
            qos_profile,
            std::bind(&BestGraspFinder::cloud_callback, this, std::placeholders::_1));

        
        // Timer apenas para republicar visualização se necessário (opcional), 
        // mas o calculo pesado agora ocorre no callback.
        // Se quiser que a visualização seja atualizada apenas quando chegar nuvem nova, pode remover o timer.
        timer_ = this->create_wall_timer(1000ms, std::bind(&BestGraspFinder::timerCallback, this));
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Subscriber para a nuvem de entrada
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_Input_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rays_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_bbox_; 
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_center_oriented_poses_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud_;
    std::vector<geometry_msgs::msg::Pose> all_candidates_;
    std::vector<geometry_msgs::msg::Pose> hit_candidates_;
    std::vector<ScoredGrasp> best_grasps_;
    bool has_best_ = false;
    Eigen::Vector4f min_pt_, max_pt_;

    // Flag atômica para descartar mensagens se estiver ocupado
    std::atomic<bool> is_processing_;

    // Substitui o antigo loadAndProcess
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Lógica de descarte: Se já estiver processando (true), retorna imediatamente.
        // compare_exchange_strong tenta setar para true APENAS se estiver false.
        bool expected = false;
        if (!is_processing_.compare_exchange_strong(expected, true)) {
            // RCLCPP_WARN(this->get_logger(), "Processamento ocupado. Descartando nuvem.");
            return; 
        }

        try {
            // Conversão ROS2 -> PCL
            pcl::PCLPointCloud2 pcl_pc2;
            pcl_conversions::toPCL(*msg, pcl_pc2);
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromPCLPointCloud2(pcl_pc2, *temp_cloud);

            if (temp_cloud->empty()) {
                is_processing_ = false;
                return;
            }

            // Voxel Grid Filter
            pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            float voxel_size = this->get_parameter("cloud_voxel_size").as_double();

            if (voxel_size > 0.0001) 
            {
                pcl::VoxelGrid<pcl::PointXYZ> sor;
                sor.setInputCloud(temp_cloud);
                sor.setLeafSize(voxel_size, voxel_size, voxel_size);
                sor.filter(*voxel_cloud);
            } 
            else 
            {
                *voxel_cloud = *temp_cloud;
            }

            // MLS Smoothing (opcional)
            if (this->get_parameter("use_mls_smoothing").as_bool()) {
                float mls_rad = this->get_parameter("mls_radius").as_double();
                pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
                pcl::PointCloud<pcl::PointNormal> mls_points;
                pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
                mls.setComputeNormals(true);
                mls.setInputCloud(voxel_cloud);
                mls.setPolynomialOrder(2);
                mls.setSearchMethod(tree);
                mls.setSearchRadius(mls_rad);
                mls.process(mls_points);
                stored_cloud_->clear();
                for (const auto& pt_n : mls_points) {
                    pcl::PointXYZ pt; pt.x = pt_n.x; pt.y = pt_n.y; pt.z = pt_n.z;
                    stored_cloud_->points.push_back(pt);
                }
            } else {
                *stored_cloud_ = *voxel_cloud;
            }
            
            stored_cloud_->header.frame_id = msg->header.frame_id; // Mantém o frame original

            // Calcula limites da nuvem
            pcl::getMinMax3D(*stored_cloud_, min_pt_, max_pt_);
            float padding = 0.02; min_pt_.array() -= padding; max_pt_.array() += padding;

            // Gera raios baseados nos novos limites
            float grid_res = this->get_parameter("grid_res").as_double();
            all_candidates_ = generateOrthogonalRays(min_pt_, max_pt_, grid_res);

            // Executa avaliação
            evaluateGrasps();

            // Opcional: Publicar visualização IMEDIATAMENTE após cálculo para reduzir lag visual
            publishBest(); 

        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Erro ao processar PointCloud");
        }

        // Libera a flag para aceitar nova nuvem
        is_processing_ = false;
    }

    std::vector<geometry_msgs::msg::Pose> generateOrthogonalRays(
        const Eigen::Vector4f& min, const Eigen::Vector4f& max, float res) 
    {
        std::vector<geometry_msgs::msg::Pose> poses;
        auto add_ray = [&](Eigen::Vector3f start, Eigen::Vector3f direction) {
            geometry_msgs::msg::Pose p;
            p.position.x = start.x(); p.position.y = start.y(); p.position.z = start.z();
            Eigen::Quaternionf q; q.setFromTwoVectors(Eigen::Vector3f::UnitX(), direction);
            p.orientation.x = q.x(); p.orientation.y = q.y(); p.orientation.z = q.z(); p.orientation.w = q.w();
            poses.push_back(p);
        };

        // Eixo X
        for(float y = min[1]; y < max[1]; y += res)
            for(float z = min[2]; z < max[2]; z += res) {
                add_ray({min[0], y+res/2, z+res/2}, {1, 0, 0});
                add_ray({max[0], y+res/2, z+res/2}, {-1, 0, 0});
            }
        // Eixo Y
        for(float x = min[0]; x < max[0]; x += res)
            for(float z = min[2]; z < max[2]; z += res) {
                add_ray({x+res/2, min[1], z+res/2}, {0, 1, 0});
                add_ray({x+res/2, max[1], z+res/2}, {0, -1, 0});
            }
        // Eixo Z
        for(float x = min[0]; x < max[0]; x += res)
            for(float y = min[1]; y < max[1]; y += res) 
            {
                add_ray({x+res/2, y+res/2, min[2]}, {0, 0, 1});
                add_ray({x+res/2, y+res/2, max[2]}, {0, 0, -1});
            }
        return poses;
    }

    StepAnalysis analyzeLocalCylinder(
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

        for (const auto& pt : cloud->points) {
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
        int min_pts = this->get_parameter("min_points_per_segment").as_int();
        if (result.point_count <= min_pts) return result;

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

    int countCollisionsOnSegment(const Eigen::Vector3f& start, const Eigen::Vector3f& end, float radius)
    {
        int count = 0;
        Eigen::Vector3f seg_vec = end - start;
        float seg_len_sq = seg_vec.squaredNorm();
        
        if (seg_len_sq < 1e-6) return 0;

        Eigen::Vector3f min_s = start.cwiseMin(end).array() - radius;
        Eigen::Vector3f max_s = start.cwiseMax(end).array() + radius;

        for (const auto& pt : stored_cloud_->points) 
        {
            if (pt.x < min_s.x() || pt.x > max_s.x() ||
                pt.y < min_s.y() || pt.y > max_s.y() ||
                pt.z < min_s.z() || pt.z > max_s.z()) continue;

            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            
            float t = (p - start).dot(seg_vec) / seg_len_sq;
            t = std::max(0.0f, std::min(1.0f, t));
            Eigen::Vector3f projection = start + t * seg_vec;
            
            if ((p - projection).norm() <= radius) {
                count++;
            }
        }
        return count;
    }

    bool checkStructureCollision(const Eigen::Vector3f& p_f1, const Eigen::Vector3f& p_f2, 
                                 const Eigen::Vector3f& p_f1_back, const Eigen::Vector3f& p_f2_back)
    {
        int limit = this->get_parameter("gripper_collision_threshold").as_int();
        float thick = this->get_parameter("gripper_structure_thickness").as_double();
        
        // Verifica dedo 1
        int c1 = countCollisionsOnSegment(p_f1, p_f1_back, thick);
        if (c1 > limit) return true;

        // Verifica dedo 2
        int c2 = countCollisionsOnSegment(p_f2, p_f2_back, thick);
        if ((c1 + c2) > limit) return true;

        // Verifica base (conexão entre backs)
        int c3 = countCollisionsOnSegment(p_f1_back, p_f2_back, thick);
        if ((c1 + c2 + c3) > limit) return true;

        return false;
    }

    void evaluateGrasps()
    {
        float radius = this->get_parameter("cylinder_radius").as_double();
        float max_width = this->get_parameter("max_gripper_width").as_double(); 
        float cyl_height = this->get_parameter("cylinder_height").as_double();
        float step_size = this->get_parameter("analysis_step_size").as_double();
        
        // NOVO: Recupera parametro offset
        float finger_offset = this->get_parameter("finger_offset").as_double();
        
        float gripper_depth = this->get_parameter("gripper_finger_depth").as_double();
        int collision_check_limit = this->get_parameter("num_collision_checks").as_int();

        float w_orient = this->get_parameter("weight_orientation").as_double();
        float w_sym = this->get_parameter("weight_symmetry").as_double();
        float w_plan = this->get_parameter("weight_planarity").as_double();
        int num_to_publish = this->get_parameter("num_best_grasps").as_int();

        std::vector<ScoredGrasp> initial_candidates; 
        hit_candidates_.clear(); 

        RCLCPP_INFO(this->get_logger(), "Fase 1: Avaliando %lu raios (Geometria local)...", all_candidates_.size());

        
        for (const auto& raw_pose : all_candidates_) 
        {
            Eigen::Quaternionf q(raw_pose.orientation.w, raw_pose.orientation.x, raw_pose.orientation.y, raw_pose.orientation.z);
            Eigen::Vector3f ray_origin(raw_pose.position.x, raw_pose.position.y, raw_pose.position.z);
            Eigen::Vector3f ray_dir = q * Eigen::Vector3f::UnitX(); 

            float t_min = 1e6, t_max = -1e6;
            bool hit = false;
            for (const auto& pt : stored_cloud_->points) 
            {
                Eigen::Vector3f p(pt.x, pt.y, pt.z);
                float t = (p - ray_origin).dot(ray_dir);
                if ((p - (ray_origin + t*ray_dir)).norm() < radius) {
                    if (t < t_min) t_min = t;
                    if (t > t_max) t_max = t;
                    hit = true;
                }
            }
            if (!hit || (t_max - t_min) < 0.005) continue;
            
            hit_candidates_.push_back(raw_pose);

            std::vector<StepAnalysis> steps;
            for (float t = t_min; t <= t_max; t += step_size) {
                Eigen::Vector3f center = ray_origin + ray_dir * t;
                StepAnalysis res = analyzeLocalCylinder(stored_cloud_, center, ray_dir, radius, cyl_height);
                if (res.valid) steps.push_back(res);
            }

            if (steps.empty()) continue;

            StepAnalysis& entry = steps.front();
            StepAnalysis& exit = steps.back();

            float object_thickness = (exit.center - entry.center).norm();
            if (object_thickness > max_width) continue; 

            
            Eigen::Vector3f p_f1 = entry.center - ray_dir * finger_offset;
            Eigen::Vector3f p_f2 = exit.center + ray_dir * finger_offset;
            Eigen::Vector3f center_grasp = (p_f1 + p_f2) / 2.0f;
            
            
            float score_ang_entry = 1.0f - (std::min(entry.angle_to_normal_deg, 90.0f) / 90.0f);
            float score_ang_exit  = 1.0f - (std::min(exit.angle_to_normal_deg, 90.0f) / 90.0f);
            float score_plan_entry = std::max(0.0f, 1.0f - (entry.curvature * 20.0f)); 
            float score_plan_exit  = std::max(0.0f, 1.0f - (exit.curvature * 20.0f));
            
            float orient_factor_entry = (score_plan_entry > 0.3) ? 1.0f : 0.5f;
            float orient_factor_exit = (score_plan_exit > 0.3) ? 1.0f : 0.5f;
            float score_sym_entry = entry.symmetry_score;
            float score_sym_exit  = exit.symmetry_score;

            double total = (score_ang_entry * w_orient * orient_factor_entry + score_sym_entry * w_sym + score_plan_entry * w_plan) * 0.5 + 
                           (score_ang_exit * w_orient * orient_factor_exit + score_sym_exit * w_sym + score_plan_exit * w_plan) * 0.5;

            if (object_thickness < 0.015) total *= 0.1;

            ScoredGrasp sg;
            sg.pose_center = raw_pose; sg.pose_center.position.x = center_grasp.x(); sg.pose_center.position.y = center_grasp.y(); sg.pose_center.position.z = center_grasp.z();
            sg.pose_finger1 = raw_pose; sg.pose_finger1.position.x = p_f1.x(); sg.pose_finger1.position.y = p_f1.y(); sg.pose_finger1.position.z = p_f1.z();
            sg.pose_finger2 = raw_pose; sg.pose_finger2.position.x = p_f2.x(); sg.pose_finger2.position.y = p_f2.y(); sg.pose_finger2.position.z = p_f2.z();
            
            
            sg.raw_ray_dir = ray_dir;
            sg.raw_p_f1 = p_f1;
            sg.raw_p_f2 = p_f2;
            
            sg.total_score = total;
            sg.entry_angle = entry.angle_to_normal_deg;
            sg.exit_angle = exit.angle_to_normal_deg;
            sg.entry_planarity = 1.0 - entry.curvature;
            sg.exit_planarity = 1.0 - exit.curvature;
            sg.entry_normal = entry.normal_vector;
            
            initial_candidates.push_back(sg);
        }

        if (initial_candidates.empty()) {
            has_best_ = false; return;
        }

        
        std::sort(initial_candidates.begin(), initial_candidates.end(), 
            [](const ScoredGrasp& a, const ScoredGrasp& b) { return a.total_score > b.total_score; });

        
        RCLCPP_INFO(this->get_logger(), "Fase 2: Verificando colisão estrutural nos top %d...", collision_check_limit);
        
        best_grasps_.clear();
        int checks_performed = 0;
        
        for (auto& sg : initial_candidates) 
        {
            if (checks_performed >= collision_check_limit) break;
            if (best_grasps_.size() >= (size_t)num_to_publish) break;

            checks_performed++;

            Eigen::Vector3f ray_dir = sg.raw_ray_dir;
            
           
            Eigen::Vector3f global_z(0, 0, 1);
            Eigen::Vector3f global_x(1, 0, 0);
            
            
            Eigen::Vector3f proj_z = global_z - ray_dir * global_z.dot(ray_dir);
            
            std::vector<Eigen::Vector3f> candidate_dirs;
            
            if (proj_z.norm() > 0.1) {
                
                Eigen::Vector3f up = proj_z.normalized();
                Eigen::Vector3f side = ray_dir.cross(up).normalized();
                
               
                candidate_dirs.push_back(up);
                candidate_dirs.push_back(side);
                candidate_dirs.push_back(-side);
                candidate_dirs.push_back(-up);
            } else {
                
                Eigen::Vector3f proj_x = global_x - ray_dir * global_x.dot(ray_dir);
                Eigen::Vector3f up_substitute = proj_x.normalized();
                Eigen::Vector3f side_substitute = ray_dir.cross(up_substitute).normalized();
                
                candidate_dirs.push_back(up_substitute);
                candidate_dirs.push_back(side_substitute);
                candidate_dirs.push_back(-up_substitute);
                candidate_dirs.push_back(-side_substitute);
            }

            bool valid_struct = false;
            
           
            for(const auto& dir : candidate_dirs) {
                Eigen::Vector3f p_f1_back = sg.raw_p_f1 + dir * gripper_depth;
                Eigen::Vector3f p_f2_back = sg.raw_p_f2 + dir * gripper_depth;

                if (!checkStructureCollision(sg.raw_p_f1, sg.raw_p_f2, p_f1_back, p_f2_back)) {
                    
                    sg.struct_finger1_back = p_f1_back;
                    sg.struct_finger2_back = p_f2_back;
                    valid_struct = true;
                    break; 
                }
            }

            if (valid_struct) {
                best_grasps_.push_back(sg);
            }
        }
        
        has_best_ = !best_grasps_.empty();
        
        RCLCPP_INFO(this->get_logger(), "Encontrados %lu grasps validos sem colisao.", best_grasps_.size());
        for(size_t i = 0; i < best_grasps_.size(); i++)
        {
            RCLCPP_INFO(this->get_logger(), "#%lu Score: %.2f", i, best_grasps_[i].total_score);
        }
    }

    void timerCallback() 
    {
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
        size_t all_lim = std::min((size_t)100, all_candidates_.size());
        for(size_t i=0; i<all_lim; ++i) {
            size_t idx = (all_candidates_.size() > 100) ? (i * (all_candidates_.size()/100)) : i;
            visualization_msgs::msg::Marker k; 
            k.header.frame_id="world"; k.header.stamp=t; k.ns="rays_scan"; k.id=i; k.type=0; k.action=0; 
            k.scale.x=ray_len * 0.3; k.scale.y=0.001; k.scale.z=0.001; 
            k.color.r=0.5; k.color.g=0.5; k.color.b=0.5; k.color.a=0.2; 
            k.pose = all_candidates_[idx];
            ma_rays.markers.push_back(k);
        }
        pub_rays_->publish(ma_rays);

        if(has_best_) publishBest();
    }

    void publishBest() 
    {
        visualization_msgs::msg::MarkerArray ma; 
        geometry_msgs::msg::PoseArray pose_array_msg;
        pose_array_msg.header.frame_id = "world";
        pose_array_msg.header.stamp = this->now();
        auto t = this->now();

        auto sphere = [&](int id, auto p, float r, float g, float b, float alpha) {
            visualization_msgs::msg::Marker m; m.header.frame_id="world"; m.header.stamp=t; 
            m.ns="fingers"; m.id=id; m.type=2; m.action=0; m.pose=p; 
            m.scale.x=0.025; m.scale.y=0.025; m.scale.z=0.025; 
            m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=alpha; 
            return m;
        };

        for(size_t i = 0; i < best_grasps_.size(); i++)
        {
            const auto& grasp = best_grasps_[i];
            pose_array_msg.poses.push_back(grasp.pose_center);

            float r=0, g=1, b=0, alpha=0.6;
            if (i == 0) { r=0; g=0; b=1; alpha=1.0; }
            int base_id = i * 20;

            ma.markers.push_back(sphere(base_id + 0, grasp.pose_finger1, r, g, b, alpha)); 
            ma.markers.push_back(sphere(base_id + 1, grasp.pose_finger2, r, g, b, alpha)); 
            
            visualization_msgs::msg::Marker l; 
            l.header.frame_id="world"; l.header.stamp=t; l.ns="lines"; l.id=base_id+2; l.type=5; l.action=0; 
            l.scale.x=0.002; l.color.r=r; l.color.g=g; l.color.b=b; l.color.a=alpha;
            l.points.push_back(grasp.pose_finger1.position); 
            l.points.push_back(grasp.pose_finger2.position);
            ma.markers.push_back(l);

            visualization_msgs::msg::Marker struc;
            struc.header.frame_id="world"; struc.header.stamp=t; struc.ns="structure"; struc.id=base_id+5; 
            struc.type=visualization_msgs::msg::Marker::LINE_STRIP; struc.action=0;
            struc.scale.x = 0.005; 
            struc.color.r=r; struc.color.g=g; struc.color.b=b; struc.color.a=0.8;
            
            geometry_msgs::msg::Point p1, p1b, p2b, p2;
            p1 = grasp.pose_finger1.position;
            p1b.x = grasp.struct_finger1_back.x(); p1b.y = grasp.struct_finger1_back.y(); p1b.z = grasp.struct_finger1_back.z();
            p2b.x = grasp.struct_finger2_back.x(); p2b.y = grasp.struct_finger2_back.y(); p2b.z = grasp.struct_finger2_back.z();
            p2 = grasp.pose_finger2.position;

            struc.points.push_back(p1);
            struc.points.push_back(p1b);
            struc.points.push_back(p2b);
            struc.points.push_back(p2);
            ma.markers.push_back(struc);

            if (i == 0) {
                visualization_msgs::msg::Marker nm; nm.header.frame_id="world"; nm.header.stamp=t; nm.ns="normal"; nm.id=base_id+4; nm.type=0; nm.action=0;
                Eigen::Quaternionf q1(grasp.pose_finger1.orientation.w, grasp.pose_finger1.orientation.x, grasp.pose_finger1.orientation.y, grasp.pose_finger1.orientation.z);
                Eigen::Vector3f ray_d = q1 * Eigen::Vector3f::UnitX();
                Eigen::Vector3f start_n = Eigen::Vector3f(grasp.pose_finger1.position.x, grasp.pose_finger1.position.y, grasp.pose_finger1.position.z) + ray_d * 0.02;
                nm.pose.position.x = start_n.x(); nm.pose.position.y = start_n.y(); nm.pose.position.z = start_n.z();
                Eigen::Quaternionf q_n; q_n.setFromTwoVectors(Eigen::Vector3f::UnitX(), grasp.entry_normal);
                nm.pose.orientation.x=q_n.x(); nm.pose.orientation.y=q_n.y(); nm.pose.orientation.z=q_n.z(); nm.pose.orientation.w=q_n.w();
                nm.scale.x=0.05; nm.scale.y=0.005; nm.scale.z=0.005; nm.color.r=1.0; nm.color.a=1.0;
                ma.markers.push_back(nm);
            }

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
        
        pub_markers_->publish(ma);
        pub_poses_->publish(pose_array_msg);
    }
};

int main(int argc, char ** argv) 
{ 
    rclcpp::init(argc, argv); 
    rclcpp::spin(std::make_shared<BestGraspFinder>()); 
    rclcpp::shutdown(); 
    return 0; 
}