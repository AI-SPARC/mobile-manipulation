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

struct StepAnalysis
{
    bool valid;
    int point_count;
    float angle_to_normal_deg; 
    Eigen::Vector3f normal_vector; 
    float symmetry_score;
    float curvature; 
    Eigen::Vector3f center;
};

struct ScoredGrasp
{
    geometry_msgs::msg::Pose pose_center;
    geometry_msgs::msg::Pose pose_finger1;
    geometry_msgs::msg::Pose pose_finger2;
    
    double total_score;
    double entry_angle;
    double exit_angle;
    double entry_planarity;
    double exit_planarity;
    Eigen::Vector3f entry_normal;

    Eigen::Vector3f raw_ray_dir;
    Eigen::Vector3f raw_p_f1;
    Eigen::Vector3f raw_p_f2;
};

class BestGraspFinder : public rclcpp::Node
{
public:
    BestGraspFinder() : Node("best_grasp_finder") 
    {
        this->declare_parameter<std::string>("pcd_path", "/home/momesso/pibic/nuvem.pcd");
        

        this->declare_parameter<std::string>("gripper_mesh_path", "/home/momesso/hand_and_finger.stl");
        this->declare_parameter<float>("gripper_mesh_scale", 0.001);
        
        
        this->declare_parameter<float>("mesh_offset_x", 0.0);
        this->declare_parameter<float>("mesh_offset_y", 0.0);
        this->declare_parameter<float>("mesh_offset_z", -0.025);
        
        
        this->declare_parameter<float>("mesh_rot_roll", 0.0);
        this->declare_parameter<float>("mesh_rot_pitch", 0.0);
        this->declare_parameter<float>("mesh_rot_yaw", 1.57); 

   
        this->declare_parameter<float>("grid_res", 0.02);
        this->declare_parameter<float>("cloud_voxel_size", 0.003);
        
        this->declare_parameter<float>("cylinder_radius", 0.015); 
        this->declare_parameter<float>("cylinder_height", 0.015);
        this->declare_parameter<float>("analysis_step_size", 0.01);
        
        this->declare_parameter<float>("max_gripper_width", 0.07); 
        
        this->declare_parameter<float>("finger_offset", 0.03); 
        
        this->declare_parameter<int>("min_points_per_segment", 6);
        this->declare_parameter<float>("weight_orientation", 0.6); 
        this->declare_parameter<float>("weight_symmetry", 0.2);
        this->declare_parameter<float>("weight_planarity", 0.2);
        
        this->declare_parameter<bool>("use_mls_smoothing", false); 
        this->declare_parameter<float>("mls_radius", 0.03);

        this->declare_parameter<int>("num_best_grasps", 5);

        pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("input_cloud", 10);
        pub_rays_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("candidate_rays", 10);
        pub_bbox_  = this->create_publisher<visualization_msgs::msg::Marker>("bounding_box", 10);
        pub_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("best_grasps_markers", 10);
        pub_poses_ = this->create_publisher<geometry_msgs::msg::PoseArray>("best_grasps_poses", 10);

        stored_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        std::string pcd_path = this->get_parameter("pcd_path").as_string();
        loadAndProcess(pcd_path);
        
        timer_ = this->create_wall_timer(1000ms, std::bind(&BestGraspFinder::timerCallback, this));
    }

private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rays_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_bbox_; 
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud_;
    std::vector<geometry_msgs::msg::Pose> all_candidates_;
    std::vector<geometry_msgs::msg::Pose> hit_candidates_;
    std::vector<ScoredGrasp> best_grasps_;
    bool has_best_ = false;
    Eigen::Vector4f min_pt_, max_pt_;

    void loadAndProcess(const std::string& path)
    {
        RCLCPP_INFO(this->get_logger(), "Lendo PCD...");
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *temp_cloud) == -1) 
        {
            return;
        }

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

        if (this->get_parameter("use_mls_smoothing").as_bool()) 
        {
            RCLCPP_INFO(this->get_logger(), "Aplicando MLS...");
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
            for (const auto& pt_n : mls_points) 
            {
                pcl::PointXYZ pt; pt.x = pt_n.x; pt.y = pt_n.y; pt.z = pt_n.z;
                stored_cloud_->points.push_back(pt);
            }
        } 
        else 
        {
            *stored_cloud_ = *voxel_cloud;
        }
        
        stored_cloud_->header.frame_id = "world";

        pcl::getMinMax3D(*stored_cloud_, min_pt_, max_pt_);
        float padding = 0.02; min_pt_.array() -= padding; max_pt_.array() += padding;

        float grid_res = this->get_parameter("grid_res").as_double();
        
        all_candidates_ = generateOrthogonalRays(min_pt_, max_pt_, grid_res);

        evaluateGrasps();
    }

    std::vector<geometry_msgs::msg::Pose> generateOrthogonalRays(
        const Eigen::Vector4f& min, const Eigen::Vector4f& max, float res) 
    {
        std::vector<geometry_msgs::msg::Pose> poses;
        auto add_ray = [&](Eigen::Vector3f start, Eigen::Vector3f direction) 
        {
            geometry_msgs::msg::Pose p;
            p.position.x = start.x(); p.position.y = start.y(); p.position.z = start.z();
            Eigen::Quaternionf q; q.setFromTwoVectors(Eigen::Vector3f::UnitX(), direction);
            p.orientation.x = q.x(); p.orientation.y = q.y(); p.orientation.z = q.z(); p.orientation.w = q.w();
            poses.push_back(p);
        };

        // Eixo X
        for(float y = min[1]; y < max[1]; y += res)
        {
            for(float z = min[2]; z < max[2]; z += res) 
            {
                add_ray({min[0], y+res/2, z+res/2}, {1, 0, 0});
                add_ray({max[0], y+res/2, z+res/2}, {-1, 0, 0});
            }
        }
        // Eixo Y
        for(float x = min[0]; x < max[0]; x += res)
        {
            for(float z = min[2]; z < max[2]; z += res) 
            {
                add_ray({x+res/2, min[1], z+res/2}, {0, 1, 0});
                add_ray({x+res/2, max[1], z+res/2}, {0, -1, 0});
            }
        }
        // Eixo Z
        for(float x = min[0]; x < max[0]; x += res)
        {
            for(float y = min[1]; y < max[1]; y += res) 
            {
                add_ray({x+res/2, y+res/2, min[2]}, {0, 0, 1});
                add_ray({x+res/2, y+res/2, max[2]}, {0, 0, -1});
            }
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

    // Função Placeholder - Por enquanto apenas retorna true
    bool check_collision(const ScoredGrasp& grasp)
    {
        // FUTURO: Aqui entrará a lógica de verificar colisão da Mesh carregada
        return true; 
    }

    void evaluateGrasps()
    {
        float radius = this->get_parameter("cylinder_radius").as_double();
        float max_width = this->get_parameter("max_gripper_width").as_double(); 
        float cyl_height = this->get_parameter("cylinder_height").as_double();
        float step_size = this->get_parameter("analysis_step_size").as_double();
        float finger_offset = this->get_parameter("finger_offset").as_double();
        
        float w_orient = this->get_parameter("weight_orientation").as_double();
        float w_sym = this->get_parameter("weight_symmetry").as_double();
        float w_plan = this->get_parameter("weight_planarity").as_double();
        int num_to_publish = this->get_parameter("num_best_grasps").as_int();

        std::vector<ScoredGrasp> initial_candidates; 
        hit_candidates_.clear(); 

        RCLCPP_INFO(this->get_logger(), "Fase 1: Avaliando %lu raios (Geometria local)...", all_candidates_.size());

        // --- INICIO DA LÓGICA DE SCORE (INTACTA) ---
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
                if ((p - (ray_origin + t*ray_dir)).norm() < radius) 
                {
                    if (t < t_min) t_min = t;
                    if (t > t_max) t_max = t;
                    hit = true;
                }
            }
            if (!hit || (t_max - t_min) < 0.005) continue;
            
            hit_candidates_.push_back(raw_pose);

            std::vector<StepAnalysis> steps;
            for (float t = t_min; t <= t_max; t += step_size) 
            {
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
        // --- FIM DA LÓGICA DE SCORE (INTACTA) ---

        if (initial_candidates.empty()) 
        {
            has_best_ = false; return;
        }

        // Ordenar pelo score
        std::sort(initial_candidates.begin(), initial_candidates.end(), 
            [](const ScoredGrasp& a, const ScoredGrasp& b) { return a.total_score > b.total_score; });

        
        RCLCPP_INFO(this->get_logger(), "Fase 2: Seleção e Verificação (Placeholder)...");
        
        best_grasps_.clear();
        
        // Loop pelos top N candidatos
        for (const auto& sg : initial_candidates) 
        {
            if (best_grasps_.size() >= (size_t)num_to_publish) break;

            // Chama a funcao de colisao (que por enquanto nao faz nada)
            if (check_collision(sg))
            {
                best_grasps_.push_back(sg);
            }
        }
        
        has_best_ = !best_grasps_.empty();
        
        RCLCPP_INFO(this->get_logger(), "Encontrados %lu grasps (sem checar colisão real).", best_grasps_.size());
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
        for(size_t i=0; i<hit_lim; ++i) 
        {
            visualization_msgs::msg::Marker k; 
            k.header.frame_id="world"; k.header.stamp=t; k.ns="rays_hit"; k.id=i; k.type=0; k.action=0; 
            k.scale.x=ray_len * 0.5; k.scale.y=0.002; k.scale.z=0.002; 
            k.color.r=0.0; k.color.g=1.0; k.color.b=1.0; k.color.a=0.5; 
            k.pose = hit_candidates_[i];
            ma_rays.markers.push_back(k);
        }
        size_t all_lim = std::min((size_t)100, all_candidates_.size());
        for(size_t i=0; i<all_lim; ++i) 
        {
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

        std::string mesh_path = this->get_parameter("gripper_mesh_path").as_string();
        float mesh_scale = this->get_parameter("gripper_mesh_scale").as_double();

        // -------------------------------------------------------------
        // LEITURA DOS PARAMETROS DE OFFSET DINAMICOS (EXECUTA A CADA FRAME)
        // -------------------------------------------------------------
        float off_x = this->get_parameter("mesh_offset_x").as_double();
        float off_y = this->get_parameter("mesh_offset_y").as_double();
        float off_z = this->get_parameter("mesh_offset_z").as_double();
        
        float rot_roll = this->get_parameter("mesh_rot_roll").as_double();
        float rot_pitch = this->get_parameter("mesh_rot_pitch").as_double();
        float rot_yaw = this->get_parameter("mesh_rot_yaw").as_double();

        // VERIFICAÇÃO DE SEGURANÇA: Adiciona o protocolo file://
        if (mesh_path.find("package://") == std::string::npos && 
            mesh_path.find("file://") == std::string::npos) 
        {
            mesh_path = "file://" + mesh_path;
        }

        auto sphere = [&](int id, auto p, float r, float g, float b, float alpha) 
        {
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

            // Marcadores dos dedos (esferas)
            ma.markers.push_back(sphere(base_id + 0, grasp.pose_finger1, r, g, b, alpha)); 
            ma.markers.push_back(sphere(base_id + 1, grasp.pose_finger2, r, g, b, alpha)); 
            
            // Linha entre os dedos
            visualization_msgs::msg::Marker l; 
            l.header.frame_id="world"; l.header.stamp=t; l.ns="lines"; l.id=base_id+2; l.type=5; l.action=0; 
            l.scale.x=0.002; l.color.r=r; l.color.g=g; l.color.b=b; l.color.a=alpha;
            l.points.push_back(grasp.pose_finger1.position); 
            l.points.push_back(grasp.pose_finger2.position);
            ma.markers.push_back(l);

            // MESH DO GRIPPER (NOVO)
            visualization_msgs::msg::Marker mesh_marker;
            mesh_marker.header.frame_id = "world";
            mesh_marker.header.stamp = t;
            mesh_marker.ns = "gripper_mesh";
            mesh_marker.id = base_id + 5;
            mesh_marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            mesh_marker.action = visualization_msgs::msg::Marker::ADD;

            // --- INICIO DO AJUSTE DE POSE (OFFSET DINAMICO) ---
            
            // 1. Converter a pose original do Grasp para Eigen
            Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
            Eigen::Quaternionf grasp_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);
            Eigen::Affine3f tf_grasp = Eigen::Translation3f(grasp_pos) * grasp_rot;

            // 2. Criar a transformação de Offset usando os Parametros
            Eigen::Affine3f tf_offset = Eigen::Affine3f::Identity();
            tf_offset.translate(Eigen::Vector3f(off_x, off_y, off_z));

            // Rotação Euler (X, Y, Z ou qualquer ordem preferida - aqui usamos independente)
            // Combinando as rotações locais
            Eigen::Matrix3f rotation_matrix;
            rotation_matrix = Eigen::AngleAxisf(rot_roll, Eigen::Vector3f::UnitX())
                            * Eigen::AngleAxisf(rot_pitch, Eigen::Vector3f::UnitY())
                            * Eigen::AngleAxisf(rot_yaw, Eigen::Vector3f::UnitZ());
            
            tf_offset.rotate(rotation_matrix);

            // 3. Aplicar o offset
            Eigen::Affine3f tf_final = tf_grasp * tf_offset;

            // 4. Converter de volta para ROS Msg
            Eigen::Vector3f final_pos = tf_final.translation();
            Eigen::Quaternionf final_rot(tf_final.rotation());

            mesh_marker.pose.position.x = final_pos.x();
            mesh_marker.pose.position.y = final_pos.y();
            mesh_marker.pose.position.z = final_pos.z();
            mesh_marker.pose.orientation.x = final_rot.x();
            mesh_marker.pose.orientation.y = final_rot.y();
            mesh_marker.pose.orientation.z = final_rot.z();
            mesh_marker.pose.orientation.w = final_rot.w();
            
            // --- FIM DO AJUSTE DE POSE ---

            mesh_marker.scale.x = mesh_scale;
            mesh_marker.scale.y = mesh_scale;
            mesh_marker.scale.z = mesh_scale;
            
            mesh_marker.color.r = r; 
            mesh_marker.color.g = g; 
            mesh_marker.color.b = b; 
            mesh_marker.color.a = alpha;
            
            mesh_marker.mesh_resource = mesh_path;
            mesh_marker.mesh_use_embedded_materials = true;
            
            ma.markers.push_back(mesh_marker);

            // Normal marker (apenas para o top 1)
            if (i == 0) 
            {
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