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
#include <unordered_map>
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
#include <pcl/kdtree/kdtree_flann.h>
// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sys/resource.h>
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
    Eigen::Vector3f debug_entry_pt;
    Eigen::Vector3f debug_exit_pt;
    pcl::PointCloud<pcl::PointXYZ> debug_inliers; 
};

class BestGraspFinder : public rclcpp::Node
{
public:
    BestGraspFinder() : Node("best_grasp_finder") 
    {
        this->declare_parameter<bool>("subscribe_to_point_cloud", false);
        this->declare_parameter<std::string>("pcd_path", "/home/momesso/pibic/nuvem.pcd");
        
        this->declare_parameter<std::string>("gripper_mesh_path", "/home/momesso/hand_and_finger.stl");
        this->declare_parameter<double>("gripper_mesh_scale", 0.001);
        
        this->declare_parameter<double>("mesh_offset_x", 0.0);
        this->declare_parameter<double>("mesh_offset_y", 0.0);
        this->declare_parameter<double>("mesh_offset_z", -0.025);
        
        this->declare_parameter<double>("mesh_rot_roll", 0.0);
        this->declare_parameter<double>("mesh_rot_pitch", 0.0);
        this->declare_parameter<double>("mesh_rot_yaw", 1.57); 

        this->declare_parameter<double>("grid_res", 0.02);
        this->declare_parameter<double>("cloud_voxel_size", 0.003);
        
        this->declare_parameter<double>("cylinder_radius", 0.02); 
        this->declare_parameter<double>("cylinder_height", 0.015);
        this->declare_parameter<double>("analysis_step_size", 0.01);
        
        this->declare_parameter<double>("max_gripper_width", 0.12); 
        this->declare_parameter<double>("finger_offset", 0.03); 
        
        this->declare_parameter<int>("min_points_per_segment", 6);
        this->declare_parameter<double>("weight_orientation", 0.6); 
        this->declare_parameter<double>("weight_symmetry", 0.2);
        this->declare_parameter<double>("weight_planarity", 0.2);
        
        this->declare_parameter<bool>("use_mean_filter", true); 
        this->declare_parameter<int>("mean_filter_k", 10);

        this->declare_parameter<int>("num_best_grasps", 5);
        this->declare_parameter<double>("rotation_step_deg", 30.0);

        subscribe_to_point_cloud_ = this->get_parameter("subscribe_to_point_cloud").as_bool();
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
        rotation_step_deg_ = static_cast<float>(this->get_parameter("rotation_step_deg").as_double());

        pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("input_cloud", 10);
        pub_rays_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("candidate_rays", 10);
        pub_bbox_  = this->create_publisher<visualization_msgs::msg::Marker>("bounding_box", 10);
        pub_markers_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("best_grasps_markers", 10);
        pub_poses_ = this->create_publisher<geometry_msgs::msg::PoseArray>("best_grasps_poses", 10);
        pub_debug_inliers_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("debug_ray_inliers", 10);

        stored_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        
    
        RCLCPP_INFO(this->get_logger(), "MODO ARQUIVO: Carregando PCD de %s...", pcd_path_.c_str());
        loadAndProcess(pcd_path_);
        
        
        timer_ = this->create_wall_timer(1000ms, std::bind(&BestGraspFinder::timerCallback, this));
    }

private:
    // --- VARIÁVEIS DE CONFIGURAÇÃO (PARAMETROS) ---
    bool subscribe_to_point_cloud_;
    std::string pcd_path_;
    std::string gripper_mesh_path_;
    float gripper_mesh_scale_;

    float mesh_offset_x_;
    float mesh_offset_y_;
    float mesh_offset_z_;

    float mesh_rot_roll_;
    float mesh_rot_pitch_;
    float mesh_rot_yaw_;

    float grid_res_;
    float cloud_voxel_size_;

    float cylinder_radius_;
    float cylinder_height_;
    float analysis_step_size_;

    float max_gripper_width_;
    float finger_offset_;

    int min_points_per_segment_;
    float weight_orientation_;
    float weight_symmetry_;
    float weight_planarity_;

    bool mean_filter;
    float mean_filter_k_;

    int num_best_grasps_;
    float rotation_step_deg_;

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_rays_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_bbox_; 
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_debug_inliers_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_input_cloud_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud_;
    std::vector<geometry_msgs::msg::Pose> all_candidates_;
    std::vector<geometry_msgs::msg::Pose> hit_candidates_;
    std::vector<ScoredGrasp> best_grasps_;
    bool has_best_ = false;
    Eigen::Vector4f min_pt_, max_pt_;

    void inputCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if(subscribe_to_point_cloud_ == true)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *temp_cloud);
            processCloud(temp_cloud);
        }
        
    }

    void loadAndProcess(const std::string& path)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *temp_cloud) == -1) 
        {
            RCLCPP_ERROR(this->get_logger(), "Falha ao ler arquivo PCD: %s", path.c_str());
            return;
        }
        processCloud(temp_cloud);
    }

    void processCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud)
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (input_cloud->empty()) return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        if (cloud_voxel_size_ > 0.0001) 
        {
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            sor.setInputCloud(input_cloud);
            sor.setLeafSize(cloud_voxel_size_, cloud_voxel_size_, cloud_voxel_size_);
            sor.filter(*voxel_cloud);
        } 
        else 
        {
            *voxel_cloud = *input_cloud;
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

            
            #pragma omp parallel for
            for (size_t i = 0; i < voxel_cloud->points.size(); ++i) 
            {
                
                std::vector<int> pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);

                
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
        else 
        {
            *stored_cloud_ = *voxel_cloud;
        }
        
        stored_cloud_->header.frame_id = "world";

        pcl::getMinMax3D(*stored_cloud_, min_pt_, max_pt_);
        float padding = 0.02; min_pt_.array() -= padding; max_pt_.array() += padding;
        
        all_candidates_ = generateOrthogonalRays(min_pt_, max_pt_, grid_res_);

           auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        RCLCPP_INFO(this->get_logger(), "Tempo Total: %ld ms", duration);
      
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
            for(float y = min[1]; y < max[1]; y += res) {
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

    bool check_collision(const ScoredGrasp& grasp)
    {
        return true; 
    }

    Eigen::Quaternionf findBestOrientation(const Eigen::Vector3f& p_f1, const Eigen::Vector3f& p_f2)
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

    struct VoxelBucket {
    Eigen::Vector3f center;
    std::vector<pcl::PointXYZ> points;
};

    void evaluateGrasps()
    {
        std::vector<ScoredGrasp> initial_candidates; 
        hit_candidates_.clear(); 

        RCLCPP_INFO(this->get_logger(), "Fase 1: Avaliando %lu raios...", all_candidates_.size());

        auto start = std::chrono::high_resolution_clock::now();

        double total_time_inliers_ms = 0.0;
        double total_time_analysis_ms = 0.0;
        double total_time_scoring_ms  = 0.0;
        
    
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
        

        for (const auto& raw_pose : all_candidates_) 
        {
            Eigen::Quaternionf q(raw_pose.orientation.w, raw_pose.orientation.x, raw_pose.orientation.y, raw_pose.orientation.z);
            Eigen::Vector3f ray_origin(raw_pose.position.x, raw_pose.position.y, raw_pose.position.z);
            Eigen::Vector3f ray_dir = q * Eigen::Vector3f::UnitX(); 

            float t_min = 1e6, t_max = -1e6;
            bool hit = false;
            
            pcl::PointCloud<pcl::PointXYZ> current_inliers;
            pcl::PointCloud<pcl::PointXYZ>::Ptr current_inliers_ptr(new pcl::PointCloud<pcl::PointXYZ>);

            auto t_inliers_start = std::chrono::high_resolution_clock::now();

            
            for (const auto& [key, bucket] : voxel_grid) 
            {
            
                Eigen::Vector3f diff = bucket.center - ray_origin;
                float dist_sq_to_ray = (diff.cross(ray_dir)).squaredNorm();

                
                if (dist_sq_to_ray > (voxel_check_threshold * voxel_check_threshold)) {
                    continue;
                }

                
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
        
            
            auto t_inliers_end = std::chrono::high_resolution_clock::now();
            total_time_inliers_ms += std::chrono::duration<double, std::milli>(t_inliers_end - t_inliers_start).count();


            if (!hit || (t_max - t_min) < 0.005) continue;
            
            hit_candidates_.push_back(raw_pose);

            std::vector<StepAnalysis> steps;

            auto t_analysis_start = std::chrono::high_resolution_clock::now();

            for (float t = t_min; t <= t_max; t += analysis_step_size_) 
            {
                Eigen::Vector3f center = ray_origin + ray_dir * t;
                StepAnalysis res = analyzeLocalCylinder(current_inliers_ptr, center, ray_dir, cylinder_radius_, cylinder_height_);
                if (res.valid) steps.push_back(res);
            }

            auto t_analysis_end = std::chrono::high_resolution_clock::now();
            total_time_analysis_ms += std::chrono::duration<double, std::milli>(t_analysis_end - t_analysis_start).count();


            if (steps.empty()) continue;

            auto t_scoring_start = std::chrono::high_resolution_clock::now();

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
            
            initial_candidates.push_back(sg);

            auto t_scoring_end = std::chrono::high_resolution_clock::now();
            total_time_scoring_ms += std::chrono::duration<double, std::milli>(t_scoring_end - t_scoring_start).count();
        }

        if (initial_candidates.empty()) 
        {
            has_best_ = false; return;
        }

        std::sort(initial_candidates.begin(), initial_candidates.end(), 
            [](const ScoredGrasp& a, const ScoredGrasp& b) { return a.total_score > b.total_score; });

        best_grasps_.clear();
        for (const auto& sg : initial_candidates) 
        {
            if (best_grasps_.size() >= (size_t)num_best_grasps_) break;
            if (check_collision(sg))
            {
                best_grasps_.push_back(sg);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        has_best_ = !best_grasps_.empty();
        RCLCPP_INFO(this->get_logger(), "Encontrados %lu grasps.", best_grasps_.size());
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        RCLCPP_INFO(this->get_logger(), "Tempo Total: %ld ms", duration);
        RCLCPP_INFO(this->get_logger(), ">> Breakdown: Inliers: %.2f ms | Analysis: %.2f ms | Scoring: %.2f ms", 
            total_time_inliers_ms, total_time_analysis_ms, total_time_scoring_ms);
            
            
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        
        long max_mem_mb = usage.ru_maxrss / 1024; 
        
        RCLCPP_INFO(this->get_logger(), "Memória Máxima Usada (RSS): %ld MB", max_mem_mb);
        
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

        std::string mesh_path = gripper_mesh_path_;
        if (mesh_path.find("package://") == std::string::npos && 
            mesh_path.find("file://") == std::string::npos) 
        {
            mesh_path = "file://" + mesh_path;
        }

        auto sphere = [&](int id, auto p, float r, float g, float b, float alpha, float sc) 
        {
            visualization_msgs::msg::Marker m; m.header.frame_id="world"; m.header.stamp=t; 
            m.ns="fingers"; m.id=id; m.type=2; m.action=0; m.pose=p; 
            m.scale.x=sc; m.scale.y=sc; m.scale.z=sc; 
            m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=alpha; 
            return m;
        };

        if (!best_grasps_.empty()) {
            sensor_msgs::msg::PointCloud2 debug_msg;
            pcl::toROSMsg(best_grasps_[0].debug_inliers, debug_msg);
            debug_msg.header.frame_id = "world";
            debug_msg.header.stamp = t;
            pub_debug_inliers_->publish(debug_msg);
        }

        for(size_t i = 0; i < best_grasps_.size(); i++)
        {
            const auto& grasp = best_grasps_[i];
            pose_array_msg.poses.push_back(grasp.pose_center);

            float r=0, g=1, b=0, alpha=0.6;
            if (i == 0) { r=0; g=0; b=1; alpha=1.0; }
            int base_id = i * 30; 

            ma.markers.push_back(sphere(base_id + 0, grasp.pose_finger1, r, g, b, alpha, 0.025)); 
            ma.markers.push_back(sphere(base_id + 1, grasp.pose_finger2, r, g, b, alpha, 0.025)); 
            
            if (i == 0) {
                geometry_msgs::msg::Pose p_start; 
                p_start.position.x = grasp.debug_entry_pt.x(); p_start.position.y = grasp.debug_entry_pt.y(); p_start.position.z = grasp.debug_entry_pt.z();
                geometry_msgs::msg::Pose p_end; 
                p_end.position.x = grasp.debug_exit_pt.x(); p_end.position.y = grasp.debug_exit_pt.y(); p_end.position.z = grasp.debug_exit_pt.z();

                ma.markers.push_back(sphere(base_id + 10, p_start, 0.0, 1.0, 0.0, 1.0, 0.015));
                ma.markers.push_back(sphere(base_id + 11, p_end, 1.0, 0.0, 0.0, 1.0, 0.015));
            }

            visualization_msgs::msg::Marker mesh_marker;
            mesh_marker.header.frame_id = "world"; mesh_marker.header.stamp = t;
            mesh_marker.ns = "gripper_mesh"; mesh_marker.id = base_id + 5;
            mesh_marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
            mesh_marker.action = visualization_msgs::msg::Marker::ADD;

            Eigen::Vector3f grasp_pos(grasp.pose_center.position.x, grasp.pose_center.position.y, grasp.pose_center.position.z);
            Eigen::Quaternionf grasp_rot(grasp.pose_center.orientation.w, grasp.pose_center.orientation.x, grasp.pose_center.orientation.y, grasp.pose_center.orientation.z);
            Eigen::Affine3f tf_grasp = Eigen::Translation3f(grasp_pos) * grasp_rot;

            Eigen::Affine3f tf_offset = Eigen::Affine3f::Identity();
            tf_offset.translate(Eigen::Vector3f(mesh_offset_x_, mesh_offset_y_, mesh_offset_z_));
            Eigen::Matrix3f rotation_matrix;
            rotation_matrix = Eigen::AngleAxisf(mesh_rot_roll_, Eigen::Vector3f::UnitX())
                            * Eigen::AngleAxisf(mesh_rot_pitch_, Eigen::Vector3f::UnitY())
                            * Eigen::AngleAxisf(mesh_rot_yaw_, Eigen::Vector3f::UnitZ());
            tf_offset.rotate(rotation_matrix);

            Eigen::Affine3f tf_final = tf_grasp * tf_offset;
            Eigen::Vector3f final_pos = tf_final.translation();
            Eigen::Quaternionf final_rot(tf_final.rotation());

            mesh_marker.pose.position.x = final_pos.x(); mesh_marker.pose.position.y = final_pos.y(); mesh_marker.pose.position.z = final_pos.z();
            mesh_marker.pose.orientation.x = final_rot.x(); mesh_marker.pose.orientation.y = final_rot.y(); mesh_marker.pose.orientation.z = final_rot.z(); mesh_marker.pose.orientation.w = final_rot.w();
            mesh_marker.scale.x = gripper_mesh_scale_; mesh_marker.scale.y = gripper_mesh_scale_; mesh_marker.scale.z = gripper_mesh_scale_;
            mesh_marker.color.r = r; mesh_marker.color.g = g; mesh_marker.color.b = b; mesh_marker.color.a = alpha;
            mesh_marker.mesh_resource = mesh_path; mesh_marker.mesh_use_embedded_materials = true;
            ma.markers.push_back(mesh_marker);
            
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