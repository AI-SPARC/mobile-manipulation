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
    double total_score;
    double entry_angle;
    double exit_angle;
    double entry_planarity;
    double exit_planarity;
    Eigen::Vector3f entry_normal;
};

class BestGraspFinder : public rclcpp::Node
{
public:
    BestGraspFinder() : Node("best_grasp_finder") 
    {
        this->declare_parameter<std::string>("pcd_path", "/home/momesso/pibic/nuvem.pcd");
        
        
        this->declare_parameter<float>("grid_res", 0.02);
        this->declare_parameter<float>("cloud_voxel_size", 0.003);
        
        
        this->declare_parameter<float>("cylinder_radius", 0.015); 
        this->declare_parameter<float>("cylinder_height", 0.015);
        this->declare_parameter<float>("analysis_step_size", 0.01);
        this->declare_parameter<float>("finger_offset", 0.025);
        
       
        this->declare_parameter<int>("min_points_per_segment", 6);
        this->declare_parameter<float>("weight_orientation", 0.6); 
        this->declare_parameter<float>("weight_symmetry", 0.2);
        this->declare_parameter<float>("weight_planarity", 0.2);
        
       
        this->declare_parameter<bool>("use_mls_smoothing", false); 
        this->declare_parameter<float>("mls_radius", 0.03);

        this->declare_parameter<int>("num_best_grasps", 5);

        pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("input_cloud", 10);
        pub_rays_  = this->create_publisher<visualization_msgs::msg::MarkerArray>("candidate_rays", 10);
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
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_poses_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr stored_cloud_;
    std::vector<geometry_msgs::msg::Pose> all_candidates_;
    std::vector<ScoredGrasp> best_grasps_;
    bool has_best_ = false;
    Eigen::Vector4f min_pt_, max_pt_;

    void loadAndProcess(const std::string& path)
    {
        RCLCPP_INFO(this->get_logger(), "Lendo PCD...");
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *temp_cloud) == -1) return;

        
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

        
        if (this->get_parameter("use_mls_smoothing").as_bool()) {
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
            for (const auto& pt_n : mls_points) {
                pcl::PointXYZ pt; pt.x = pt_n.x; pt.y = pt_n.y; pt.z = pt_n.z;
                stored_cloud_->points.push_back(pt);
            }
        } else {
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

    void evaluateGrasps()
    {
        float radius = this->get_parameter("cylinder_radius").as_double();
        float finger_offset = this->get_parameter("finger_offset").as_double();
        float cyl_height = this->get_parameter("cylinder_height").as_double();
        float step_size = this->get_parameter("analysis_step_size").as_double();
        
        float w_orient = this->get_parameter("weight_orientation").as_double();
        float w_sym = this->get_parameter("weight_symmetry").as_double();
        float w_plan = this->get_parameter("weight_planarity").as_double();
        int num_to_publish = this->get_parameter("num_best_grasps").as_int();

        std::vector<ScoredGrasp> ranked_grasps;
        RCLCPP_INFO(this->get_logger(), "Avaliando %lu raios...", all_candidates_.size());

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

            std::vector<StepAnalysis> steps;
            for (float t = t_min; t <= t_max; t += step_size) {
                Eigen::Vector3f center = ray_origin + ray_dir * t;
                StepAnalysis res = analyzeLocalCylinder(stored_cloud_, center, ray_dir, radius, cyl_height);
                if (res.valid) steps.push_back(res);
            }

            if (steps.empty()) continue;

            StepAnalysis& entry = steps.front();
            StepAnalysis& exit = steps.back();

            Eigen::Vector3f p_f1 = entry.center - ray_dir * finger_offset;
            Eigen::Vector3f p_f2 = exit.center + ray_dir * finger_offset;
            Eigen::Vector3f p_c  = (p_f1 + p_f2) / 2.0f;

            
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

            if ((exit.center - entry.center).norm() < 0.015) total *= 0.1;

            ScoredGrasp sg;
            sg.pose_center = raw_pose; sg.pose_center.position.x = p_c.x(); sg.pose_center.position.y = p_c.y(); sg.pose_center.position.z = p_c.z();
            sg.pose_finger1 = raw_pose; sg.pose_finger1.position.x = p_f1.x(); sg.pose_finger1.position.y = p_f1.y(); sg.pose_finger1.position.z = p_f1.z();
            sg.pose_finger2 = raw_pose; sg.pose_finger2.position.x = p_f2.x(); sg.pose_finger2.position.y = p_f2.y(); sg.pose_finger2.position.z = p_f2.z();
            sg.total_score = total;
            sg.entry_angle = entry.angle_to_normal_deg;
            sg.exit_angle = exit.angle_to_normal_deg;
            sg.entry_planarity = 1.0 - entry.curvature;
            sg.exit_planarity = 1.0 - exit.curvature;
            sg.entry_normal = entry.normal_vector;
            
            ranked_grasps.push_back(sg);
        }

        if (ranked_grasps.empty()) {
            has_best_ = false; return;
        }

        std::sort(ranked_grasps.begin(), ranked_grasps.end(), 
            [](const ScoredGrasp& a, const ScoredGrasp& b) { return a.total_score > b.total_score; });

        best_grasps_.clear();
        int count = std::min((int)ranked_grasps.size(), num_to_publish);
        for(int i = 0; i < count; i++) best_grasps_.push_back(ranked_grasps[i]);
        
        has_best_ = true;
        for(int i = 0; i < num_to_publish; i++)
        {
            RCLCPP_INFO(this->get_logger(), "%d Score: %.2f (Planar: %.2f)", i, best_grasps_[i].total_score, best_grasps_[i].entry_planarity);
        }
        
    }

    void timerCallback() 
    {
        sensor_msgs::msg::PointCloud2 m; 
        pcl::toROSMsg(*stored_cloud_, m); 
        m.header.stamp=now(); m.header.frame_id="world"; 
        pub_cloud_->publish(m);
        
        visualization_msgs::msg::MarkerArray ma_rays; 
        size_t lim = std::min((size_t)200, all_candidates_.size());
        for(size_t i=0; i<lim; ++i) {
            visualization_msgs::msg::Marker k; k.header.frame_id="world"; k.header.stamp=now(); k.ns="rays"; k.id=i; k.type=5; k.action=0; k.scale.x=0.001; k.color.a=0.1; k.color.b=1.0; k.color.g=1.0;
            geometry_msgs::msg::Point p1=all_candidates_[i].position, p2; 
            Eigen::Quaternionf q(all_candidates_[i].orientation.w, all_candidates_[i].orientation.x, all_candidates_[i].orientation.y, all_candidates_[i].orientation.z);
            Eigen::Vector3f d=q*Eigen::Vector3f::UnitX(); p2.x=p1.x+d.x()*0.2; p2.y=p1.y+d.y()*0.2; p2.z=p1.z+d.z()*0.2;
            k.points.push_back(p1); k.points.push_back(p2); ma_rays.markers.push_back(k);
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
            int base_id = i * 10;

            ma.markers.push_back(sphere(base_id + 0, grasp.pose_finger1, r, g, b, alpha)); 
            ma.markers.push_back(sphere(base_id + 1, grasp.pose_finger2, r, g, b, alpha)); 
            
            visualization_msgs::msg::Marker l; 
            l.header.frame_id="world"; l.header.stamp=t; l.ns="lines"; l.id=base_id+2; l.type=5; l.action=0; 
            l.scale.x=0.005; l.color.r=r; l.color.g=g; l.color.b=b; l.color.a=alpha;
            l.points.push_back(grasp.pose_finger1.position); 
            l.points.push_back(grasp.pose_finger2.position);
            ma.markers.push_back(l);

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
            if (i==0) sprintf(buf, "TOP 1\nS:%.2f P:%.2f", grasp.total_score, grasp.entry_planarity);
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