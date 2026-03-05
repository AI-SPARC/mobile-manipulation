#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <cmath>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.hpp> 
#include <sensor_msgs/msg/camera_info.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h> 
#include <gtsam/nonlinear/ISAM2.h>

#include <Eigen/Dense>

#include "slam_interfaces/msg/gtsam_data.hpp"
#include "slam_feature_matching/FaissLoopDetector.hpp"
#include "slam_feature_matching/LightGlueMatcher.hpp"



class GtsamOptimizationNode : public rclcpp::Node
{
public:
    GtsamOptimizationNode() : Node("gtsam_optimization_node")
    {
        this->declare_parameter<std::string>("main_frame_id", "base_link");
        this->declare_parameter("num_robots", 1);
        this->declare_parameter("use_ground_truth", true);
        this->declare_parameter<std::string>("lightglue_path", 
            "/home/momesso/pibic/src/mobile_manipulation_packages/slam/slam_core/onxx/superpoint_lightglue_pipeline.onnx"
        );


        main_frame_id_ = this->get_parameter("main_frame_id").as_string();
        int num_robots = this->get_parameter("num_robots").as_int();
        use_ground_truth_ = this->get_parameter("use_ground_truth").as_bool();
        std::string lightglue_path = this->get_parameter("lightglue_path").as_string();
        
        RCLCPP_INFO(this->get_logger(), "Iniciando No GTSAM para %d robo(s)...", num_robots);

        if(use_ground_truth_) 
        {
            RCLCPP_INFO(this->get_logger(), "Comparacao com Ground Truth ATIVADA.");
        }

        RCLCPP_INFO(this->get_logger(), "LightGlue Path: %s", lightglue_path.c_str());
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        
        float threshold = 0.875f;
        faiss_loop_detector_ = std::make_shared<slam_feature_matching::FaissLoopDetector>(threshold);
        lightglue_matcher_ = std::make_shared<slam_feature_matching::LightGlueMatcher>(lightglue_path);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        robot_states_.reserve(num_robots);
        odom_pubs_.reserve(num_robots);
        graph_markers_pubs_.reserve(num_robots);
        path_pubs_.reserve(num_robots);
        
        factor_subs_.reserve(num_robots);
        image_subs_.reserve(num_robots);
        depth_subs_.reserve(num_robots); 
        syncs_.reserve(num_robots);
        gt_subs_.reserve(num_robots);
        cb_groups_.reserve(num_robots);

      // Parâmetros: Roll (x), Pitch (y), Yaw (z)
        gtsam::Rot3 rot = gtsam::Rot3::RzRyRx(-1.2407, 0.0000, -1.5278);
        gtsam::Point3 trans(0.1190, 0.0000, 0.3435);
        gtsam::Pose3 T_base_opt_(rot, trans);

        for (int i = 0; i < num_robots; ++i) 
        {
            robot_states_.push_back(std::make_unique<RobotSlamState>());

            auto cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            cb_groups_.push_back(cb_group);

            rclcpp::SubscriptionOptions sub_options;
            sub_options.callback_group = cb_group;

          
            std::string factor_topic = "/robot_" + std::to_string(i) + "/slam/camera_factors";
            std::string image_topic = "/robot_" + std::to_string(i) + "/loop_closure/dino_image";
            std::string depth_topic = "/robot_" + std::to_string(i) + "/loop_closure/depth_image"; 
            
           
            auto factor_sub = std::make_shared<message_filters::Subscriber<slam_interfaces::msg::GtsamData>>(
                this, factor_topic, rmw_qos_profile_default, sub_options);
                
            auto image_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                this, image_topic, rmw_qos_profile_default, sub_options);

            auto depth_sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                this, depth_topic, rmw_qos_profile_default, sub_options);

               
            auto sync = std::make_shared<Synchronizer>(ExactSyncPolicy(10), *factor_sub, *image_sub, *depth_sub);
            
            sync->registerCallback(
                std::bind(&GtsamOptimizationNode::sync_callback, this, i, 
                          std::placeholders::_1, std::placeholders::_2, std::placeholders::_3)
            );

            factor_subs_.push_back(factor_sub);
            image_subs_.push_back(image_sub);
            depth_subs_.push_back(depth_sub); 
            syncs_.push_back(sync);
            
            if (use_ground_truth_)
            {
                std::string gt_topic = "/robot_" + std::to_string(i) + "/ground_truth";
                auto gt_sub = this->create_subscription<nav_msgs::msg::Odometry>(
                    gt_topic, 10,
                    [this, i](const nav_msgs::msg::Odometry::SharedPtr msg) {
                        this->ground_truth_callback(i, msg);
                    },
                    sub_options
                );
                gt_subs_.push_back(gt_sub);
            }

         
            std::string odom_topic = "/robot_" + std::to_string(i) + "/odom";
            std::string marker_topic = "/robot_" + std::to_string(i) + "/gtsam_graph";
            std::string path_topic = "/robot_" + std::to_string(i) + "/gtsam_path";
            std::string camera_info_topic = "/robot_" + std::to_string(i) + "/camera_0/depth/camera_info"; 
            
            auto info_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                camera_info_topic, 10,
                [this, i](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                    this->camera_info_callback(i, msg);
                },
                sub_options 
            );
            camera_info_subs_.push_back(info_sub);

           
            odom_pubs_.push_back(this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10));
            graph_markers_pubs_.push_back(this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic, 10));
            path_pubs_.push_back(this->create_publisher<nav_msgs::msg::Path>(path_topic, 10));
        }

        
    }

private:

    struct FrameData 
    { 
        cv::Mat image;  
        cv::Mat depth_image; 
    };

    struct FrameProcessResult
    {
        gtsam::Pose3 delta_base;
        gtsam::SharedNoiseModel visual_noise;
        gtsam::Pose3 estimate;
        std::vector<float> signature;

        bool has_loop_closure = false;
        int loop_target_robot_id = -1;
        int loop_target_keyframe_id = -1;
        gtsam::Pose3 loop_transform;
        gtsam::SharedNoiseModel loop_noise;
    };

    struct RobotSlamState
    {
        int keyframe_id = 0;
        bool has_keyframe = false;

        bool has_gt = false;
        bool first_gt_received = false;
        gtsam::Pose3 initial_gt_pose;
        gtsam::Pose3 latest_gt_pose;
        gtsam::Pose3 previous_gt_pose;
        double total_gt_distance = 0.0;
        bool has_camera_info = false;
    };

    

    using ExactSyncPolicy = message_filters::sync_policies::ExactTime<
        slam_interfaces::msg::GtsamData, 
        sensor_msgs::msg::Image, 
        sensor_msgs::msg::Image>;
    using Synchronizer = message_filters::Synchronizer<ExactSyncPolicy>;

    std::shared_ptr<slam_feature_matching::FaissLoopDetector> faiss_loop_detector_;
    std::shared_ptr<slam_feature_matching::LightGlueMatcher> lightglue_matcher_;

    std::vector<std::unique_ptr<RobotSlamState>> robot_states_;

    std::vector<std::shared_ptr<message_filters::Subscriber<slam_interfaces::msg::GtsamData>>> factor_subs_;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>> image_subs_;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>> depth_subs_;
    std::vector<std::shared_ptr<Synchronizer>> syncs_;
    
  
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> gt_subs_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> camera_info_subs_;

    std::vector<rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr> odom_pubs_;
    std::vector<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr> graph_markers_pubs_;
    std::vector<rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr> path_pubs_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::mutex compute_mutex;
    std::string main_frame_id_;
    std::vector<cv::Mat> camera_matrix_;
    std::vector<cv::Mat> dist_coeffs_;
    gtsam::Pose3 T_base_opt_;

    std::vector<rclcpp::CallbackGroup::SharedPtr> cb_groups_;
    bool use_ground_truth_;
    bool tf_received = false;

    std::map<int, std::map<int, FrameData>> keyframe_database_;
    std::mutex global_gtsam_mutex_;
    gtsam::ISAM2 global_isam2_;
    gtsam::NonlinearFactorGraph global_graph_;
    gtsam::Values global_initial_estimates_;
    gtsam::Values global_optimized_estimates_;

    void publish_gtsam_data(int robot_id, const gtsam::Pose3& optimized_pose, const rclcpp::Time& stamp)
    {
        auto& state = *robot_states_[robot_id];
        char robot_prefix = 'a' + robot_id;
        
        std::string odom_frame = "robot_" + std::to_string(robot_id) + "/odom";
        std::string main_frame_id = "base_link";

        try 
        {
            if (!global_optimized_estimates_.exists(gtsam::Symbol(robot_prefix, state.keyframe_id))) return;
            
            gtsam::Matrix6 covariance_gtsam = global_isam2_.marginalCovariance(gtsam::Symbol(robot_prefix, state.keyframe_id));
            
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = stamp;
            odom_msg.header.frame_id = odom_frame;        
            odom_msg.child_frame_id = main_frame_id;    

            gtsam::Pose3 base_pose = optimized_pose;
            
            odom_msg.pose.pose.position.x = base_pose.x();
            odom_msg.pose.pose.position.y = base_pose.y();
            odom_msg.pose.pose.position.z = base_pose.z();
            
            Eigen::Quaterniond q(base_pose.rotation().matrix());
            odom_msg.pose.pose.orientation.x = q.x();
            odom_msg.pose.pose.orientation.y = q.y();
            odom_msg.pose.pose.orientation.z = q.z();
            odom_msg.pose.pose.orientation.w = q.w();
            
            geometry_msgs::msg::TransformStamped t;
            t.header.stamp = stamp;
            t.header.frame_id = odom_frame;
            t.child_frame_id = main_frame_id;
            t.transform.translation.x = base_pose.x();
            t.transform.translation.y = base_pose.y();
            t.transform.translation.z = base_pose.z();
            t.transform.rotation.x = q.x();
            t.transform.rotation.y = q.y();
            t.transform.rotation.z = q.z();
            t.transform.rotation.w = q.w();

            if (tf_broadcaster_) {
                tf_broadcaster_->sendTransform(t);
            }
           
            for (int i = 0; i < 3; ++i) 
            {
                for (int j = 0; j < 3; ++j) 
                {
                    odom_msg.pose.covariance[i * 6 + j] = covariance_gtsam(i + 3, j + 3);
                    odom_msg.pose.covariance[(i + 3) * 6 + (j + 3)] = covariance_gtsam(i, j);
                    odom_msg.pose.covariance[i * 6 + (j + 3)] = covariance_gtsam(i + 3, j);
                    odom_msg.pose.covariance[(i + 3) * 6 + j] = covariance_gtsam(i, j + 3);
                }
            }

            odom_pubs_[robot_id]->publish(odom_msg);

            visualization_msgs::msg::MarkerArray marker_array;
            nav_msgs::msg::Path path_msg;
            path_msg.header.stamp = stamp;
            path_msg.header.frame_id = odom_frame; 

            visualization_msgs::msg::Marker nodes_marker;
            nodes_marker.header.frame_id = odom_frame; 
            nodes_marker.header.stamp = stamp;
            nodes_marker.ns = "gtsam_nodes";
            nodes_marker.id = 0;
            nodes_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            nodes_marker.action = visualization_msgs::msg::Marker::ADD;
            nodes_marker.scale.x = 0.05;
            nodes_marker.scale.y = 0.05;
            nodes_marker.scale.z = 0.05;
            nodes_marker.color.a = 1.0;
            nodes_marker.color.r = 0.0;
            nodes_marker.color.g = 1.0;
            nodes_marker.color.b = 0.0;

            visualization_msgs::msg::Marker edges_marker;
            edges_marker.header.frame_id = odom_frame; 
            edges_marker.header.stamp = stamp;
            edges_marker.ns = "gtsam_edges";
            edges_marker.id = 1;
            edges_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            edges_marker.action = visualization_msgs::msg::Marker::ADD;
            edges_marker.scale.x = 0.02;
            edges_marker.color.a = 1.0;
            edges_marker.color.r = 1.0;
            edges_marker.color.g = 0.0;
            edges_marker.color.b = 0.0;

            for (const auto& key_value : global_optimized_estimates_) 
            {
                gtsam::Symbol sym(key_value.key);
                
                if (sym.chr() != robot_prefix) continue;

                gtsam::Pose3 node_base_pose = key_value.value.cast<gtsam::Pose3>();

                geometry_msgs::msg::Point p;
                p.x = node_base_pose.x();
                p.y = node_base_pose.y();
                p.z = node_base_pose.z();
                nodes_marker.points.push_back(p);

                geometry_msgs::msg::PoseStamped path_pose;
                path_pose.header.frame_id = odom_frame; 
                path_pose.pose.position = p;
                path_msg.poses.push_back(path_pose);
            }

            const gtsam::NonlinearFactorGraph& isam_graph = global_isam2_.getFactorsUnsafe();

            for (size_t i = 0; i < isam_graph.size(); ++i) 
            {
                auto factor = isam_graph.at(i);
                auto between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
                
                if (between_factor) 
                {
                    gtsam::Key key1 = between_factor->front();
                    gtsam::Key key2 = between_factor->back();
                    
                    gtsam::Symbol sym1(key1);
                    gtsam::Symbol sym2(key2);

                    if (sym1.chr() == robot_prefix || sym2.chr() == robot_prefix)
                    {
                        if (global_optimized_estimates_.exists(key1) && global_optimized_estimates_.exists(key2)) 
                        {
                            gtsam::Pose3 pose1_base = global_optimized_estimates_.at<gtsam::Pose3>(key1);
                            gtsam::Pose3 pose2_base = global_optimized_estimates_.at<gtsam::Pose3>(key2);

                            geometry_msgs::msg::Point p1, p2;
                            p1.x = pose1_base.x(); p1.y = pose1_base.y(); p1.z = pose1_base.z();
                            p2.x = pose2_base.x(); p2.y = pose2_base.y(); p2.z = pose2_base.z();
                            
                            edges_marker.points.push_back(p1);
                            edges_marker.points.push_back(p2);
                        }
                    }
                }
            }

            marker_array.markers.push_back(nodes_marker);
            marker_array.markers.push_back(edges_marker);

            graph_markers_pubs_[robot_id]->publish(marker_array);
            path_pubs_[robot_id]->publish(path_msg);

            RCLCPP_INFO(this->get_logger(), "[Robo %d] --- RELATORIO GTSAM GLOBAL ---", robot_id);
            RCLCPP_INFO(this->get_logger(), "[Grafo Global] Nos Totais: %d", (int)global_optimized_estimates_.size());
            RCLCPP_INFO(this->get_logger(), "[Grafo Global] Arestas Totais: %d", (int)global_isam2_.getFactorsUnsafe().size());
            RCLCPP_INFO(this->get_logger(), "[Robo %d] Pose odom->%s [X: %.3f | Y: %.3f | Z: %.3f]", robot_id, main_frame_id.c_str(), base_pose.x(), base_pose.y(), base_pose.z());
            
            
            if (use_ground_truth_ && state.has_gt) 
            {
                gtsam::Pose3 relative_gt = state.initial_gt_pose.inverse() * state.latest_gt_pose;
                
                double trans_error = (base_pose.translation() - relative_gt.translation()).norm();
                double trans_error_pct = (state.total_gt_distance > 0.001) ? (trans_error / state.total_gt_distance) * 100.0 : 0.0;

                gtsam::Rot3 rot_diff = base_pose.rotation().between(relative_gt.rotation());
                double rot_error_rad = gtsam::Rot3::Logmap(rot_diff).norm();
                double rot_error_deg = rot_error_rad * (180.0 / M_PI);
                
                RCLCPP_INFO(this->get_logger(), "[Robo %d] --- COMPARACAO GROUND TRUTH ---", robot_id);
                RCLCPP_INFO(this->get_logger(), "[Robo %d] Erro Translacao: %.4f m (%.2f%%) | Erro Rotacao: %.2f°", 
                            robot_id, trans_error, trans_error_pct, rot_error_deg);
            }

            RCLCPP_INFO(this->get_logger(), "-----------------------");
        } 
        catch (const gtsam::IndeterminantLinearSystemException& e) {
            RCLCPP_WARN(this->get_logger(), "[Robo %d] GTSAM IndeterminantLinearSystemException: Grafo instavel no momento.", robot_id);
        }
        catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "[Robo %d] Erro na publicacao dos dados do GTSAM: %s", robot_id, e.what());
        }
    }

   
    void ground_truth_callback(int robot_id, const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        auto& state = *robot_states_[robot_id];
        
       
        std::lock_guard<std::mutex> lock(global_gtsam_mutex_);

        Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        gtsam::Point3 t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        
        gtsam::Pose3 current_gt_pose = gtsam::Pose3(gtsam::Rot3(q), t);

        if (!state.first_gt_received) 
        {
            state.initial_gt_pose = current_gt_pose;
            state.previous_gt_pose = current_gt_pose; 
            state.total_gt_distance = 0.0;            
            state.first_gt_received = true;
        }
        else
        {
            double step_distance = (current_gt_pose.translation() - state.previous_gt_pose.translation()).norm();
            state.total_gt_distance += step_distance;
        }

        state.latest_gt_pose = current_gt_pose;
        state.previous_gt_pose = current_gt_pose; 
        state.has_gt = true;
    }

    void camera_info_callback(int robot_id, const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        
        if (robot_id < 0 || robot_id >= (int)robot_states_.size()) return;

        

        auto& state = *robot_states_[robot_id];
        
        if (state.has_camera_info) return;

        std::lock_guard<std::mutex> lock(global_gtsam_mutex_);

        
        if (robot_id >= (int)camera_matrix_.size()) {
            camera_matrix_.resize(robot_id + 1);
        }
        if (robot_id >= (int)dist_coeffs_.size()) {
            dist_coeffs_.resize(robot_id + 1);
        }

        camera_matrix_[robot_id] = cv::Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 9; ++i) 
        {
            camera_matrix_[robot_id].at<double>(i / 3, i % 3) = msg->k[i];
        }

        if (!msg->d.empty()) 
        {
            dist_coeffs_[robot_id] = cv::Mat::zeros(msg->d.size(), 1, CV_64F);

            for (size_t i = 0; i < msg->d.size(); ++i) 
            {
                dist_coeffs_[robot_id].at<double>(i) = msg->d[i];
            }
        } 
        else 
        {
            dist_coeffs_[robot_id] = cv::Mat::zeros(4, 1, CV_64F);
        }

        state.has_camera_info = true;
        
        RCLCPP_INFO(this->get_logger(), 
            "[Robo %d] CameraInfo recebido! fx: %.1f, fy: %.1f. Pronto para Loop Closure.", 
            robot_id, msg->k[0], msg->k[4]);
    }
    void sync_callback(
        int robot_id, 
        const slam_interfaces::msg::GtsamData::ConstSharedPtr& factor_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) 
    {

        if (!robot_states_[robot_id]->has_camera_info) 
        {
            RCLCPP_WARN_ONCE(this->get_logger(), "[Robo %d] Imagens ignoradas: aguardando CameraInfo...", robot_id);
            return; 
        }
     
        if(!tf_received)
        {
            try {
                geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                    main_frame_id_, 
                    image_msg->header.frame_id, 
                    tf2::TimePointZero,
                    std::chrono::milliseconds(100));
                    
              
                Eigen::Quaterniond q(transform_stamped.transform.rotation.w, transform_stamped.transform.rotation.x,
                                    transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z);
                Eigen::Vector3d t(transform_stamped.transform.translation.x, transform_stamped.transform.translation.y,
                                transform_stamped.transform.translation.z);
                
                T_base_opt_ = gtsam::Pose3(gtsam::Rot3(q.toRotationMatrix()), gtsam::Point3(t));
                tf_received = true;
            } 
            catch (const tf2::TransformException & ex) 
            { 
                RCLCPP_WARN(this->get_logger(), 
                    "\n"
                   
                    "Odometry can have serious errors and severe deviations.!\n"
                    "Error: %s", ex.what());
                
                return; 
            }
        }

       

      
        int current_keyframe = factor_msg->keyframe;

        cv::Mat converted_image, converted_depth; 
        
        try 
        {
            converted_image = cv_bridge::toCvCopy(image_msg, "bgr8")->image;
            converted_depth = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding)->image; 
        } 
        catch (const cv_bridge::Exception& e) 
        {
            RCLCPP_ERROR(this->get_logger(), "Erro cv_bridge: %s", e.what());
            return; 
        }

        {
            std::lock_guard<std::mutex> lock(global_gtsam_mutex_);
            keyframe_database_[robot_id][current_keyframe].image = converted_image;
            keyframe_database_[robot_id][current_keyframe].depth_image = converted_depth; 
        }

        FrameProcessResult result;

        gtsam::Rot3 delta_rot = gtsam::Rot3::Quaternion(
            factor_msg->delta_base.pose.orientation.w, factor_msg->delta_base.pose.orientation.x,
            factor_msg->delta_base.pose.orientation.y, factor_msg->delta_base.pose.orientation.z);
        
        gtsam::Point3 delta_trans(
            factor_msg->delta_base.pose.position.x, factor_msg->delta_base.pose.position.y, factor_msg->delta_base.pose.position.z);
        
        result.delta_base = gtsam::Pose3(delta_rot, delta_trans);

        Eigen::MatrixXd cov_eigen = Eigen::MatrixXd::Zero(6, 6);
        for (int i = 0; i < 6; ++i) 
        {
            for (int j = 0; j < 6; ++j) 
            {
                cov_eigen(i, j) = factor_msg->delta_base.covariance[i * 6 + j];
            }
        }
        result.visual_noise = gtsam::noiseModel::Gaussian::Covariance(cov_eigen);

        gtsam::Rot3 est_rot = gtsam::Rot3::Quaternion(
            factor_msg->estimate.orientation.w, factor_msg->estimate.orientation.x,
            factor_msg->estimate.orientation.y, factor_msg->estimate.orientation.z);
        
        gtsam::Point3 est_trans(
            factor_msg->estimate.position.x, factor_msg->estimate.position.y, factor_msg->estimate.position.z);
        
        result.estimate = gtsam::Pose3(est_rot, est_trans);
        result.signature = factor_msg->signature; 
        
      
        std::pair<int, int> loop_match = faiss_loop_detector_->process_feature_and_find_loop(
            robot_id, current_keyframe, factor_msg->signature);
        
        if (std::get<1>(loop_match) != -1) 
        {
            int best_loop_robot_id = std::get<0>(loop_match);
            int best_loop_kf_id = std::get<1>(loop_match);
            calculate_loop_closure(best_loop_robot_id, best_loop_kf_id, robot_id, current_keyframe, result);
        }

            
        process_gtsam(robot_id, result);
    }





    void process_gtsam(int robot_id, const FrameProcessResult& result)
    {
        auto& state = *robot_states_[robot_id]; 
        
        std::lock_guard<std::mutex> lock(global_gtsam_mutex_);
        
        char robot_prefix = 'a' + robot_id; 
        
        if (!state.has_keyframe) 
        {
            state.has_keyframe = true;
            state.keyframe_id = 0;
            
            global_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
                gtsam::Symbol(robot_prefix, 0), gtsam::Pose3(), gtsam::noiseModel::Isotropic::Variance(6, 1e-6)));
            global_initial_estimates_.insert(gtsam::Symbol(robot_prefix, 0), gtsam::Pose3());
            return; 
        }

        state.keyframe_id++;
        int kf_id = state.keyframe_id;

        global_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
            gtsam::Symbol(robot_prefix, kf_id - 1), 
            gtsam::Symbol(robot_prefix, kf_id), 
            result.delta_base, 
            result.visual_noise
        ));

        if (!global_initial_estimates_.exists(gtsam::Symbol(robot_prefix, kf_id))) 
        {
            global_initial_estimates_.insert(gtsam::Symbol(robot_prefix, kf_id), result.estimate);
        }

        if (result.has_loop_closure)
        {
            char from_robot_prefix = 'a' + result.loop_target_robot_id;

            global_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::Symbol(from_robot_prefix, result.loop_target_keyframe_id), 
                gtsam::Symbol(robot_prefix, kf_id), 
                result.loop_transform, 
                result.loop_noise 
            ));
            
            RCLCPP_INFO(this->get_logger(), "[Robo %d] Loop Closure com Robo %d (KF %d -> KF %d) adicionado e enviado ao ISAM2!", 
                        robot_id, result.loop_target_robot_id, result.loop_target_keyframe_id, kf_id);
        }

        // Otimização
        global_isam2_.update(global_graph_, global_initial_estimates_);
        global_optimized_estimates_ = global_isam2_.calculateEstimate();
        
        global_graph_.resize(0);
        global_initial_estimates_.clear();

        if (global_optimized_estimates_.exists(gtsam::Symbol(robot_prefix, state.keyframe_id))) 
        {
            gtsam::Pose3 corrected_pose = global_optimized_estimates_.at<gtsam::Pose3>(gtsam::Symbol(robot_prefix, state.keyframe_id));
            
            publish_gtsam_data(robot_id, corrected_pose, this->now());
        }
    }


    float get_robust_depth(const cv::Mat& depth_img, float x_f, float y_f) 
    {
        int x = std::round(x_f);
        int y = std::round(y_f);

        if (x < 1 || x >= depth_img.cols - 1 || y < 1 || y >= depth_img.rows - 1) return -1.0f; 

        float min_depth = 9999.0f;
        float max_depth = 0.0f;
        float center_depth = 0.0f;
        int valid_pixels = 0;

        for (int dy = -1; dy <= 1; ++dy) 
        {
            for (int dx = -1; dx <= 1; ++dx) 
            {
                float d = 0.0f;
                if (depth_img.type() == CV_32FC1) {
                    d = depth_img.at<float>(y + dy, x + dx);
                } else if (depth_img.type() == CV_16UC1) {
                    d = depth_img.at<uint16_t>(y + dy, x + dx) * 0.001f;
                }

                if (dx == 0 && dy == 0) center_depth = d;

                if (d > 0.1f && d < 7.0f) 
                {
                    if (d < min_depth) min_depth = d;
                    if (d > max_depth) max_depth = d;
                    valid_pixels++;
                }
            }
        }

        if (valid_pixels < 6) return -1.0f; 
        // if ((max_depth - min_depth) > 0.05f) return -1.0f; 

        return center_depth;
    }

    bool solveWeightedKabsch(const std::vector<Eigen::Vector3d>& pts_source, 
                         const std::vector<Eigen::Vector3d>& pts_target, 
                         const std::vector<double>& weights, 
                         Eigen::Matrix4d& out_T) 
    {
        if (pts_source.empty() || pts_source.size() != pts_target.size() || pts_source.size() != weights.size()) 
        {
            return false;
        }

        double total_weight = 0.0;
        Eigen::Vector3d centroid_source = Eigen::Vector3d::Zero();
        Eigen::Vector3d centroid_target = Eigen::Vector3d::Zero();

        for (size_t i = 0; i < pts_source.size(); ++i) {
            total_weight += weights[i];
            centroid_source += weights[i] * pts_source[i];
            centroid_target += weights[i] * pts_target[i];
        }

        if (total_weight < 1e-6) return false;

        centroid_source /= total_weight;
        centroid_target /= total_weight;

        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < pts_source.size(); ++i) {
            Eigen::Vector3d p_source = pts_source[i] - centroid_source;
            Eigen::Vector3d p_target = pts_target[i] - centroid_target;
            H += weights[i] * p_source * p_target.transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        Eigen::Matrix3d R = V * U.transpose();
        
        if (R.determinant() < 0) 
        {
            V.col(2) *= -1.0;
            R = V * U.transpose();
        }

        
        Eigen::Vector3d t = centroid_target - R * centroid_source;
        
        
        out_T = Eigen::Matrix4d::Identity();
        out_T.block<3,3>(0,0) = R;
        out_T.block<3,1>(0,3) = t;

        return true;
    }



    void calculate_loop_closure(int best_loop_robot_id, int best_loop_kf_id, int current_robot_id, int current_keyframe_id, FrameProcessResult& result)
    {
         

        if (best_loop_kf_id == -1 || 
            keyframe_database_[best_loop_robot_id].count(best_loop_kf_id) == 0 ||
            keyframe_database_[current_robot_id].count(current_keyframe_id) == 0) 
        {
            return; 
        }

        FrameData candidate_kf = keyframe_database_[best_loop_robot_id][best_loop_kf_id]; 
        FrameData current_kf = keyframe_database_[current_robot_id][current_keyframe_id]; 

        std::vector<cv::Point2f> kp1, kp2;
        std::vector<cv::DMatch> matches;
        {
            std::lock_guard<std::mutex> lock(compute_mutex);
            lightglue_matcher_->compute_matches(current_kf.image, candidate_kf.image, kp1, kp2, matches);
        }

        if (matches.size() >= 15) 
        {
            std::vector<cv::Point2f> train_pts, query_pts;
            for (const auto& match : matches) 
            {
                if (match.trainIdx < 0 || match.trainIdx >= (int)kp2.size() || match.queryIdx < 0 || match.queryIdx >= (int)kp1.size()) continue;
                
                train_pts.push_back(kp2[match.trainIdx]); 
                query_pts.push_back(kp1[match.queryIdx]); 
            }

            std::vector<cv::Point2f> train_pts_undist, query_pts_undist;

            if (cv::norm(dist_coeffs_[current_robot_id]) > 0.0001) 
            {
                cv::undistortPoints(train_pts, train_pts_undist, camera_matrix_[current_robot_id], dist_coeffs_[current_robot_id], cv::noArray(), camera_matrix_[current_robot_id]);
                cv::undistortPoints(query_pts, query_pts_undist, camera_matrix_[current_robot_id], dist_coeffs_[current_robot_id], cv::noArray(), camera_matrix_[current_robot_id]);
            } 
            else 
            {
                train_pts_undist = train_pts;
                query_pts_undist = query_pts;
            }

            double fx = camera_matrix_[current_robot_id].at<double>(0, 0);
            double fy = camera_matrix_[current_robot_id].at<double>(1, 1);
            double cx = camera_matrix_[current_robot_id].at<double>(0, 2);
            double cy = camera_matrix_[current_robot_id].at<double>(1, 2);

            std::vector<cv::Point3f> object_pts_3d;
            std::vector<cv::Point2f> image_pts_2d;

            for (size_t i = 0; i < train_pts.size(); ++i) 
            {
                cv::Point2f pt2d_train = train_pts[i];
                float z_center = get_robust_depth(candidate_kf.depth_image, pt2d_train.x, pt2d_train.y);

                if (z_center <= 0.1f || z_center > 7.0) continue;

                float min_z = z_center, max_z = z_center;
                for (int dy = -1; dy <= 1; ++dy) 
                {
                    for (int dx = -1; dx <= 1; ++dx) 
                    {
                        if (dx == 0 && dy == 0) continue; 
                        float z_neighbor = get_robust_depth(candidate_kf.depth_image, pt2d_train.x + dx, pt2d_train.y + dy);
                        if (z_neighbor > 0.1f) 
                        {
                            min_z = std::min(min_z, z_neighbor);
                            max_z = std::max(max_z, z_neighbor);
                        }
                    }
                }

                float x_cand = (train_pts_undist[i].x - cx) * z_center / fx;
                float y_cand = (train_pts_undist[i].y - cy) * z_center / fy;
                
                object_pts_3d.push_back(cv::Point3f(x_cand, y_cand, z_center));
                image_pts_2d.push_back(query_pts_undist[i]);
            }

            if (object_pts_3d.size() >= 15) 
            {
                cv::Mat rvec_guess = cv::Mat::zeros(3, 1, CV_64F);
                cv::Mat tvec_guess = cv::Mat::zeros(3, 1, CV_64F);
                std::vector<int> loop_inliers;
                
                bool pnp_loop_success = cv::solvePnPRansac(
                    object_pts_3d, image_pts_2d, camera_matrix_[current_robot_id], cv::Mat::zeros(4, 1, CV_64F), 
                    rvec_guess, tvec_guess, false, 1000, 10.0f, 0.95, loop_inliers, cv::SOLVEPNP_SQPNP
                );
                
                if (pnp_loop_success && loop_inliers.size() >= 15 && !rvec_guess.empty() && !tvec_guess.empty()) 
                {
                    std::vector<Eigen::Vector3d> inlier_pts_kf, inlier_pts_curr;
                    std::vector<double> inlier_weights;

                    cv::Mat R_ransac;
                    cv::Rodrigues(rvec_guess, R_ransac);
                    Eigen::Matrix3d R_eigen;
                    Eigen::Vector3d t_eigen;
                    for(int r=0; r<3; r++) {
                        for(int c=0; c<3; c++) R_eigen(r,c) = R_ransac.at<double>(r,c);
                        t_eigen(r) = tvec_guess.at<double>(r);
                    }

                    for (int idx : loop_inliers) 
                    {
                        inlier_pts_kf.push_back(Eigen::Vector3d(object_pts_3d[idx].x, object_pts_3d[idx].y, object_pts_3d[idx].z));
                        
                        float z_est = (R_eigen.row(2) * inlier_pts_kf.back())(0) + t_eigen(2);
                        float x_curr = (image_pts_2d[idx].x - cx) * z_est / fx;
                        float y_curr = (image_pts_2d[idx].y - cy) * z_est / fy;
                        inlier_pts_curr.push_back(Eigen::Vector3d(x_curr, y_curr, z_est));

                        inlier_weights.push_back(1.0 / (object_pts_3d[idx].z * object_pts_3d[idx].z));
                    }
                    
                    Eigen::Matrix4d T_camera_world_refined;
                    solveWeightedKabsch(inlier_pts_kf, inlier_pts_curr, inlier_weights, T_camera_world_refined);

                    Eigen::Matrix4d T_loop_relative = T_camera_world_refined.inverse();

                    gtsam::Pose3 loop_pose_opt(T_loop_relative);
                    gtsam::Pose3 loop_pose_base = T_base_opt_ * loop_pose_opt * T_base_opt_.inverse();

                   

                    Eigen::Matrix3d R_opt = T_loop_relative.block<3,3>(0,0);
                    Eigen::Vector3d t_opt = T_loop_relative.block<3,1>(0,3);

                    double loop_trans_dist = t_opt.norm();
                    double loop_rot_dist = Eigen::AngleAxisd(R_opt).angle();

                    double mean_loop_depth = 0.0;
                    for (int idx : loop_inliers) mean_loop_depth += object_pts_3d[idx].z;
                    mean_loop_depth /= std::max((double)loop_inliers.size(), 1.0); 

                    double inlier_ratio = 100.0 / std::max((double)loop_inliers.size(), 25.0); 
                    double penalty_inliers = inlier_ratio * inlier_ratio;
                    double penalty_motion = 1.0 + (loop_trans_dist * 2.0) + (loop_rot_dist * 2.0);
                    double penalty_depth = std::max(1.0, mean_loop_depth * mean_loop_depth * 0.5);

                    double base_var_trans = 0.01; 
                    double base_var_rot   = 0.07; 

                    Eigen::MatrixXd cov_eigen_loop = Eigen::MatrixXd::Zero(6, 6);
                    cov_eigen_loop(0, 0) = base_var_rot * penalty_inliers * penalty_motion;      
                    cov_eigen_loop(1, 1) = base_var_rot * penalty_inliers * penalty_motion;      
                    cov_eigen_loop(2, 2) = base_var_rot * penalty_inliers * penalty_motion;      
                    cov_eigen_loop(3, 3) = base_var_trans * penalty_inliers * penalty_motion; 
                    cov_eigen_loop(4, 4) = base_var_trans * penalty_inliers * penalty_motion; 
                    cov_eigen_loop(5, 5) = base_var_trans * penalty_inliers * penalty_motion * penalty_depth;  

                    auto loop_noise = gtsam::noiseModel::Gaussian::Covariance(cov_eigen_loop);
                    auto robust_loop_noise = gtsam::noiseModel::Robust::Create(
                        gtsam::noiseModel::mEstimator::Huber::Create(1.2), loop_noise);

                    // --- ATUALIZAÇÃO DIRETA NO RESULT (Sem precisar de Mutex!) ---
                    result.has_loop_closure = true;
                    result.loop_target_robot_id = best_loop_robot_id;
                    result.loop_target_keyframe_id = best_loop_kf_id;
                    result.loop_transform = loop_pose_base;
                    result.loop_noise = robust_loop_noise;

                    RCLCPP_INFO(this->get_logger(), "!!! LOOP CLOSURE !!! Robô %d (KF %d) pareou com Robô %d (KF %d). Inliers: %d", 
                                current_robot_id, current_keyframe_id, best_loop_robot_id, best_loop_kf_id, (int)loop_inliers.size());
                } 
            } 
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<GtsamOptimizationNode>();
    int num_robots = node->get_parameter("num_robots").as_int();
    
    if (num_robots > 1) 
    {
        rclcpp::executors::MultiThreadedExecutor executor(
            rclcpp::ExecutorOptions(), 
            num_robots
        );
        executor.add_node(node);
        executor.spin();
    } 
    else 
    {
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(node);
        executor.spin();
    }
    
    rclcpp::shutdown();
    return 0;
}