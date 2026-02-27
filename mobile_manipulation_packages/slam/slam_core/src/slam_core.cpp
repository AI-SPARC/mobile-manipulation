#include <memory>
#include <vector>
#include <iostream>
#include <chrono> 
#include <map>
#include <cmath> 

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp" 
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/path.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp" 
#include <opencv2/core/eigen.hpp>

#include <slam_core/DinoLoopNode.hpp>
#include <slam_core/Mapping.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

struct FrameData 
{ 
    int id; 
    cv::Mat image; 
    cv::Mat depth_image;
};

class SlamCoreNode : public rclcpp::Node 
{
public:
    SlamCoreNode(
        std::shared_ptr<slam_core::DinoLoopNode> dino_loop_node_node,
        std::shared_ptr<slam_core::Mapping> mapping_node
    ) : Node("slam_core_node") , 
        dino_loop_node_node_(dino_loop_node_node),
        mapping_node_(mapping_node)
    {
        this->declare_parameter<std::string>("main_frame_id", "base_link");

        main_frame_id_ = this->get_parameter("main_frame_id").as_string();


        rgb_sub_.subscribe(this, "/camera/camera/color/image_raw", rmw_qos_profile_sensor_data);
        depth_sub_.subscribe(this, "/camera/camera/depth/image_rect_raw", rmw_qos_profile_sensor_data);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(3), rgb_sub_, depth_sub_);
        sync_->registerCallback(std::bind(&SlamCoreNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&SlamCoreNode::cmd_vel_callback, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera/color/camera_info", 10, std::bind(&SlamCoreNode::camera_info_callback, this, std::placeholders::_1));

        gt_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ground_truth", 10, std::bind(&SlamCoreNode::ground_truth_callback, this, std::placeholders::_1));

        graph_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/factor_graph", 1);

        current_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/flann/current_image", 10);
        odometry_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/flann/odometry_matches", 10);
        
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odom", 10);
        graph_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/slam/graph_markers", 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/slam/trajectory_path", 10);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        orb_ = cv::ORB::create(1000);
        local_matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);

        camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
        dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);

        global_pose_ = cv::Mat::eye(4, 4, CV_64F);

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        last_processed_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "--- NO DE ODOMETRIA VISUAL E GTSAM INICIADO ---");
   
    }

private:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimates_;
    gtsam::Values optimized_estimates_;
    gtsam::ISAM2 isam2_;
    int keyframe_id_ = 0;
    std::map<int, FrameData> keyframe_database_;
    cv::Mat last_keyframe_pose_;
    FrameData last_keyframe_;
    bool has_keyframe_ = false;
    int tracking_lost_counter_ = 0;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::string main_frame_id_;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_; 
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gt_sub_;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr current_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr odometry_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_markers_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    std::shared_ptr<slam_core::DinoLoopNode> dino_loop_node_node_;
    std::shared_ptr<slam_core::Mapping> mapping_node_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    bool tf_received_ = false;
    gtsam::Pose3 T_base_cam_;
    
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::BFMatcher> local_matcher_; 
    
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv_bridge::CvImagePtr last_depth_msg_; 
    bool camera_info_received_ = false; 
    gtsam::Pose3 T_base_opt_;
    cv::Mat global_pose_;

    gtsam::Pose3 initial_gt_pose_;
    gtsam::Pose3 latest_gt_pose_;
    bool has_gt_ = false;
    bool first_gt_received_ = false;

    std::vector<FrameData> history_frames_;
    
    int frame_count_ = 0;
    bool is_moving_ = false;
    rclcpp::Time last_processed_time_;
    double total_gt_distance_ = 0.0;
    gtsam::Pose3 previous_gt_pose_;

    std::unordered_map<int, std::shared_ptr<std::vector<cv::Point3f>>> dense_clouds_database_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    void publish_factor_graph(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& current_estimate)
    {
        if (graph_pub_->get_subscription_count() == 0 || current_estimate.empty()) return;

        visualization_msgs::msg::MarkerArray marker_array;

        // ==========================================
        // 1. MARCADOR PARA OS NÓS (Esferas)
        // ==========================================
        visualization_msgs::msg::Marker nodes_marker;
        nodes_marker.header.frame_id = "map";
        nodes_marker.header.stamp = this->now();
        nodes_marker.ns = "gtsam_nodes";
        nodes_marker.id = 0;
        nodes_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        nodes_marker.action = visualization_msgs::msg::Marker::ADD;
        nodes_marker.pose.orientation.w = 1.0;
        
        // Tamanho das esferas
        nodes_marker.scale.x = 0.1; 
        nodes_marker.scale.y = 0.1;
        nodes_marker.scale.z = 0.1;
        
        // Cor (Azul)
        nodes_marker.color.r = 0.0f;
        nodes_marker.color.g = 0.5f;
        nodes_marker.color.b = 1.0f;
        nodes_marker.color.a = 1.0f;

        // Preenche as coordenadas das esferas
        for (const auto& key_value : current_estimate) {
            auto pose = key_value.value.cast<gtsam::Pose3>();
            geometry_msgs::msg::Point p;
            p.x = pose.x();
            p.y = pose.y();
            p.z = pose.z();
            nodes_marker.points.push_back(p);
        }
        marker_array.markers.push_back(nodes_marker);

        // ==========================================
        // 2. MARCADOR PARA AS ARESTAS (Linhas)
        // ==========================================
        visualization_msgs::msg::Marker edges_marker;
        edges_marker.header.frame_id = "map";
        edges_marker.header.stamp = this->now();
        edges_marker.ns = "gtsam_edges";
        edges_marker.id = 1;
        edges_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        edges_marker.action = visualization_msgs::msg::Marker::ADD;
        edges_marker.pose.orientation.w = 1.0;
        
        // Espessura da linha
        edges_marker.scale.x = 0.02; 
        
        // Cor (Verde para odometria, Vermelho para Loop Closure - opcional)
        edges_marker.color.r = 0.0f;
        edges_marker.color.g = 1.0f;
        edges_marker.color.b = 0.0f;
        edges_marker.color.a = 0.8f;

        // Varre o grafo procurando as "molas" que conectam duas poses
        for (const auto& factor : graph) {
            auto between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
            if (between_factor) {
                gtsam::Key key1 = between_factor->key1();
                gtsam::Key key2 = between_factor->key2();

                // Só desenha se ambas as poses já existirem na estimativa atual
                if (current_estimate.exists(key1) && current_estimate.exists(key2)) {
                    gtsam::Pose3 pose1 = current_estimate.at<gtsam::Pose3>(key1);
                    gtsam::Pose3 pose2 = current_estimate.at<gtsam::Pose3>(key2);

                    geometry_msgs::msg::Point p1, p2;
                    p1.x = pose1.x(); p1.y = pose1.y(); p1.z = pose1.z();
                    p2.x = pose2.x(); p2.y = pose2.y(); p2.z = pose2.z();

                    edges_marker.points.push_back(p1);
                    edges_marker.points.push_back(p2);
                }
            }
        }
        marker_array.markers.push_back(edges_marker);

        // Publica tudo de uma vez
        graph_pub_->publish(marker_array);
    }




    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) 
    {
        is_moving_ = (std::abs(msg->linear.x) > 0.01 || std::abs(msg->angular.z) > 0.01);
    }

    void ground_truth_callback(const nav_msgs::msg::Odometry::SharedPtr msg) 
    {
        Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        gtsam::Point3 t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        
        gtsam::Pose3 current_gt_pose = gtsam::Pose3(gtsam::Rot3(q), t);

        if (!first_gt_received_) 
        {
            initial_gt_pose_ = current_gt_pose;
            previous_gt_pose_ = current_gt_pose; 
            total_gt_distance_ = 0.0;            
            first_gt_received_ = true;
        }
        else
        {
            double step_distance = (current_gt_pose.translation() - previous_gt_pose_.translation()).norm();
            
            total_gt_distance_ += step_distance;
        }

        latest_gt_pose_ = current_gt_pose;
        previous_gt_pose_ = current_gt_pose; 
        has_gt_ = true;
    }

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) 
    {
        if (camera_info_received_) 
        {
            return; 
        }

        double fx = msg->k[0];
        double cx = msg->k[2];
        double fy = msg->k[4];
        double cy = msg->k[5];

        camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

        if (!msg->d.empty()) 
        {
            dist_coeffs_ = cv::Mat(msg->d.size(), 1, CV_64F);
            for (size_t i = 0; i < msg->d.size(); ++i) 
            {
                dist_coeffs_.at<double>(i) = msg->d[i];
            }
        }
        mapping_node_->set_camera_info(msg, main_frame_id_, 1.0);
        camera_info_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Matriz da Camera Carregada! fx:%.1f, fy:%.1f, cx:%.1f, cy:%.1f", fx, fy, cx, cy);
    }

    float get_depth_meters(const cv::Mat& depth_img, int x, int y) 
    {
        if (x < 0 || x >= depth_img.cols || y < 0 || y >= depth_img.rows) 
        {
            return 0.0f;
        }

        if (depth_img.type() == CV_32FC1) 
        {
            return depth_img.at<float>(y, x); 
        } 
        else if (depth_img.type() == CV_16UC1) 
        {
            return depth_img.at<uint16_t>(y, x) * 0.001f; 
        }
        
        return 0.0f;
    }

    float get_robust_depth(const cv::Mat& depth_img, float x_f, float y_f) 
    {
        int x = std::round(x_f);
        int y = std::round(y_f);

        if (x < 1 || x >= depth_img.cols - 1 || y < 1 || y >= depth_img.rows - 1) 
        {
            return -1.0f; 
        }

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

                if (d > 0.1f && d < 100.0f) 
                {
                    if (d < min_depth) min_depth = d;
                    if (d > max_depth) max_depth = d;
                    valid_pixels++;
                }
            }
        }

        if (valid_pixels < 6) return -1.0f; 

        if ((max_depth - min_depth) > 0.05f) 
        {
            return -1.0f; 
        }

        return center_depth;
    }

    void publish_gtsam_data(const gtsam::Pose3& optimized_pose, const rclcpp::Time& stamp)
    {
        try 
        {
            if (!optimized_estimates_.exists(keyframe_id_ - 1)) {
                return;
            }
            gtsam::Matrix6 covariance_gtsam = isam2_.marginalCovariance(keyframe_id_ - 1);
            
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = stamp;
            odom_msg.header.frame_id = "odom";        
            odom_msg.child_frame_id = "base_link";    

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
            t.header.frame_id = "odom";
            t.child_frame_id = "base_link";

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

            odom_pub_->publish(odom_msg);

            visualization_msgs::msg::MarkerArray marker_array;
            nav_msgs::msg::Path path_msg;
            path_msg.header.stamp = stamp;
            path_msg.header.frame_id = "odom"; 

            visualization_msgs::msg::Marker nodes_marker;
            nodes_marker.header.frame_id = "odom"; 
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
            edges_marker.header.frame_id = "odom"; 
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

            for (const auto& key_value : optimized_estimates_) 
            {
                gtsam::Pose3 node_base_pose = key_value.value.cast<gtsam::Pose3>();

                geometry_msgs::msg::Point p;
                p.x = node_base_pose.x();
                p.y = node_base_pose.y();
                p.z = node_base_pose.z();
                nodes_marker.points.push_back(p);

                geometry_msgs::msg::PoseStamped path_pose;
                path_pose.header.frame_id = "odom"; 
                path_pose.pose.position = p;
                path_msg.poses.push_back(path_pose);
            }

            for (size_t i = 0; i < graph_.size(); ++i) 
            {
                auto factor = graph_.at(i);
                auto between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
                
                if (between_factor) 
                {
                    gtsam::Key key1 = between_factor->front();
                    gtsam::Key key2 = between_factor->back();

                    if (optimized_estimates_.exists(key1) && optimized_estimates_.exists(key2)) 
                    {
                        gtsam::Pose3 pose1_base = optimized_estimates_.at<gtsam::Pose3>(key1);
                        gtsam::Pose3 pose2_base = optimized_estimates_.at<gtsam::Pose3>(key2);

                        geometry_msgs::msg::Point p1, p2;
                        p1.x = pose1_base.x();
                        p1.y = pose1_base.y();
                        p1.z = pose1_base.z();
                        p2.x = pose2_base.x();
                        p2.y = pose2_base.y();
                        p2.z = pose2_base.z();
                        
                        edges_marker.points.push_back(p1);
                        edges_marker.points.push_back(p2);
                    }
                }
            }

            marker_array.markers.push_back(nodes_marker);
            marker_array.markers.push_back(edges_marker);

            graph_markers_pub_->publish(marker_array);
            path_pub_->publish(path_msg);

            RCLCPP_INFO(this->get_logger(), "--- RELATORIO GTSAM ---");
            RCLCPP_INFO(this->get_logger(), "Nos Totais no Grafo: %d", (int)optimized_estimates_.size());
            RCLCPP_INFO(this->get_logger(), "Arestas (Fatores) Totais: %d", (int)graph_.size());
            RCLCPP_INFO(this->get_logger(), "Pose base_link [X: %7.3f | Y: %7.3f | Z: %7.3f | | x_rot: %7.3f | y_rot: %7.3f | z_rot: %7.3f]", 
            base_pose.x(), base_pose.y(), base_pose.z(), odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z);
            RCLCPP_INFO(this->get_logger(), "Covariancia Marginal GTSAM (Trace): %f", covariance_gtsam.trace());
            RCLCPP_INFO(this->get_logger(), "-----------------------");
        } 
        catch (const gtsam::IndeterminantLinearSystemException& e) 
        {
            RCLCPP_WARN(this->get_logger(), "GTSAM IndeterminantLinearSystemException: Grafo instavel no momento.");
        }
        catch (const std::exception& e)
        {
            RCLCPP_WARN(this->get_logger(), "Erro na publicacao dos dados do GTSAM: %s", e.what());
        }
    }




    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg, const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) 
    {
        try {
            last_depth_msg_ = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
        } catch (cv_bridge::Exception& e) {
            return;
        }

        auto msg_copy = std::make_shared<sensor_msgs::msg::Image>(*rgb_msg);

        
        std::string camera_frame_id = rgb_msg->header.frame_id;

        if (!tf_received_) 
        {
            if (main_frame_id_ == camera_frame_id) 
            {
                
                T_base_opt_ = gtsam::Pose3();
                tf_received_ = true;
                RCLCPP_INFO(this->get_logger(), "Frame principal e o mesmo da camera [%s]. Transformacao Extrinseca = Identidade.", main_frame_id_.c_str());
            }
            else 
            {
                try {
                    
                    geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                        main_frame_id_, camera_frame_id, tf2::TimePointZero);

                    Eigen::Quaterniond q(transform_stamped.transform.rotation.w, transform_stamped.transform.rotation.x,
                                        transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z);
                    Eigen::Vector3d t(transform_stamped.transform.translation.x, transform_stamped.transform.translation.y,
                                    transform_stamped.transform.translation.z);

                    T_base_opt_ = gtsam::Pose3(gtsam::Rot3(q.toRotationMatrix()), gtsam::Point3(t));
                    tf_received_ = true;
                    RCLCPP_INFO(this->get_logger(), "Transformacao %s -> %s recebida e alinhada!", main_frame_id_.c_str(), camera_frame_id.c_str());
                }
                catch (tf2::TransformException &ex) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                                         "Aguardando TF de %s para %s...", main_frame_id_.c_str(), camera_frame_id.c_str());
                    return;
                }
            }
        }

        if (!camera_info_received_  || !last_depth_msg_) return;

        auto now = this->now();
        last_processed_time_ = now;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) { return; }

        double fx = camera_matrix_.at<double>(0, 0);
        double fy = camera_matrix_.at<double>(1, 1);
        double cx = camera_matrix_.at<double>(0, 2);
        double cy = camera_matrix_.at<double>(1, 2);

        FrameData current_frame;
        current_frame.id = frame_count_;
        current_frame.image = cv_ptr->image.clone();
        current_frame.depth_image = last_depth_msg_->image.clone();

        auto solveKabschRansac = [](const std::vector<Eigen::Vector3d>& src, const std::vector<Eigen::Vector3d>& dst, 
                                    int iterations, double threshold_sq, Eigen::Matrix4d& out_T, std::vector<int>& out_inliers) -> bool {
            if (src.size() < 3) return false;

            auto compute_kabsch = [](const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<int>& indices) -> Eigen::Matrix4d {
                int N = indices.size();
                Eigen::Vector3d centroid_P = Eigen::Vector3d::Zero();
                Eigen::Vector3d centroid_Q = Eigen::Vector3d::Zero();
                for (int idx : indices) {
                    centroid_P += P[idx];
                    centroid_Q += Q[idx];
                }
                centroid_P /= N;
                centroid_Q /= N;

                Eigen::MatrixXd P_centered(3, N);
                Eigen::MatrixXd Q_centered(3, N);
                for (int i = 0; i < N; ++i) {
                    P_centered.col(i) = P[indices[i]] - centroid_P;
                    Q_centered.col(i) = Q[indices[i]] - centroid_Q;
                }

                Eigen::Matrix3d H = P_centered * Q_centered.transpose();
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();

                if (R.determinant() < 0) {
                    Eigen::Matrix3d V = svd.matrixV();
                    V.col(2) *= -1;
                    R = V * svd.matrixU().transpose();
                }

                Eigen::Vector3d t = centroid_Q - R * centroid_P;
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                T.block<3,3>(0,0) = R;
                T.block<3,1>(0,3) = t;
                return T;
            };

            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<int> dist(0, src.size() - 1);

            int best_inlier_count = 0;
            out_T = Eigen::Matrix4d::Identity();

            for (int iter = 0; iter < iterations; ++iter) {
                int i1 = dist(rng), i2 = dist(rng), i3 = dist(rng);
                if (i1 == i2 || i1 == i3 || i2 == i3) continue;

                Eigen::Matrix4d T_est = compute_kabsch(src, dst, {i1, i2, i3});

                std::vector<int> current_inliers;
                for (size_t i = 0; i < src.size(); ++i) {
                    Eigen::Vector3d p_transformed = (T_est.block<3,3>(0,0) * src[i]) + T_est.block<3,1>(0,3);
                    if ((p_transformed - dst[i]).squaredNorm() < threshold_sq) {
                        current_inliers.push_back(i);
                    }
                }

                if (current_inliers.size() > best_inlier_count) {
                    best_inlier_count = current_inliers.size();
                    out_inliers = current_inliers;
                    out_T = T_est;
                }
            }

            if (best_inlier_count >= 3) {
                out_T = compute_kabsch(src, dst, out_inliers);
                return true;
            }
            return false;
        };

        if (!has_keyframe_) 
        {
            last_keyframe_ = current_frame;
            global_pose_ = cv::Mat::eye(4, 4, CV_64F);
            last_keyframe_pose_ = cv::Mat::eye(4, 4, CV_64F);
            gtsam::Pose3 initial_pose = gtsam::Pose3();
            auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
            
            graph_.add(gtsam::PriorFactor<gtsam::Pose3>(keyframe_id_, initial_pose, prior_noise));
            initial_estimates_.insert(keyframe_id_, initial_pose);
            keyframe_database_[keyframe_id_] = current_frame; 

            msg_copy->header.frame_id = std::to_string(keyframe_id_);
            dino_loop_node_node_->keyframe_callback(msg_copy);
            mapping_node_->add_keyframe_data(keyframe_id_, current_frame.image, last_depth_msg_->image);
            std::vector<std::pair<int, gtsam::Pose3>> optimized_poses_for_mapping;
            optimized_poses_for_mapping.push_back(std::make_pair(0, initial_pose));
        
            mapping_node_->update_global_map(optimized_poses_for_mapping);

            keyframe_id_++;
            has_keyframe_ = true;
            frame_count_++;

            return;
        }
        else if(tracking_lost_counter_ >= 8)
        {
            RCLCPP_WARN(this->get_logger(), "[SEQUESTRO] Robo perdido por muito tempo. Criando nova Ilha no GTSAM!");


            keyframe_id_ += 1000; 
            
          
            tracking_lost_counter_ = 0; 

            last_keyframe_ = current_frame;

            Eigen::Matrix4d global_pose_eigen;
            cv::cv2eigen(global_pose_, global_pose_eigen);
            gtsam::Pose3 current_global_pose(global_pose_eigen);

            auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 100.0, 100.0, 100.0, 100.0, 100.0, 100.0).finished());
            
            graph_.add(gtsam::PriorFactor<gtsam::Pose3>(keyframe_id_, current_global_pose, prior_noise));
            initial_estimates_.insert(keyframe_id_, current_global_pose);
            keyframe_database_[keyframe_id_] = current_frame; 

            msg_copy->header.frame_id = std::to_string(keyframe_id_);
            dino_loop_node_node_->keyframe_callback(msg_copy);
            mapping_node_->add_keyframe_data(keyframe_id_, current_frame.image, last_depth_msg_->image);

            keyframe_id_++;
            has_keyframe_ = true;
            frame_count_++;


            std::vector<std::pair<int, gtsam::Pose3>> optimized_poses_for_mapping;
            optimized_poses_for_mapping.push_back(std::make_pair(0, current_global_pose));
        
            mapping_node_->update_global_map(optimized_poses_for_mapping);

            
            return;
        }


        std::vector<cv::Point2f> kp1, kp2;
        std::vector<cv::DMatch> matches;

        dino_loop_node_node_->compute_matches(current_frame.image, last_keyframe_.image, kp1, kp2, matches);
        
        if (!matches.empty())
        {
            cv::Mat debug_image;
            cv::hconcat(current_frame.image, last_keyframe_.image, debug_image);

            for (const auto& match : matches) 
            {
                if (match.queryIdx >= 0 && match.queryIdx < (int)kp1.size() && 
                    match.trainIdx >= 0 && match.trainIdx < (int)kp2.size()) 
                {
                    cv::Point2f pt_current = kp1[match.queryIdx];
                    cv::Point2f pt_keyframe = kp2[match.trainIdx];
                    
                    pt_keyframe.x += current_frame.image.cols; 

                    cv::circle(debug_image, pt_current, 3, cv::Scalar(0, 255, 0), -1); 
                    cv::circle(debug_image, pt_keyframe, 3, cv::Scalar(0, 0, 255), -1); 
                }
            }

            std_msgs::msg::Header match_header;
            match_header.stamp = now;
            match_header.frame_id = camera_frame_id; 
            odometry_pub_->publish(*cv_bridge::CvImage(match_header, "bgr8", debug_image).toImageMsg());
        }
        
        bool tracking_success = false; 

        if (matches.size() >= 5) 
        {
            int edge_filter_rejected = 0;
            float min_pixel_dist = 999999.0f;
            float max_pixel_dist = 0.0f;
            float sum_pixel_dist = 0.0f;

            std::vector<cv::Point2f> train_pts, query_pts;
            for (const auto& match : matches) 
            {
                if (match.trainIdx < 0 || match.trainIdx >= (int)kp2.size() || 
                    match.queryIdx < 0 || match.queryIdx >= (int)kp1.size()) continue;
                
                train_pts.push_back(kp2[match.trainIdx]); 
                query_pts.push_back(kp1[match.queryIdx]); 
            }

            std::vector<cv::Point2f> train_pts_undist, query_pts_undist;
            if (cv::norm(dist_coeffs_) > 0.0001) {
                cv::undistortPoints(train_pts, train_pts_undist, camera_matrix_, dist_coeffs_, cv::noArray(), camera_matrix_);
                cv::undistortPoints(query_pts, query_pts_undist, camera_matrix_, dist_coeffs_, cv::noArray(), camera_matrix_);
            } else {
                train_pts_undist = train_pts;
                query_pts_undist = query_pts;
            }

            std::vector<Eigen::Vector3d> pts_kf, pts_curr;
            
            for (size_t i = 0; i < train_pts.size(); ++i) 
            {
                cv::Point2f pt2d_train = train_pts[i]; 
                cv::Point2f pt2d_query = query_pts[i]; 

                float dist_2d = cv::norm(pt2d_query - pt2d_train);
                sum_pixel_dist += dist_2d;
                if (dist_2d < min_pixel_dist) min_pixel_dist = dist_2d;
                if (dist_2d > max_pixel_dist) max_pixel_dist = dist_2d;

                float z_kf = get_robust_depth(last_keyframe_.depth_image, pt2d_train.x, pt2d_train.y);
                float z_curr = get_robust_depth(current_frame.depth_image, pt2d_query.x, pt2d_query.y);

                if (z_kf <= 0.0f || z_curr <= 0.0f) 
                {
                    edge_filter_rejected++; 
                    continue; 
                }

                float x_kf = (train_pts_undist[i].x - cx) * z_kf / fx;
                float y_kf = (train_pts_undist[i].y - cy) * z_kf / fy;
                
                float x_curr = (query_pts_undist[i].x - cx) * z_curr / fx;
                float y_curr = (query_pts_undist[i].y - cy) * z_curr / fy;
                
                pts_kf.push_back(Eigen::Vector3d(x_kf, y_kf, z_kf));
                pts_curr.push_back(Eigen::Vector3d(x_curr, y_curr, z_curr));
            }

            if (pts_kf.size() >= 8) 
            { 
                Eigen::Matrix4d T_curr_kf;
                std::vector<int> inliers;
                
                double threshold_sq = 0.01 * 0.01; 
                
                bool kabsch_success = solveKabschRansac(pts_kf, pts_curr, 1000, threshold_sq, T_curr_kf, inliers);
                
                if (kabsch_success && inliers.size() >= 8) 
                {
                    Eigen::Matrix3d R_kabsch = T_curr_kf.block<3,3>(0,0);
                    Eigen::Vector3d t_kabsch = T_curr_kf.block<3,1>(0,3);

                    double translation_dist = t_kabsch.norm();
                    Eigen::AngleAxisd aa(R_kabsch);
                    double rotation_dist = aa.angle(); 

                    Eigen::Matrix4d delta_opt_eigen = T_curr_kf.inverse();
                    gtsam::Pose3 delta_opt(delta_opt_eigen);

                    gtsam::Pose3 delta_base = T_base_opt_ * delta_opt * T_base_opt_.inverse();

                    gtsam::Point3 t_base = delta_base.translation();
                    gtsam::Rot3 r_base = delta_base.rotation();
                    double real_dist = t_base.norm();
                    
                    if (real_dist > 1.0) 
                    {
                        RCLCPP_WARN(this->get_logger(), "[REJEICAO] !!! Pulo absurdo detectado (Distancia > 1.0m) !!! Rastreamento rejeitado.");
                        tracking_success = false; 
                    }
                    else 
                    {
                        tracking_success = true; 
                        
                        Eigen::Matrix4d delta_base_eigen = delta_base.matrix();
                        cv::Mat delta_base_cv;
                        cv::eigen2cv(delta_base_eigen, delta_base_cv);
                        
                        global_pose_ = last_keyframe_pose_ * delta_base_cv;

                        double inlier_ratio = 100.0 / std::max((double)inliers.size(), 15.0);
                        double penalty_inliers = inlier_ratio * inlier_ratio; 
                        double penalty_motion = 1.0 + (translation_dist * 2.0) + (rotation_dist * 2.0);
                        
                        double mean_depth = 0.0;
                        for (int idx : inliers) mean_depth += pts_kf[idx].z();
                        mean_depth /= inliers.size();
                        double penalty_depth = std::max(1.0, mean_depth * mean_depth * 0.5);

                        double base_var_trans = 0.01; 
                        double base_var_rot   = 0.07; 

                        double var_x = base_var_trans * penalty_inliers * penalty_motion;
                        double var_y = base_var_trans * penalty_inliers * penalty_motion;
                        double var_z = base_var_trans * penalty_inliers * penalty_motion * penalty_depth; 
                        double var_roll  = base_var_rot * penalty_inliers * penalty_motion;
                        double var_pitch = base_var_rot * penalty_inliers * penalty_motion;
                        double var_yaw   = base_var_rot * penalty_inliers * penalty_motion;
                        
                        Eigen::MatrixXd cov_eigen = Eigen::MatrixXd::Zero(6, 6);
                        cov_eigen(0, 0) = var_roll;  
                        cov_eigen(1, 1) = var_pitch; 
                        cov_eigen(2, 2) = var_yaw;   
                        cov_eigen(3, 3) = var_x;     
                        cov_eigen(4, 4) = var_y;     
                        cov_eigen(5, 5) = var_z;     

                        Eigen::Matrix4d global_pose_eigen;
                        cv::cv2eigen(global_pose_, global_pose_eigen);
                        gtsam::Pose3 current_global_pose(global_pose_eigen);

                        // publish_gtsam_data(current_global_pose, msg_copy->header.stamp);
                        
                        if (translation_dist > 0.15 || rotation_dist > 0.1) 
                        {
                            RCLCPP_INFO(this->get_logger(), "[GTSAM] Condicao alcancada (Robo andou %.3f m). Criando NOVO KEYFRAME ID %d.", real_dist, keyframe_id_);
                            last_keyframe_ = current_frame;
                            
                            auto noise_model = gtsam::noiseModel::Gaussian::Covariance(cov_eigen);

                            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                                keyframe_id_ - 1, keyframe_id_, delta_base, noise_model));
                                
                            initial_estimates_.insert(keyframe_id_, current_global_pose);

                            msg_copy->header.frame_id = std::to_string(keyframe_id_);
                            
                            int loop_candidate_id = dino_loop_node_node_->keyframe_callback(msg_copy);
                            mapping_node_->add_keyframe_data(keyframe_id_, current_frame.image, last_depth_msg_->image);
                            
                            bool loop_detected = false;
                            Eigen::Matrix4d T_loop_relative = Eigen::Matrix4d::Identity(); 
                            int num_loop_inliers = 0;
                            std::vector<Eigen::Vector3d> loop_pts_cand, loop_pts_curr;
                            std::vector<int> loop_inliers; 

                            if (loop_candidate_id != -1 && keyframe_database_.count(loop_candidate_id)) 
                            {
                                if ((keyframe_id_ - loop_candidate_id) >= 20) 
                                {
                                    FrameData candidate_kf = keyframe_database_[loop_candidate_id];
                                    
                                    std::vector<cv::Point2f> loop_kp1, loop_kp2;
                                    std::vector<cv::DMatch> loop_matches;
                                    dino_loop_node_node_->compute_matches(current_frame.image, candidate_kf.image, loop_kp1, loop_kp2, loop_matches);

                                    if (loop_matches.size() >= 60) 
                                    {
                                        std::vector<cv::Point2f> loop_train_pts, loop_query_pts;
                                        for (const auto& match : loop_matches) 
                                        {
                                            if (match.trainIdx < 0 || match.trainIdx >= (int)loop_kp2.size() || 
                                                match.queryIdx < 0 || match.queryIdx >= (int)loop_kp1.size()) continue;
                                            
                                            loop_train_pts.push_back(loop_kp2[match.trainIdx]); 
                                            loop_query_pts.push_back(loop_kp1[match.queryIdx]); 
                                        }

                                        std::vector<cv::Point2f> loop_train_pts_undist, loop_query_pts_undist;
                                        if (cv::norm(dist_coeffs_) > 0.0001) 
                                        {
                                            cv::undistortPoints(loop_train_pts, loop_train_pts_undist, camera_matrix_, dist_coeffs_, cv::noArray(), camera_matrix_);
                                            cv::undistortPoints(loop_query_pts, loop_query_pts_undist, camera_matrix_, dist_coeffs_, cv::noArray(), camera_matrix_);
                                        }
                                        else 
                                        {
                                            loop_train_pts_undist = loop_train_pts;
                                            loop_query_pts_undist = loop_query_pts;
                                        }

                                        for (size_t i = 0; i < loop_train_pts.size(); ++i) 
                                        {
                                            cv::Point2f pt2d_train = loop_train_pts[i]; 
                                            cv::Point2f pt2d_query = loop_query_pts[i]; 

                                            float z_cand = get_robust_depth(candidate_kf.depth_image, pt2d_train.x, pt2d_train.y);
                                            float z_curr_loop = get_robust_depth(current_frame.depth_image, pt2d_query.x, pt2d_query.y);

                                            if (z_cand > 0.0f && z_curr_loop > 0.0f) 
                                            {
                                                float x_cand = (loop_train_pts_undist[i].x - cx) * z_cand / fx;
                                                float y_cand = (loop_train_pts_undist[i].y - cy) * z_cand / fy;
                                                
                                                float x_curr_l = (loop_query_pts_undist[i].x - cx) * z_curr_loop / fx;
                                                float y_curr_l = (loop_query_pts_undist[i].y - cy) * z_curr_loop / fy;
                                                
                                                loop_pts_cand.push_back(Eigen::Vector3d(x_cand, y_cand, z_cand));
                                                loop_pts_curr.push_back(Eigen::Vector3d(x_curr_l, y_curr_l, z_curr_loop));
                                            }
                                        }

                                        if (loop_pts_cand.size() >= 15) 
                                        {
                                            bool kabsch_loop_success = solveKabschRansac(loop_pts_cand, loop_pts_curr, 1000, threshold_sq, T_loop_relative, loop_inliers);

                                            if (kabsch_loop_success && loop_inliers.size() >= 15) 
                                            {
                                                loop_detected = true;
                                                num_loop_inliers = loop_inliers.size();
                                                RCLCPP_INFO(this->get_logger(), "!!! LOOP CLOSURE !!! Fechando ciclo entre KF %d e KF %d (Inliers: %d)", loop_candidate_id, keyframe_id_, num_loop_inliers);
                                            } 
                                        } 
                                    } 
                                } 
                            }

                            if (loop_detected) 
                            {
                                Eigen::Matrix4d relative_loop_eigen = T_loop_relative.inverse(); 
                                gtsam::Pose3 loop_pose_opt(relative_loop_eigen);
                                
                                gtsam::Pose3 loop_pose_base = T_base_opt_ * loop_pose_opt * T_base_opt_.inverse();

                                double loop_trans_dist = T_loop_relative.block<3,1>(0,3).norm();
                                double loop_rot_dist = Eigen::AngleAxisd(T_loop_relative.block<3,3>(0,0)).angle();

                                double mean_loop_depth = 0.0;
                                for (int idx : loop_inliers) { 
                                    mean_loop_depth += loop_pts_cand[idx].z(); 
                                }
                                mean_loop_depth /= std::max((double)loop_inliers.size(), 1.0); 

                                double inlier_ratio = 100.0 / std::max((double)loop_inliers.size(), 15.0); 
                                double penalty_inliers = inlier_ratio * inlier_ratio;
                                double penalty_motion = 1.0 + (loop_trans_dist * 2.0) + (loop_rot_dist * 2.0);
                                double penalty_depth = std::max(1.0, mean_loop_depth * mean_loop_depth * 0.5);

                                double base_var_trans = 0.01; 
                                double base_var_rot   = 0.07; 

                                double var_trans_xy = base_var_trans * penalty_inliers * penalty_motion;
                                double var_trans_z  = base_var_trans * penalty_inliers * penalty_motion * penalty_depth; 
                                double var_rot      = base_var_rot   * penalty_inliers * penalty_motion;

                                Eigen::MatrixXd cov_eigen = Eigen::MatrixXd::Zero(6, 6);
                                cov_eigen(0, 0) = var_rot;      
                                cov_eigen(1, 1) = var_rot;      
                                cov_eigen(2, 2) = var_rot;      
                                cov_eigen(3, 3) = var_trans_xy; 
                                cov_eigen(4, 4) = var_trans_xy; 
                                cov_eigen(5, 5) = var_trans_z;  

                                auto loop_noise = gtsam::noiseModel::Gaussian::Covariance(cov_eigen);
                                auto robust_loop_noise = gtsam::noiseModel::Robust::Create(
                                    gtsam::noiseModel::mEstimator::Huber::Create(1.345), loop_noise);

                                graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                                    loop_candidate_id, keyframe_id_, loop_pose_base, robust_loop_noise));

                                std::vector<std::pair<int, gtsam::Pose3>> optimized_poses_for_mapping;
                                for (const auto& key_value : optimized_estimates_) 
                                {
                                    int node_id = static_cast<int>(key_value.key);
                                    gtsam::Pose3 node_pose = key_value.value.cast<gtsam::Pose3>();
                                    optimized_poses_for_mapping.emplace_back(node_id, node_pose);
                                }
                                

                                mapping_node_->update_global_map(optimized_poses_for_mapping);
                                    
                                RCLCPP_INFO(this->get_logger(), "Loop Noise: Trans_Z=%.3f, Rot=%.3f (Inliers: %d, Depth: %.2f)", 
                                            var_trans_z, var_rot, (int)loop_inliers.size(), mean_loop_depth);
                            }
                            else
                            {
                                std::vector<std::pair<int, gtsam::Pose3>> optimized_poses_for_mapping;
                                optimized_poses_for_mapping.push_back(std::make_pair(keyframe_id_, current_global_pose));
                            
                                mapping_node_->update_global_map(optimized_poses_for_mapping);
                            }
                            
                            RCLCPP_INFO(this->get_logger(), "[GTSAM] Atualizando ISAM2...");

                            isam2_.update(graph_, initial_estimates_);
                            publish_factor_graph(graph_, isam2_.calculateEstimate());

                            optimized_estimates_ = isam2_.calculateEstimate();

                            graph_.resize(0);
                            initial_estimates_.clear();

                            gtsam::Pose3 corrected_pose = optimized_estimates_.at<gtsam::Pose3>(keyframe_id_);
                            Eigen::Matrix4d corrected_eigen = corrected_pose.matrix();
                            cv::eigen2cv(corrected_eigen, global_pose_);

                            last_keyframe_pose_ = global_pose_.clone();
                            keyframe_database_[keyframe_id_] = current_frame; 

                            if (has_gt_) 
                            {
                                gtsam::Pose3 relative_gt = initial_gt_pose_.inverse() * latest_gt_pose_;
                                
                                double slam_dist_euclidean = corrected_pose.translation().norm();
                                double gt_dist_euclidean = relative_gt.translation().norm();
                                
                                double trans_error = (corrected_pose.translation() - relative_gt.translation()).norm();
                                
                                double trans_error_pct = (total_gt_distance_ > 0.001) ? (trans_error / total_gt_distance_) * 100.0 : 0.0;

                                gtsam::Rot3 rot_diff = corrected_pose.rotation().between(relative_gt.rotation());
                                double rot_error_rad = gtsam::Rot3::Logmap(rot_diff).norm();
                                double rot_error_deg = rot_error_rad * (180.0 / M_PI);

                                double gt_rot_rad = gtsam::Rot3::Logmap(relative_gt.rotation()).norm();
                                double gt_rot_deg = gt_rot_rad * (180.0 / M_PI);
                                double rot_error_pct = (gt_rot_deg > 0.001) ? (rot_error_deg / gt_rot_deg) * 100.0 : 0.0;

                                RCLCPP_INFO(this->get_logger(), "--- COMPARACAO GROUND TRUTH ---");
                                RCLCPP_INFO(this->get_logger(), "Hodometro Total Percorrido (GT) : %.4f m", total_gt_distance_);
                                RCLCPP_INFO(this->get_logger(), "Distancia em Linha Reta (GT)    : %.4f m | SLAM Estimou: %.4f m", gt_dist_euclidean, slam_dist_euclidean);
                                RCLCPP_INFO(this->get_logger(), "Erro Absoluto Translacao        : %.4f m (%.2f%% da trajetoria)", trans_error, trans_error_pct);
                                RCLCPP_INFO(this->get_logger(), "Rotacao Total Real (GT)         : %.2f° | Erro Absoluto: %.2f° (%.2f%%)", gt_rot_deg, rot_error_deg, rot_error_pct);
                                RCLCPP_INFO(this->get_logger(), "Pose Relativa GT [X: %7.4f | Y: %7.4f | Z: %7.4f]", relative_gt.x(), relative_gt.y(), relative_gt.z());
                                RCLCPP_INFO(this->get_logger(), "-------------------------------");
                            }

                           

                            publish_gtsam_data(corrected_pose, msg_copy->header.stamp);

                            keyframe_id_++;
                            RCLCPP_INFO(this->get_logger(), "[GTSAM] Otimizacao concluida com sucesso.");
                        }
                    }
                }
                else 
                {
                    
                    tracking_lost_counter_++;
                    RCLCPP_WARN(this->get_logger(), "[FALHA CRITICA] Kabsch RANSAC nao obteve consenso (Inliers: %zu/8 minimos necessarios)", inliers.size());
                    return;
                }   
            }
            else 
            {
                

                tracking_lost_counter_++;
                RCLCPP_WARN(this->get_logger(), "[FALHA CRITICA] Pares 3D validos (%zu) insuficientes para resolver a geometria (4 minimos)", pts_kf.size());
                return;
            }
        }
    
        frame_count_++;
    }

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions dino_opts;
    dino_opts.arguments({"--ros-args", "-r", "__node:=dino_loop_node", "-p", "use_sim_time:=true"});

    rclcpp::NodeOptions mapping_opts;
    mapping_opts.arguments({"--ros-args", "-r", "__node:=mapping_node", "-p", "use_sim_time:=true"});


    auto dino_loop_node = std::make_shared<slam_core::DinoLoopNode>(dino_opts);
    auto mapping_node = std::make_shared<slam_core::Mapping>(mapping_opts);
    auto server_node = std::make_shared<SlamCoreNode>(dino_loop_node, mapping_node);

    rclcpp::executors::MultiThreadedExecutor executor;
    
    executor.add_node(dino_loop_node);
    executor.add_node(mapping_node);
    executor.add_node(server_node);
    
    executor.spin();

    rclcpp::shutdown();
    return 0;
}