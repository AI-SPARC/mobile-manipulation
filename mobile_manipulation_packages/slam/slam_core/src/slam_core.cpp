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
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h> 

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp" 
#include <opencv2/core/eigen.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <slam_core/DinoLoopNode.hpp>
#include <slam_core/Mapping.hpp>
#include <slam_core/ImuIntegration.hpp>

struct FrameData 
{ 
    int id; 
    cv::Mat image; 
    cv::Mat depth_image;
    std::string rgb_frame;
    std::string depth_frame;
    gtsam::Pose3 global_pose;
};

class SlamCoreNode : public rclcpp::Node 
{
public:
    SlamCoreNode(
        std::shared_ptr<slam_core::DinoLoopNode> dino_loop_node_node,
        std::shared_ptr<slam_core::Mapping> mapping_node,
        std::shared_ptr<slam_core::ImuIntegration> imu_integration_node
    ) : Node("slam_core_node") , 
        dino_loop_node_node_(dino_loop_node_node),
        mapping_node_(mapping_node),
        imu_integration_node_(imu_integration_node)
    {
        this->declare_parameter<std::string>("main_frame_id", "base_link");
        this->declare_parameter<bool>("use_imu", false);

        main_frame_id_ = this->get_parameter("main_frame_id").as_string();
        use_imu = this->get_parameter("use_imu").as_bool();
       
        rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();
        rclcpp::QoS default_qos(10);                     

     
        rgb_sub_.subscribe(this, "/camera/rgb/image_raw", sensor_qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, "/camera/depth/image_rect_raw", sensor_qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), rgb_sub_, depth_sub_);
        sync_->registerCallback(std::bind(&SlamCoreNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", default_qos, std::bind(&SlamCoreNode::cmd_vel_callback, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/depth/camera_info", default_qos, std::bind(&SlamCoreNode::camera_info_callback, this, std::placeholders::_1));

        gt_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ground_truth", default_qos, std::bind(&SlamCoreNode::ground_truth_callback, this, std::placeholders::_1));

        
        current_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/flann/current_image", sensor_qos);
        odometry_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/flann/odometry_matches", sensor_qos);
        
        
        graph_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("~/factor_graph", default_qos);
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odom", default_qos);
        graph_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/slam/graph_markers", default_qos);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/slam/trajectory_path", default_qos);

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
    std::shared_ptr<slam_core::ImuIntegration> imu_integration_node_;

    
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

    gtsam::imuBias::ConstantBias current_bias_;

    gtsam::Pose3 initial_gt_pose_;
    gtsam::Pose3 latest_gt_pose_;
    bool has_gt_ = false;
    bool first_gt_received_ = false;
    bool use_imu = false;

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

        visualization_msgs::msg::Marker nodes_marker;
        nodes_marker.header.frame_id = "map";
        nodes_marker.header.stamp = this->now();
        nodes_marker.ns = "gtsam_nodes";
        nodes_marker.id = 0;
        nodes_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        nodes_marker.action = visualization_msgs::msg::Marker::ADD;
        nodes_marker.pose.orientation.w = 1.0;
        nodes_marker.scale.x = 0.1; 
        nodes_marker.scale.y = 0.1;
        nodes_marker.scale.z = 0.1;
        nodes_marker.color.r = 0.0f;
        nodes_marker.color.g = 0.5f;
        nodes_marker.color.b = 1.0f;
        nodes_marker.color.a = 1.0f;

        for (const auto& key_value : current_estimate) {
            auto pose = key_value.value.cast<gtsam::Pose3>();
            geometry_msgs::msg::Point p;
            p.x = pose.x();
            p.y = pose.y();
            p.z = pose.z();
            nodes_marker.points.push_back(p);
        }
        marker_array.markers.push_back(nodes_marker);

        visualization_msgs::msg::Marker edges_marker;
        edges_marker.header.frame_id = "map";
        edges_marker.header.stamp = this->now();
        edges_marker.ns = "gtsam_edges";
        edges_marker.id = 1;
        edges_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        edges_marker.action = visualization_msgs::msg::Marker::ADD;
        edges_marker.pose.orientation.w = 1.0;
        edges_marker.scale.x = 0.02; 
        edges_marker.color.r = 0.0f;
        edges_marker.color.g = 1.0f;
        edges_marker.color.b = 0.0f;
        edges_marker.color.a = 0.8f;

        for (const auto& factor : graph) {
            auto between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
            if (between_factor) {
                gtsam::Key key1 = between_factor->key1();
                gtsam::Key key2 = between_factor->key2();

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
        if (camera_info_received_) return; 

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
        mapping_node_->set_camera_info(msg, main_frame_id_, 1000.0);
        camera_info_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Matriz da Camera Carregada! fx:%.1f, fy:%.1f, cx:%.1f, cy:%.1f", fx, fy, cx, cy);
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

    void publish_gtsam_data(const gtsam::Pose3& optimized_pose, const rclcpp::Time& stamp)
    {
        try 
        {
            // CORREÇÃO 1: Usa o Símbolo X na verificação E na extração da covariância
            if (!optimized_estimates_.exists(gtsam::symbol_shorthand::X(keyframe_id_ - 1))) return;
            gtsam::Matrix6 covariance_gtsam = isam2_.marginalCovariance(gtsam::symbol_shorthand::X(keyframe_id_ - 1));
            
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = stamp;
            odom_msg.header.frame_id = "odom";        
            odom_msg.child_frame_id = main_frame_id_;    

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
            t.child_frame_id = main_frame_id_;
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
                // CORREÇÃO 2: Pula as Velocidades (V) e os Biases (B) para não quebrar o cast para Pose3
                gtsam::Symbol sym(key_value.key);
                if (sym.chr() != 'x') continue;

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

            const gtsam::NonlinearFactorGraph& isam_graph = isam2_.getFactorsUnsafe();

            for (size_t i = 0; i < isam_graph.size(); ++i) 
            {
                auto factor = isam_graph.at(i);
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
                        p1.x = pose1_base.x(); p1.y = pose1_base.y(); p1.z = pose1_base.z();
                        p2.x = pose2_base.x(); p2.y = pose2_base.y(); p2.z = pose2_base.z();
                        
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
            RCLCPP_INFO(this->get_logger(), "Arestas (Fatores) Totais: %d", (int)isam2_.getFactorsUnsafe().size());
            RCLCPP_INFO(this->get_logger(), "Pose %s [X: %7.3f | Y: %7.3f | Z: %7.3f]", main_frame_id_.c_str(), base_pose.x(), base_pose.y(), base_pose.z());
            RCLCPP_INFO(this->get_logger(), "-----------------------");
        } 
        catch (const gtsam::IndeterminantLinearSystemException& e) {
            RCLCPP_WARN(this->get_logger(), "GTSAM IndeterminantLinearSystemException: Grafo instavel no momento.");
        }
        catch (const std::exception& e) {
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
    
    std::string rgb_frame = rgb_msg->header.frame_id;
    std::string depth_frame = depth_msg->header.frame_id;

    if (!tf_received_) 
    {
        if (main_frame_id_ == rgb_frame) 
        {
            T_base_opt_ = gtsam::Pose3();
            tf_received_ = true;
        }
        else 
        {
            try {
                geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                    main_frame_id_, rgb_frame, tf2::TimePointZero);

                Eigen::Quaterniond q(transform_stamped.transform.rotation.w, transform_stamped.transform.rotation.x,
                                    transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z);
                Eigen::Vector3d t(transform_stamped.transform.translation.x, transform_stamped.transform.translation.y,
                                transform_stamped.transform.translation.z);

                T_base_opt_ = gtsam::Pose3(gtsam::Rot3(q.toRotationMatrix()), gtsam::Point3(t));
                tf_received_ = true;
                RCLCPP_INFO(this->get_logger(), "Transformacao %s -> %s recebida!", main_frame_id_.c_str(), rgb_frame.c_str());
            }
            catch (tf2::TransformException &ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                                        "Aguardando TF de %s para %s...", main_frame_id_.c_str(), rgb_frame.c_str());
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
    current_frame.rgb_frame = rgb_frame;
    current_frame.depth_frame = depth_frame;


    auto solveWeightedKabsch = [](const std::vector<Eigen::Vector3d>& pts_source, 
                                  const std::vector<Eigen::Vector3d>& pts_target, 
                                  const std::vector<double>& weights, 
                                  Eigen::Matrix4d& out_T) -> bool {
        if (pts_source.empty() || pts_source.size() != pts_target.size() || pts_source.size() != weights.size()) {
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
        if (R.determinant() < 0) {
            V.col(2) *= -1.0;
            R = V * U.transpose();
        }

        Eigen::Vector3d t = centroid_target - R * centroid_source;
        
        out_T = Eigen::Matrix4d::Identity();
        out_T.block<3,3>(0,0) = R;
        out_T.block<3,1>(0,3) = t;

        return true;
    };

    

    if (!has_keyframe_) 
    {
        last_keyframe_ = current_frame;
        global_pose_ = cv::Mat::eye(4, 4, CV_64F);
        last_keyframe_pose_ = cv::Mat::eye(4, 4, CV_64F);
        gtsam::Pose3 initial_pose = gtsam::Pose3();
        
        
        auto prior_noise_pose = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
        graph_.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::symbol_shorthand::X(keyframe_id_), initial_pose, prior_noise_pose));
        initial_estimates_.insert(gtsam::symbol_shorthand::X(keyframe_id_), initial_pose);

        if (use_imu) 
        {
            gtsam::Vector3 initial_velocity(0.0, 0.0, 0.0);
            auto prior_noise_vel = gtsam::noiseModel::Isotropic::Sigma(3, 0.1); 
            graph_.add(gtsam::PriorFactor<gtsam::Vector3>(gtsam::symbol_shorthand::V(keyframe_id_), initial_velocity, prior_noise_vel));
            initial_estimates_.insert(gtsam::symbol_shorthand::V(keyframe_id_), initial_velocity);

            gtsam::imuBias::ConstantBias initial_bias; 
            auto prior_noise_bias = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
            graph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(gtsam::symbol_shorthand::B(0), initial_bias, prior_noise_bias));
            initial_estimates_.insert(gtsam::symbol_shorthand::B(0), initial_bias);

            current_bias_ = initial_bias;
        }

        current_frame.global_pose = initial_pose; 
        keyframe_database_[keyframe_id_] = current_frame;

        msg_copy->header.frame_id = std::to_string(keyframe_id_);
        dino_loop_node_node_->keyframe_callback(msg_copy);
        // mapping_node_->add_keyframe_data(keyframe_id_, current_frame.image, current_frame.depth_image, rgb_frame, depth_frame);
        
        std::vector<std::pair<int, gtsam::Pose3>> optimized_poses_for_mapping;
        optimized_poses_for_mapping.push_back(std::make_pair(0, initial_pose));
        // mapping_node_->update_global_map(optimized_poses_for_mapping);

        isam2_.update(graph_, initial_estimates_);
        optimized_estimates_ = isam2_.calculateEstimate();
        
        graph_.resize(0);
        initial_estimates_.clear();

        keyframe_id_++;
        has_keyframe_ = true;
        frame_count_++;

        return;
    }
    else if(tracking_lost_counter_ >= 8)
    {
        RCLCPP_WARN(this->get_logger(), "[SEQUESTRO] Robo perdido. Criando nova Ilha no GTSAM!");

        keyframe_id_ += 1000; 
        tracking_lost_counter_ = 0; 
        last_keyframe_ = current_frame;

        Eigen::Matrix4d global_pose_eigen;
        cv::cv2eigen(global_pose_, global_pose_eigen);
        gtsam::Pose3 current_global_pose(global_pose_eigen);

        auto prior_noise_pose = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 100.0, 100.0, 100.0, 100.0, 100.0, 100.0).finished());
        graph_.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::symbol_shorthand::X(keyframe_id_), current_global_pose, prior_noise_pose));
        initial_estimates_.insert(gtsam::symbol_shorthand::X(keyframe_id_), current_global_pose);
        
        if (use_imu) 
        {
            gtsam::Vector3 initial_velocity(0.0, 0.0, 0.0);
            auto prior_noise_vel = gtsam::noiseModel::Isotropic::Sigma(3, 10.0); 
            graph_.add(gtsam::PriorFactor<gtsam::Vector3>(gtsam::symbol_shorthand::V(keyframe_id_), initial_velocity, prior_noise_vel));
            initial_estimates_.insert(gtsam::symbol_shorthand::V(keyframe_id_), initial_velocity);

            gtsam::imuBias::ConstantBias initial_bias; 
            auto prior_noise_bias = gtsam::noiseModel::Isotropic::Sigma(6, 1e-1); 
            graph_.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(gtsam::symbol_shorthand::B(0), initial_bias, prior_noise_bias));
            initial_estimates_.insert(gtsam::symbol_shorthand::B(0), initial_bias);

            current_bias_ = initial_bias;

            if (imu_integration_node_) 
            {
                imu_integration_node_->getAndResetPreintegratedMeasurements(current_bias_);
            }
        }
       
        current_frame.global_pose = current_global_pose; 
        keyframe_database_[keyframe_id_] = current_frame; 

        msg_copy->header.frame_id = std::to_string(keyframe_id_);
        dino_loop_node_node_->keyframe_callback(msg_copy);
        // mapping_node_->add_keyframe_data(keyframe_id_, current_frame.image, current_frame.depth_image, rgb_frame, depth_frame);

        
        isam2_.update(graph_, initial_estimates_);
        if (use_imu) 
        {
            isam2_.update(); 
        }
        optimized_estimates_ = isam2_.calculateEstimate();
        
        graph_.resize(0);
        initial_estimates_.clear();

        keyframe_id_++;
        has_keyframe_ = true;
        frame_count_++;

        std::vector<std::pair<int, gtsam::Pose3>> optimized_poses_for_mapping;
        optimized_poses_for_mapping.push_back(std::make_pair(0, current_global_pose));
        // mapping_node_->update_global_map(optimized_poses_for_mapping);
        
        return;
    }

    std::vector<cv::Point2f> kp1, kp2;
    std::vector<cv::DMatch> matches;
    dino_loop_node_node_->compute_matches(current_frame.image, last_keyframe_.image, kp1, kp2, matches);
    
    if (!matches.empty())
    {
        cv::Mat debug_image;
        cv::hconcat(current_frame.image, last_keyframe_.image, debug_image);
        for (const auto& match : matches) {
            if (match.queryIdx >= 0 && match.queryIdx < (int)kp1.size() && match.trainIdx >= 0 && match.trainIdx < (int)kp2.size()) {
                cv::Point2f pt_current = kp1[match.queryIdx];
                cv::Point2f pt_keyframe = kp2[match.trainIdx];
                pt_keyframe.x += current_frame.image.cols; 
                cv::circle(debug_image, pt_current, 3, cv::Scalar(0, 255, 0), -1); 
                cv::circle(debug_image, pt_keyframe, 3, cv::Scalar(0, 0, 255), -1); 
            }
        }
        std_msgs::msg::Header match_header;
        match_header.stamp = now;
        match_header.frame_id = rgb_frame; 
        odometry_pub_->publish(*cv_bridge::CvImage(match_header, "bgr8", debug_image).toImageMsg());
    }
    
    bool tracking_success = false; 

    if (matches.size() >= 5) 
    {
        int edge_filter_rejected = 0;
        int spatial_variance_rejected = 0; 

        std::vector<cv::Point2f> train_pts, query_pts;
        for (const auto& match : matches) 
        {
            if (match.trainIdx < 0 || match.trainIdx >= (int)kp2.size() || 
                match.queryIdx < 0 || match.queryIdx >= (int)kp1.size()) continue;
            train_pts.push_back(kp2[match.trainIdx]); 
            query_pts.push_back(kp1[match.queryIdx]); 
        }

        std::vector<cv::Point2f> train_pts_undist, query_pts_undist;
        if (cv::norm(dist_coeffs_) > 0.0001) 
        {
            cv::undistortPoints(train_pts, train_pts_undist, camera_matrix_, dist_coeffs_, cv::noArray(), camera_matrix_);
            cv::undistortPoints(query_pts, query_pts_undist, camera_matrix_, dist_coeffs_, cv::noArray(), camera_matrix_);
        } 
        else 
        {
            train_pts_undist = train_pts;
            query_pts_undist = query_pts;
        }

        std::vector<cv::Point3f> object_pts_3d; 
        std::vector<cv::Point2f> image_pts_2d;  
        
        for (size_t i = 0; i < train_pts.size(); ++i) 
        {
            cv::Point2f pt2d_train = train_pts[i]; 
            cv::Point2f pt2d_query = query_pts[i]; 

            float z_center = get_robust_depth(last_keyframe_.depth_image, pt2d_train.x, pt2d_train.y);
            
            if (z_center <= 0.1f || z_center > 7.0) 
            {
                edge_filter_rejected++; 
                continue; 
            }

            float min_z = z_center, max_z = z_center;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue; 
                    
                    float z_neighbor = get_robust_depth(last_keyframe_.depth_image, pt2d_train.x + dx, pt2d_train.y + dy);
                    if (z_neighbor > 0.1f) {
                        min_z = std::min(min_z, z_neighbor);
                        max_z = std::max(max_z, z_neighbor);
                    }
                }
            }
            
          
            float x_kf = (train_pts_undist[i].x - cx) * z_center / fx;
            float y_kf = (train_pts_undist[i].y - cy) * z_center / fy;
            
            object_pts_3d.push_back(cv::Point3f(x_kf, y_kf, z_center));
            image_pts_2d.push_back(query_pts_undist[i]); 
        }

        if (object_pts_3d.size() >= 8) 
        { 
            int iterationsCount = 1000;
            float reprojectionError = 10.0f; 
            double confidence = 0.96; 
            
            cv::Mat empty_dist_coeffs = cv::Mat::zeros(4, 1, CV_64F); 
            
            
            cv::Mat rvec_guess = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat tvec_guess = cv::Mat::zeros(3, 1, CV_64F);

         
            Eigen::Matrix4d global_pose_eigen, last_kf_pose_eigen;
            cv::cv2eigen(global_pose_, global_pose_eigen);
            cv::cv2eigen(last_keyframe_pose_, last_kf_pose_eigen);

            Eigen::Matrix4d delta_base_guess = last_kf_pose_eigen.inverse() * global_pose_eigen;

            gtsam::Pose3 delta_base_gtsam(delta_base_guess);
            gtsam::Pose3 delta_cam_guess = T_base_opt_.inverse() * delta_base_gtsam * T_base_opt_;

            Eigen::Matrix4d T_pnp_guess = delta_cam_guess.inverse().matrix();

            Eigen::Matrix3d R_guess = T_pnp_guess.block<3,3>(0,0);
            Eigen::Vector3d t_guess_eigen = T_pnp_guess.block<3,1>(0,3);

            cv::Mat R_cv;
            cv::eigen2cv(R_guess, R_cv);
            cv::Rodrigues(R_cv, rvec_guess); 

            tvec_guess.at<double>(0) = t_guess_eigen(0);
            tvec_guess.at<double>(1) = t_guess_eigen(1);
            tvec_guess.at<double>(2) = t_guess_eigen(2);

           
            std::vector<int> inliers;
            
            bool pnp_success = cv::solvePnPRansac(
                object_pts_3d,      
                image_pts_2d,       
                camera_matrix_,     
                empty_dist_coeffs,       
                rvec_guess,         
                tvec_guess,         
                true,               
                iterationsCount, 
                reprojectionError, 
                confidence, 
                inliers,
                cv::SOLVEPNP_ITERATIVE 
            );
            
            if (pnp_success && inliers.size() >= 8 && !rvec_guess.empty() && !tvec_guess.empty()) 
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

                for (int idx : inliers) 
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

                Eigen::Matrix4d T_curr_kf = T_camera_world_refined.inverse();

                Eigen::Matrix3d R_opt = T_curr_kf.block<3,3>(0,0);
                Eigen::Vector3d t_opt = T_curr_kf.block<3,1>(0,3);

                double translation_dist = t_opt.norm();
                Eigen::AngleAxisd aa(R_opt);
                double rotation_dist = aa.angle(); 

                Eigen::Matrix4d delta_opt_eigen = T_curr_kf;
                gtsam::Pose3 delta_opt(delta_opt_eigen);
                gtsam::Pose3 delta_base = T_base_opt_ * delta_opt * T_base_opt_.inverse();
                double real_dist = delta_base.translation().norm();
                
                if (real_dist > 1.0) 
                {
                    RCLCPP_WARN(this->get_logger(), "[REJEICAO] Pulo absurdo detectado (>1m). Rastreamento rejeitado.");
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
                    for (int idx : inliers) mean_depth += object_pts_3d[idx].z;
                    mean_depth /= inliers.size();
                    double penalty_depth = std::max(1.0, mean_depth * mean_depth * 0.5);

                    double base_var_trans = 0.01; 
                    double base_var_rot   = 0.07; 

                    Eigen::MatrixXd cov_eigen = Eigen::MatrixXd::Zero(6, 6);
                    cov_eigen(0, 0) = base_var_rot * penalty_inliers * penalty_motion;
                    cov_eigen(1, 1) = base_var_rot * penalty_inliers * penalty_motion; 
                    cov_eigen(2, 2) = base_var_rot * penalty_inliers * penalty_motion;   
                    cov_eigen(3, 3) = base_var_trans * penalty_inliers * penalty_motion;     
                    cov_eigen(4, 4) = base_var_trans * penalty_inliers * penalty_motion;     
                    cov_eigen(5, 5) = base_var_trans * penalty_inliers * penalty_motion * penalty_depth;     

                    Eigen::Matrix4d global_pose_eigen;
                    cv::cv2eigen(global_pose_, global_pose_eigen);
                    gtsam::Pose3 current_global_pose(global_pose_eigen);

                    // if (has_gt_) 
                    // {
                    //     gtsam::Pose3 relative_gt = initial_gt_pose_.inverse() * latest_gt_pose_;
                    
                    //     double trans_error = (current_global_pose.translation() - relative_gt.translation()).norm();
                    //     double trans_error_pct = (total_gt_distance_ > 0.001) ? (trans_error / total_gt_distance_) * 100.0 : 0.0;

                    //     gtsam::Rot3 rot_diff = current_global_pose.rotation().between(relative_gt.rotation());
                    //     double rot_error_rad = gtsam::Rot3::Logmap(rot_diff).norm();
                    //     double rot_error_deg = rot_error_rad * (180.0 / M_PI);
                    //     double gt_rot_rad = gtsam::Rot3::Logmap(relative_gt.rotation()).norm();
                    //     double gt_rot_deg = gt_rot_rad * (180.0 / M_PI);
                    //     double rot_error_pct = (gt_rot_deg > 0.001) ? (rot_error_deg / gt_rot_deg) * 100.0 : 0.0;

                    //     RCLCPP_INFO(this->get_logger(), "--- COMPARACAO GROUND TRUTH ---");
                    //     RCLCPP_INFO(this->get_logger(), "Erro Absoluto Translacao        : %.4f m (%.2f%%)", trans_error, trans_error_pct);
                    //     RCLCPP_INFO(this->get_logger(), "Erro Absoluto Rotacao           : %.2f° (%.2f%%)", rot_error_deg, rot_error_pct);
                    // }

                    // publish_gtsam_data(current_global_pose, msg_copy->header.stamp);

                    if (translation_dist > 0.15 || rotation_dist > 0.1) 
                    {
                        last_keyframe_ = current_frame;
                        auto visual_noise = gtsam::noiseModel::Gaussian::Covariance(cov_eigen);


                        

                        if(use_imu)
                        {
                            auto preint_imu = imu_integration_node_->getAndResetPreintegratedMeasurements(current_bias_);

                       
                            graph_.add(gtsam::ImuFactor(
                                gtsam::symbol_shorthand::X(keyframe_id_ - 1), gtsam::symbol_shorthand::V(keyframe_id_ - 1), 
                                gtsam::symbol_shorthand::X(keyframe_id_),     gtsam::symbol_shorthand::V(keyframe_id_),     
                                gtsam::symbol_shorthand::B(0),                                                            
                                *preint_imu
                            ));
                        }


                        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                            gtsam::symbol_shorthand::X(keyframe_id_ - 1), 
                            gtsam::symbol_shorthand::X(keyframe_id_), 
                            delta_base, 
                            visual_noise
                        ));

                        initial_estimates_.insert(gtsam::symbol_shorthand::X(keyframe_id_), current_global_pose);


                        gtsam::Vector3 last_vel(0.0, 0.0, 0.0); 
                        
                        if (optimized_estimates_.exists(gtsam::symbol_shorthand::V(keyframe_id_ - 1))) 
                        {
                            last_vel = optimized_estimates_.at<gtsam::Vector3>(gtsam::symbol_shorthand::V(keyframe_id_ - 1));
                        } 
                        else 
                        {
                            RCLCPP_WARN(this->get_logger(), 
                                "[SALVAMENTO] Velocidade V(%d) nao encontrada na arvore. Usando (0,0,0) para evitar crash.", 
                                keyframe_id_ - 1);
                        }
                        
                        initial_estimates_.insert(gtsam::symbol_shorthand::V(keyframe_id_), last_vel);
                        
                        msg_copy->header.frame_id = std::to_string(keyframe_id_);
                        
                        int loop_candidate_id = dino_loop_node_node_->keyframe_callback(msg_copy);
                        // mapping_node_->add_keyframe_data(keyframe_id_, current_frame.image, current_frame.depth_image, rgb_frame, depth_frame);
                        
                        bool loop_detected = false;
                        Eigen::Matrix4d T_loop_relative = Eigen::Matrix4d::Identity(); 
                        int num_loop_inliers = 0;
                        
                        std::vector<cv::Point3f> loop_object_pts_3d; 
                        std::vector<cv::Point2f> loop_image_pts_2d;  
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
                                        if (match.trainIdx < 0 || match.trainIdx >= (int)loop_kp2.size() || match.queryIdx < 0 || match.queryIdx >= (int)loop_kp1.size()) continue;
                                        
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
                                        float z_cand = get_robust_depth(candidate_kf.depth_image, loop_train_pts[i].x, loop_train_pts[i].y);

                                        if (z_cand > 0.1f && z_cand < 7.0) 
                                        {
                                            float x_cand = (loop_train_pts_undist[i].x - cx) * z_cand / fx;
                                            float y_cand = (loop_train_pts_undist[i].y - cy) * z_cand / fy;
                                            
                                            loop_object_pts_3d.push_back(cv::Point3f(x_cand, y_cand, z_cand));
                                            loop_image_pts_2d.push_back(loop_query_pts_undist[i]);
                                        }
                                    }

                                    if (loop_object_pts_3d.size() >= 15) 
                                    {
                                        cv::Mat rvec_loop, tvec_loop;
                                        cv::Mat empty_loop_coeffs = cv::Mat::zeros(4, 1, CV_64F); 
                                        
                                        bool pnp_loop_success = cv::solvePnPRansac(
                                            loop_object_pts_3d, loop_image_pts_2d, camera_matrix_, empty_loop_coeffs, 
                                            rvec_loop, tvec_loop, false, 1000, 10.0f, 0.95, loop_inliers, cv::SOLVEPNP_SQPNP
                                        );
                                        
                                        if (pnp_loop_success && loop_inliers.size() >= 15 && !rvec_loop.empty() && !tvec_loop.empty()) 
                                        {
                                            cv::Mat R_loop;
                                            cv::Rodrigues(rvec_loop, R_loop);

                                            Eigen::Matrix4d T_loop_world = Eigen::Matrix4d::Identity();
                                            for(int r = 0; r < 3; r++) {
                                                for(int c = 0; c < 3; c++) {
                                                    T_loop_world(r,c) = R_loop.at<double>(r,c);
                                                }
                                                T_loop_world(r,3) = tvec_loop.at<double>(r);
                                            }

                                            T_loop_relative = T_loop_world.inverse();

                                            loop_detected = true;
                                            num_loop_inliers = loop_inliers.size();
                                            RCLCPP_INFO(this->get_logger(), "!!! LOOP CLOSURE !!! Fechando ciclo entre KF %d e KF %d (Inliers PnP: %d)", loop_candidate_id, keyframe_id_, num_loop_inliers);
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
                            for (int idx : loop_inliers) mean_loop_depth += loop_object_pts_3d[idx].z;
                            mean_loop_depth /= std::max((double)loop_inliers.size(), 1.0); 

                            double inlier_ratio = 100.0 / std::max((double)loop_inliers.size(), 15.0); 
                            double penalty_inliers = inlier_ratio * inlier_ratio;
                            double penalty_motion = 1.0 + (loop_trans_dist * 2.0) + (loop_rot_dist * 2.0);
                            double penalty_depth = std::max(1.0, mean_loop_depth * mean_loop_depth * 0.5);

                            Eigen::MatrixXd cov_eigen = Eigen::MatrixXd::Zero(6, 6);
                            cov_eigen(0, 0) = base_var_rot * penalty_inliers * penalty_motion;      
                            cov_eigen(1, 1) = base_var_rot * penalty_inliers * penalty_motion;      
                            cov_eigen(2, 2) = base_var_rot * penalty_inliers * penalty_motion;      
                            cov_eigen(3, 3) = base_var_trans * penalty_inliers * penalty_motion; 
                            cov_eigen(4, 4) = base_var_trans * penalty_inliers * penalty_motion; 
                            cov_eigen(5, 5) = base_var_trans * penalty_inliers * penalty_motion * penalty_depth;  

                            auto loop_noise = gtsam::noiseModel::Gaussian::Covariance(cov_eigen);
                            auto robust_loop_noise = gtsam::noiseModel::Robust::Create(
                                gtsam::noiseModel::mEstimator::Huber::Create(1.345), loop_noise);

                            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                                gtsam::symbol_shorthand::X(loop_candidate_id), 
                                gtsam::symbol_shorthand::X(keyframe_id_), 
                                loop_pose_base, 
                                robust_loop_noise
                            ));

                            std::vector<std::pair<int, gtsam::Pose3>> optimized_poses_for_mapping;
                            for (const auto& key_value : optimized_estimates_) 
                            {
                                gtsam::Symbol sym(key_value.key);
                                
                                if (sym.chr() == 'x') 
                                {
                                    optimized_poses_for_mapping.emplace_back(
                                        static_cast<int>(sym.index()), 
                                        key_value.value.cast<gtsam::Pose3>()
                                    );
                                }
                            }
                            // mapping_node_->update_global_map(optimized_poses_for_mapping);
                        }
                        else
                        {
                            std::vector<std::pair<int, gtsam::Pose3>> optimized_poses_for_mapping;
                            optimized_poses_for_mapping.push_back(std::make_pair(keyframe_id_, current_global_pose));
                            // mapping_node_->update_global_map(optimized_poses_for_mapping);
                        }
                        
                        isam2_.update(graph_, initial_estimates_);
                        publish_factor_graph(graph_, isam2_.calculateEstimate());
                        optimized_estimates_ = isam2_.calculateEstimate();

                        gtsam::Pose3 corrected_pose = optimized_estimates_.at<gtsam::Pose3>(gtsam::symbol_shorthand::X(keyframe_id_));

                        if(use_imu)
                        {
                            current_bias_ = optimized_estimates_.at<gtsam::imuBias::ConstantBias>(gtsam::symbol_shorthand::B(0));
                        }
                        graph_.resize(0);
                        initial_estimates_.clear();

                        Eigen::Matrix4d corrected_eigen = corrected_pose.matrix();
                        cv::eigen2cv(corrected_eigen, global_pose_);

                        last_keyframe_pose_ = global_pose_.clone();
                        current_frame.global_pose = corrected_pose; 
        
                        keyframe_database_[keyframe_id_] = current_frame;

                        if (has_gt_) 
                        {
                            gtsam::Pose3 relative_gt = initial_gt_pose_.inverse() * latest_gt_pose_;
                            
                            
                            
                            double trans_error = (corrected_pose.translation() - relative_gt.translation()).norm();
                            double trans_error_pct = (total_gt_distance_ > 0.001) ? (trans_error / total_gt_distance_) * 100.0 : 0.0;

                            gtsam::Rot3 rot_diff = corrected_pose.rotation().between(relative_gt.rotation());
                            double rot_error_rad = gtsam::Rot3::Logmap(rot_diff).norm();
                            double rot_error_deg = rot_error_rad * (180.0 / M_PI);
                            double gt_rot_rad = gtsam::Rot3::Logmap(relative_gt.rotation()).norm();
                            double gt_rot_deg = gt_rot_rad * (180.0 / M_PI);
                            double rot_error_pct = (gt_rot_deg > 0.001) ? (rot_error_deg / gt_rot_deg) * 100.0 : 0.0;

                            RCLCPP_INFO(this->get_logger(), "--- COMPARACAO GROUND TRUTH ---");
                            RCLCPP_INFO(this->get_logger(), "Erro Absoluto Translacao        : %.4f m (%.2f%%)", trans_error, trans_error_pct);
                            RCLCPP_INFO(this->get_logger(), "Erro Absoluto Rotacao           : %.2f° (%.2f%%)", rot_error_deg, rot_error_pct);
                        }

                        publish_gtsam_data(corrected_pose, msg_copy->header.stamp);
                        keyframe_id_++;
                    }
                }
            }
            else 
            {
                tracking_lost_counter_++;
                RCLCPP_WARN(this->get_logger(), 
                    "[Rastreamento Perdido] PnP RANSAC falhou ou não obteve consenso. "
                    "Apenas %zu inliers de 8 necessarios. (Causa: Movimento muito brusco ou ruido extremo)", 
                    inliers.size());
                return;
            }   
        }
        else 
        {
            tracking_lost_counter_++;
            RCLCPP_WARN(this->get_logger(), 
            "[Rastreamento Perdido] Pontos 3D validos insuficientes (%zu/8 minimos). "
            "(Causa: Falha no LightGlue ou profundidade invalida/cega no sensor)", 
            object_pts_3d.size());
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

    rclcpp::NodeOptions imu_integration_opts;
    imu_integration_opts.arguments({"--ros-args", "-r", "__node:=imu_integration_node", "-p", "use_sim_time:=true"});

    auto dino_loop_node = std::make_shared<slam_core::DinoLoopNode>(dino_opts);
    auto mapping_node = std::make_shared<slam_core::Mapping>(mapping_opts);
    auto imu_integration_node = std::make_shared<slam_core::ImuIntegration>(imu_integration_opts);

    auto server_node = std::make_shared<SlamCoreNode>(dino_loop_node, mapping_node, imu_integration_node);

    rclcpp::executors::MultiThreadedExecutor executor;
    
    executor.add_node(dino_loop_node);
    executor.add_node(mapping_node);
    executor.add_node(imu_integration_node);
    executor.add_node(server_node);
    
    executor.spin();

    rclcpp::shutdown();
    return 0;
}