#include <memory>
#include <vector>
#include <iostream>
#include <chrono> 
#include <map>
#include <cmath> 

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp" 
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/path.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <nav_msgs/msg/odometry.hpp>

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "DBoW3/DBoW3.h"

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

struct FrameData 
{
    int frame_id;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point3f> points_3d; 
};

class BowLoopNode : public rclcpp::Node 
{
public:
    BowLoopNode() : Node("bow_loop_node") 
    {
        std::string vocab_path = "/home/momesso/ORBvoc.txt"; 
        
        RCLCPP_INFO(this->get_logger(), "Carregando DBoW3...");
        voc_.load(vocab_path);
        
        if (voc_.empty()) 
        {
            RCLCPP_ERROR(this->get_logger(), "FALHA CRITICA: Vocabulario nao encontrado.");
            throw std::runtime_error("Vocabulario ausente.");
        }   
        
        RCLCPP_INFO(this->get_logger(), "Vocabulario carregado! Palavras: %d", (int)voc_.size());

        db_.setVocabulary(voc_, false, 0);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/rgb/image_raw", 10, std::bind(&BowLoopNode::image_callback, this, std::placeholders::_1));
        
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_rect_raw", 10, std::bind(&BowLoopNode::depth_callback, this, std::placeholders::_1));

        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10, std::bind(&BowLoopNode::cmd_vel_callback, this, std::placeholders::_1));

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/depth/camera_info", 10, std::bind(&BowLoopNode::camera_info_callback, this, std::placeholders::_1));

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

        last_processed_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "--- NO DE ODOMETRIA VISUAL E GTSAM INICIADO ---");
        RCLCPP_INFO(this->get_logger(), "Aguardando topico /camera/depth/camera_info...");
    }

private:


    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_estimates_;
    gtsam::Values optimized_estimates_;
    int keyframe_id_ = 0;
    std::map<int, FrameData> keyframe_database_;
    cv::Mat last_keyframe_pose_;
    FrameData last_keyframe_;
    bool has_keyframe_ = false;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_; 
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_; 
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr current_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr odometry_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_markers_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

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

    DBoW3::Vocabulary voc_;
    DBoW3::Database db_;
    std::vector<FrameData> history_frames_;
    
    int frame_count_ = 0;
    bool is_moving_ = false;
    rclcpp::Time last_processed_time_;





    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) 
    {
        is_moving_ = (std::abs(msg->linear.x) > 0.01 || std::abs(msg->angular.z) > 0.01);
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
        try 
        {
            last_depth_msg_ = cv_bridge::toCvCopy(msg, msg->encoding);
        } 
        catch (cv_bridge::Exception& e) 
        {
            RCLCPP_ERROR(this->get_logger(), "Erro na profundidade: %s", e.what());
        }
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

    void publish_gtsam_data(const gtsam::Pose3& optimized_pose)
    {
        try 
        {
            gtsam::Marginals marginals(graph_, optimized_estimates_);
            gtsam::Matrix6 covariance_gtsam = marginals.marginalCovariance(keyframe_id_ - 1);
            
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = this->now();
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
            path_msg.header.stamp = this->now();
            path_msg.header.frame_id = "odom"; 

            visualization_msgs::msg::Marker nodes_marker;
            nodes_marker.header.frame_id = "odom"; 
            nodes_marker.header.stamp = this->now();
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
            edges_marker.header.stamp = this->now();
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



    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) 
    {
        if (!tf_received_) 
        {
            try 
            {
                geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                    "base_link", "Camera_Pseudo_Depth", tf2::TimePointZero);

                Eigen::Quaterniond q(
                    transform_stamped.transform.rotation.w,
                    transform_stamped.transform.rotation.x,
                    transform_stamped.transform.rotation.y,
                    transform_stamped.transform.rotation.z
                );
                
                Eigen::Vector3d t(
                    transform_stamped.transform.translation.x,
                    transform_stamped.transform.translation.y,
                    transform_stamped.transform.translation.z
                );

                
                T_base_opt_ = gtsam::Pose3(gtsam::Rot3(q.toRotationMatrix()), gtsam::Point3(t));
                
                tf_received_ = true;
                RCLCPP_INFO(this->get_logger(), "Transformacao base_link -> Camera recebida e alinhada!");
            }
            catch (tf2::TransformException &ex) 
            {
                RCLCPP_WARN(this->get_logger(), "Aguardando TF base_link -> Camera_Pseudo_Depth: %s", ex.what());
                return;
            }
        }

        if (!camera_info_received_ || !is_moving_ || !last_depth_msg_) 
        {
            return;
        }

        auto now = this->now();
        last_processed_time_ = now;

        cv_bridge::CvImagePtr cv_ptr;
        try 
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } 
        catch (cv_bridge::Exception& e) 
        { 
            return; 
        }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb_->detectAndCompute(cv_ptr->image, cv::noArray(), keypoints, descriptors);

        if (descriptors.empty()) 
        {
            return;
        }

        std::vector<cv::Point3f> points_3d;
        double fx = camera_matrix_.at<double>(0, 0);
        double fy = camera_matrix_.at<double>(1, 1);
        double cx = camera_matrix_.at<double>(0, 2);
        double cy = camera_matrix_.at<double>(1, 2);

        for (const auto& kp : keypoints) 
        {
            float z = get_depth_meters(last_depth_msg_->image, (int)kp.pt.x, (int)kp.pt.y);

            if (z > 0.1f && z < 5.0f) 
            { 
                float x = (kp.pt.x - cx) * z / fx;
                float y = (kp.pt.y - cy) * z / fy;
                points_3d.push_back(cv::Point3f(x, y, z));
            } 
            else 
            {
                points_3d.push_back(cv::Point3f(0, 0, 0)); 
            }
        }
       
        FrameData current_frame = {frame_count_, cv_ptr->image.clone(), keypoints, descriptors.clone(), points_3d};

        if (!has_keyframe_) 
        {
            last_keyframe_ = current_frame;
            
            
            global_pose_ = cv::Mat::eye(4, 4, CV_64F);
            last_keyframe_pose_ = cv::Mat::eye(4, 4, CV_64F);
            gtsam::Pose3 initial_pose = gtsam::Pose3();
            
            auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
            
            graph_.add(gtsam::PriorFactor<gtsam::Pose3>(keyframe_id_, initial_pose, prior_noise));
            initial_estimates_.insert(keyframe_id_, initial_pose);
            
            keyframe_database_[keyframe_id_] = current_frame; 
            db_.add(current_frame.descriptors);
            
            keyframe_id_++;
            has_keyframe_ = true;
            frame_count_++;
            return;
        }
       
        std::vector<std::vector<cv::DMatch>> knn_matches;
        local_matcher_->knnMatch(descriptors, last_keyframe_.descriptors, knn_matches, 2);

        std::vector<cv::DMatch> good_matches;
        for (auto& m : knn_matches) 
        {
            if (m.size() >= 2 && m[0].distance < 0.5f * m[1].distance) 
            {
                good_matches.push_back(m[0]);
            }
        }

        bool tracking_success = false; 

        if (good_matches.size() >= 10) 
        {
            std::vector<cv::Point3f> object_points; 
            std::vector<cv::Point2f> image_points;  

            for (const auto& match : good_matches) 
            {
                cv::Point3f pt3d = last_keyframe_.points_3d[match.trainIdx];

                if (pt3d.z > 0.0f) 
                { 
                    object_points.push_back(pt3d);
                    image_points.push_back(keypoints[match.queryIdx].pt);
                }
            }

            if (object_points.size() >= 15) 
            { 
                cv::Mat rvec, tvec;
                std::vector<int> inliers;
                
                bool pnp_success = cv::solvePnPRansac(
                    object_points, image_points, camera_matrix_, dist_coeffs_,
                    rvec, tvec, false, 100, 2.0f, 0.99, inliers, cv::SOLVEPNP_EPNP);
                
                if (pnp_success && inliers.size() >= 15) 
                {
                    double translation_dist = cv::norm(tvec);
                    double rotation_dist = cv::norm(rvec);

                    if (translation_dist > 1.0) 
                    {
                        RCLCPP_WARN(this->get_logger(), "!!! EXPLOSAO EVITADA !!! Pulo irreal de %.2f metros.", translation_dist);
                        tracking_success = false; 
                    } 
                    else 
                    {
                        tracking_success = true; 

                        cv::Mat R;
                        cv::Rodrigues(rvec, R);
                        
                        cv::Mat T_curr_kf = cv::Mat::eye(4, 4, CV_64F);
                        R.copyTo(T_curr_kf(cv::Rect(0, 0, 3, 3)));
                        tvec.copyTo(T_curr_kf(cv::Rect(3, 0, 1, 3)));

                       
                        cv::Mat T_kf_curr = T_curr_kf.inv();
                        Eigen::Matrix4d delta_opt_eigen;
                        cv::cv2eigen(T_kf_curr, delta_opt_eigen);
                        gtsam::Pose3 delta_opt(delta_opt_eigen);

                        
                        gtsam::Pose3 delta_base = T_base_opt_ * delta_opt * T_base_opt_.inverse();

                        Eigen::Matrix4d delta_base_eigen = delta_base.matrix();
                        cv::Mat delta_base_cv;
                        cv::eigen2cv(delta_base_eigen, delta_base_cv);
                        
                        
                        global_pose_ = last_keyframe_pose_ * delta_base_cv;

                        
                        double inlier_ratio = 100.0 / std::max((double)inliers.size(), 15.0);
                        double penalty_inliers = inlier_ratio * inlier_ratio; 
                        double penalty_motion = 1.0 + (translation_dist * 2.0) + (rotation_dist * 2.0);
                        
                        double mean_depth = 0.0;
                        for (int idx : inliers) 
                        {
                            mean_depth += object_points[idx].z;
                        }
                        mean_depth /= inliers.size();
                        double penalty_depth = std::max(1.0, mean_depth * mean_depth * 0.5);

                        double base_var_trans = 0.0002; 
                        double base_var_rot   = 0.0002; 

                        double var_x = base_var_trans * penalty_inliers * penalty_motion;
                        double var_y = base_var_trans * penalty_inliers * penalty_motion;
                        double var_z = base_var_trans * penalty_inliers * penalty_motion * penalty_depth; 
                        double var_roll  = base_var_rot * penalty_inliers * penalty_motion;
                        double var_pitch = base_var_rot * penalty_inliers * penalty_motion;
                        double var_yaw   = base_var_rot * penalty_inliers * penalty_motion;
                        
                        cv::Mat covariance = cv::Mat::zeros(6, 6, CV_64F);
                        covariance.at<double>(0, 0) = var_x;
                        covariance.at<double>(1, 1) = var_y;
                        covariance.at<double>(2, 2) = var_z;
                        covariance.at<double>(3, 3) = var_roll;
                        covariance.at<double>(4, 4) = var_pitch;
                        covariance.at<double>(5, 5) = var_yaw;

                        Eigen::Matrix4d global_pose_eigen;
                        cv::cv2eigen(global_pose_, global_pose_eigen);
                        gtsam::Pose3 current_global_pose(global_pose_eigen);
                        publish_gtsam_data(current_global_pose);

                        
                        if (translation_dist > 0.2 || rotation_dist > 0.1 || inliers.size() < 25) 
                        {
                            last_keyframe_ = current_frame;

                            Eigen::MatrixXd cov_eigen;
                            cv::cv2eigen(covariance, cov_eigen);
                            auto noise_model = gtsam::noiseModel::Gaussian::Covariance(cov_eigen);

                            
                            graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                                keyframe_id_ - 1, keyframe_id_, delta_base, noise_model));
                                
                            initial_estimates_.insert(keyframe_id_, current_global_pose);

                            
                            DBoW3::QueryResults results;
                            int max_results = 50; 
                            db_.query(current_frame.descriptors, results, max_results);

                            bool loop_detected = false;
                            int loop_candidate_id = -1;
                            cv::Mat T_loop_relative; 

                            for (const auto& result : results) 
                            {
                                if (keyframe_id_ - result.Id < 3) continue;

                                if (result.Score > 0.1) 
                                {
                                    FrameData candidate_kf = keyframe_database_[result.Id];
                                    
                                    std::vector<std::vector<cv::DMatch>> loop_knn_matches;
                                    local_matcher_->knnMatch(current_frame.descriptors, candidate_kf.descriptors, loop_knn_matches, 2);
                                    
                                    std::vector<cv::DMatch> loop_good_matches;
                                    for (auto& m : loop_knn_matches) {
                                        if (m.size() >= 2 && m[0].distance < 0.7f * m[1].distance) {
                                            loop_good_matches.push_back(m[0]);
                                        }
                                    }

                                    if (loop_good_matches.size() >= 20) 
                                    {
                                        std::vector<cv::Point3f> loop_obj_points;
                                        std::vector<cv::Point2f> loop_img_points;

                                        for (const auto& match : loop_good_matches) {
                                            cv::Point3f pt3d = candidate_kf.points_3d[match.trainIdx];
                                            if (pt3d.z > 0.0f) {
                                                loop_obj_points.push_back(pt3d);
                                                loop_img_points.push_back(current_frame.keypoints[match.queryIdx].pt);
                                            }
                                        }

                                        if (loop_obj_points.size() >= 30) {
                                            cv::Mat rvec_loop, tvec_loop;
                                            std::vector<int> loop_inliers;
                                            
                                            bool pnp_loop_success = cv::solvePnPRansac(
                                                loop_obj_points, loop_img_points, camera_matrix_, dist_coeffs_,
                                                rvec_loop, tvec_loop, false, 100, 2.0f, 0.99, loop_inliers, cv::SOLVEPNP_EPNP);

                                            if (pnp_loop_success && loop_inliers.size() >= 20) {
                                                cv::Mat R_loop;
                                                cv::Rodrigues(rvec_loop, R_loop);
                                                T_loop_relative = cv::Mat::eye(4, 4, CV_64F);
                                                R_loop.copyTo(T_loop_relative(cv::Rect(0, 0, 3, 3)));
                                                tvec_loop.copyTo(T_loop_relative(cv::Rect(3, 0, 1, 3)));
                                                
                                                loop_detected = true;
                                                loop_candidate_id = result.Id;
                                                RCLCPP_INFO(this->get_logger(), "!!! LOOP CLOSURE !!! Fechando ciclo entre KF %d e KF %d (Score: %.3f)", loop_candidate_id, keyframe_id_, result.Score);
                                                break; 
                                            }
                                        }
                                    }
                                }
                            }

                            if (loop_detected) 
                            {
                                Eigen::Matrix4d relative_loop_eigen;
                                cv::cv2eigen(T_loop_relative.inv(), relative_loop_eigen); 
                                gtsam::Pose3 loop_pose_opt(relative_loop_eigen);
                                
                                
                                gtsam::Pose3 loop_pose_base = T_base_opt_ * loop_pose_opt * T_base_opt_.inverse();

                                auto loop_noise = gtsam::noiseModel::Diagonal::Sigmas(
                                    (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished());

                                graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
                                    loop_candidate_id, keyframe_id_, loop_pose_base, loop_noise));
                            }
                            
                           
                            gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimates_);
                            gtsam::Values optimized_estimates = optimizer.optimize();
                            optimized_estimates_ = optimized_estimates;

                            gtsam::Pose3 corrected_pose = optimized_estimates_.at<gtsam::Pose3>(keyframe_id_);
                            Eigen::Matrix4d corrected_eigen = corrected_pose.matrix();
                            cv::eigen2cv(corrected_eigen, global_pose_);

                            last_keyframe_pose_ = global_pose_.clone();
                            
                            keyframe_database_[keyframe_id_] = current_frame; 
                            db_.add(current_frame.descriptors); 

                            publish_gtsam_data(corrected_pose);

                            keyframe_id_++;
                        }
                    
                    }
                }
            }
        }

       
        frame_count_++;
    }


};

int main(int argc, char * argv[]) 
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BowLoopNode>());
    rclcpp::shutdown();
    return 0;
}