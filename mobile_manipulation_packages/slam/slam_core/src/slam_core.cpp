#include <memory>
#include <vector>
#include <iostream>
#include <chrono> 
#include <map>
#include <cmath> 
#include <thread>

#include <mutex>
#include <condition_variable>
#include <rclcpp/time.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
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
#include <image_transport/image_transport.hpp>

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
#include <gtsam/nonlinear/ISAM2.h>

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp" 
#include <opencv2/core/eigen.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "slam_interfaces/msg/gtsam_data.hpp"

#include <slam_core/Mapping.hpp>
#include <slam_core/ImuIntegration.hpp>
#include <slam_core/CameraIntegration.hpp>

#include "slam_feature_matching/DinoExtractor.hpp"
#include "slam_feature_matching/LightGlueMatcher.hpp"


class SlamCoreNode : public rclcpp::Node 
{
public:
    SlamCoreNode(
        std::shared_ptr<slam_core::Mapping> mapping_node,
        std::shared_ptr<slam_core::ImuIntegration> imu_integration_node,
        std::shared_ptr<slam_core::CameraIntegration> camera_integration_node,
        const rclcpp::NodeOptions & options = rclcpp::NodeOptions()
    ) : Node("slam_core_node", options) , 
        mapping_node_(mapping_node),
        imu_integration_node_(imu_integration_node),
        camera_integration_node_(camera_integration_node)
    {
        this->declare_parameter<std::string>("main_frame_id", "base_link");
        this->declare_parameter<bool>("use_imu", false);
        this->declare_parameter<int>("num_cameras", 1);
        
        this->declare_parameter<std::string>("robot_namespace", "robot_0"); 

        main_frame_id_ = this->get_parameter("main_frame_id").as_string();
        use_imu = this->get_parameter("use_imu").as_bool();
        num_cameras_ = this->get_parameter("num_cameras").as_int();


        std::string dino_path = "/home/momesso/pibic/src/mobile_manipulation_packages/slam/slam_core/onxx/dinov2_small.onnx";
        std::string lightglue_path = "/home/momesso/pibic/src/mobile_manipulation_packages/slam/slam_core/onxx/superpoint_lightglue_pipeline.onnx";
        

        dino_extractor_ = std::make_shared<slam_feature_matching::DinoExtractor>(dino_path);
        lightglue_matcher_ = std::make_shared<slam_feature_matching::LightGlueMatcher>(lightglue_path);


        std::string robot_ns = this->get_parameter("robot_namespace").as_string();
        std::string ns_prefix = "/" + robot_ns;

        {
            std::lock_guard<std::mutex> lock(frame_process_result_mutex);
            frame_process_result.resize(num_cameras_);
        }

        for (int i = 0; i < num_cameras_; ++i) 
        {
            camera_matrix_[i] = cv::Mat::eye(3, 3, CV_64F);
            dist_coeffs_[i] = cv::Mat::zeros(4, 1, CV_64F);
            global_pose_[i] = cv::Mat::eye(4, 4, CV_64F);
            last_keyframe_pose_[i] = cv::Mat::eye(4, 4, CV_64F);
            
            camera_info_received_[i] = false; 

            threads_.emplace_back(&SlamCoreNode::execute_visual_process, this, i);
        }

        rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();
        rclcpp::QoS default_qos(10);                     
       
       
        gt_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            ns_prefix + "/ground_truth", default_qos, std::bind(&SlamCoreNode::ground_truth_callback, this, std::placeholders::_1));
        
        current_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            ns_prefix + "/flann/current_image", sensor_qos);
            
        odometry_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            ns_prefix + "/flann/odometry_matches", sensor_qos);
        
        graph_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            ns_prefix + "/slam/factor_graph", default_qos);
            
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            ns_prefix + "/slam/odom", default_qos);
            
        graph_markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            ns_prefix + "/slam/graph_markers", default_qos);
            
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            ns_prefix + "/slam/trajectory_path", default_qos);

        factor_pub_ = this->create_publisher<slam_interfaces::msg::GtsamData>(
            ns_prefix + "/slam/camera_factors", 10);

        image_transport_pub_ = image_transport::create_publisher(this, ns_prefix + "/loop_closure/dino_image");
        depth_image_transport_pub_ = image_transport::create_publisher(this, ns_prefix + "/loop_closure/depth_image");
       
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        orb_ = cv::ORB::create(1000);
        local_matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        last_processed_time_ = this->now();

        gtsam_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(25), 
            std::bind(&SlamCoreNode::process_gtsam, this)
        );

        RCLCPP_INFO(this->get_logger(), "--- NO DE ODOMETRIA VISUAL INICIADO PARA O ROBO: %s ---", robot_ns.c_str());
    }

    ~SlamCoreNode() {
        for (auto& t : threads_) 
        {
            if (t.joinable()) {
                t.join(); 
            }
        }
    }

private:
  
    struct FrameData 
    { 
        int id; 
        cv::Mat image; 
        cv::Mat depth_image;
        std::string rgb_frame;
        std::string depth_frame;
        gtsam::Pose3 global_pose;
        
    };

    struct FrameProcessResult
    {
        gtsam::Pose3 delta_base;
        gtsam::SharedNoiseModel visual_noise;
        gtsam::Pose3 estimate;
        int target_keyframe_id = 0;
        bool is_new = false;
        std_msgs::msg::Header header;
        std::vector<float> signature;
        sensor_msgs::msg::Image::SharedPtr image;
        sensor_msgs::msg::Image::SharedPtr depth_image;
    };

    struct LoopFactorData
    {
        int from_id;
        int to_id;
        gtsam::Pose3 delta_loop;
        gtsam::SharedNoiseModel loop_noise;
    };

    std::vector<LoopFactorData> pending_loop_factors_;


    bool loop_detected = false;
    bool first_gt_received_ = false;
    bool has_gt_ = false;
    bool has_keyframe_ = false;
    bool is_moving_ = false;
    bool tf_received_ = false;
    bool tracking_success = false;   
    bool use_imu = false;
    
    double total_gt_distance_ = 0.0;
    
    int frame_count_ = 0;
    int keyframe_id_ = 0;
    int tracking_lost_counter_ = 0;

    std::mutex sync_process_mutex_; 
    
    rclcpp::Time current_target_timestamp_;

    int num_cameras_ = 1; 
    int cameras_ready_for_init_ = 0;
    std::condition_variable init_cv_;

    
    std::unordered_map<int, FrameData> last_keyframe_;
    std::unordered_map<int, rclcpp::Time> last_processed_timestamp_;
  
    gtsam::ISAM2 isam2_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Pose3 initial_gt_pose_;
    gtsam::Pose3 latest_gt_pose_;
    gtsam::Pose3 previous_gt_pose_;
    gtsam::Pose3 T_base_cam_;
    gtsam::Pose3 T_base_opt_;
    gtsam::Values initial_estimates_;
    gtsam::Values optimized_estimates_;
    gtsam::imuBias::ConstantBias current_bias_;

    std::unordered_map<int, cv::Mat> camera_matrix_;
    std::unordered_map<int, cv::Mat> dist_coeffs_;
    std::unordered_map<int, cv::Mat> global_pose_;
    std::unordered_map<int, cv::Mat> last_keyframe_pose_;
    std::unordered_map<int, bool> camera_info_received_;

    cv::Ptr<cv::BFMatcher> local_matcher_; 
    cv::Ptr<cv::ORB> orb_;

    image_transport::Publisher depth_image_transport_pub_;
    image_transport::Publisher image_transport_pub_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr current_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr odometry_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_markers_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_pub_;
    rclcpp::Publisher<slam_interfaces::msg::GtsamData>::SharedPtr factor_pub_;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gt_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_; 

    rclcpp::Time last_processed_time_;
    rclcpp::TimerBase::SharedPtr gtsam_timer_;
    
    std::map<int, std::map<int, FrameData>> keyframe_database_;
    std::mutex keyframe_lock;
    std::string main_frame_id_;
    std::unordered_map<int, std::shared_ptr<std::vector<cv::Point3f>>> dense_clouds_database_;
    std::vector<FrameData> history_frames_;
    std::vector<FrameProcessResult> frame_process_result;
    std::vector<std::thread> threads_;

  
    std::shared_ptr<slam_feature_matching::DinoExtractor> dino_extractor_;
    std::shared_ptr<slam_feature_matching::LightGlueMatcher> lightglue_matcher_;
        
    std::shared_ptr<slam_core::Mapping> mapping_node_;
    std::shared_ptr<slam_core::ImuIntegration> imu_integration_node_;
    std::shared_ptr<slam_core::CameraIntegration> camera_integration_node_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::mutex compute_mutex;
    std::mutex frame_process_result_mutex;


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

    void execute_visual_process(int camera_id) 
    {
        while (rclcpp::ok()) 
        {
            slam_core::CameraData cam_data;
            bool found = false;

            {
                std::unique_lock<std::mutex> lock(sync_process_mutex_); 

                if (camera_id == 0) 
                {
                    found = camera_integration_node_->get_latest_frame(camera_id, cam_data);
                    
                } 
                else 
                {
                    if (current_target_timestamp_.nanoseconds() == 0 || 
                        current_target_timestamp_.nanoseconds() == last_processed_timestamp_[camera_id].nanoseconds()) 
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(5)); 
                        continue; 
                    }
                    
                    found = camera_integration_node_->retrieve_frame_by_timestamp(
                        camera_id, current_target_timestamp_, cam_data, 0.01);
                }
            } 

            if (!found) 
            {
               
                if (camera_id != 0 && current_target_timestamp_.nanoseconds() != 0) 
                {
                    RCLCPP_WARN(this->get_logger(), "[Cam %d] Imagem do TS %.3f s perdida no buffer! Ignorando este Keyframe.", camera_id, current_target_timestamp_.seconds());
                    
                    
                    last_processed_timestamp_[camera_id] = current_target_timestamp_;
                } 
                else 
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                continue; 
            }
            

          
            auto current_time = this->now();
            last_processed_time_ = current_time;

            FrameData current_frame;
            current_frame.id = frame_count_;
            current_frame.rgb_frame = cam_data.rgb->header.frame_id;
            current_frame.depth_frame = cam_data.depth->header.frame_id;
            try 
            {
                current_frame.image = cv_bridge::toCvCopy(cam_data.rgb, sensor_msgs::image_encodings::BGR8)->image.clone();
                current_frame.depth_image = cv_bridge::toCvCopy(cam_data.depth, cam_data.depth->encoding)->image.clone();
            } 
            catch (cv_bridge::Exception& e) 
            {
                continue; 
            }   
            if (camera_info_received_.find(camera_id) == camera_info_received_.end() || !camera_info_received_[camera_id]) 
            {
                auto info_msg = cam_data.info; 
                double fx = info_msg->k[0]; double cx = info_msg->k[2];
                double fy = info_msg->k[4]; double cy = info_msg->k[5];
                camera_matrix_[camera_id] = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
                
                if (!info_msg->d.empty()) 
                {
                    dist_coeffs_[camera_id] = cv::Mat(info_msg->d.size(), 1, CV_64F);
                    for (size_t i = 0; i < info_msg->d.size(); ++i) dist_coeffs_[camera_id].at<double>(i) = info_msg->d[i];
                } 
                else 
                {
                    dist_coeffs_[camera_id] = cv::Mat::zeros(4, 1, CV_64F);
                }
                
                if (camera_id == 0) mapping_node_->set_camera_info(info_msg, main_frame_id_, 1000.0);
                camera_info_received_[camera_id] = true;
            }

            auto msg_copy = std::make_shared<sensor_msgs::msg::Image>(*cam_data.rgb);
            auto depth_msg_copy = std::make_shared<sensor_msgs::msg::Image>(*cam_data.depth);

            if (!tf_received_) 
            {
                if (main_frame_id_ == current_frame.rgb_frame) 
                {
                    T_base_opt_ = gtsam::Pose3(); tf_received_ = true;
                } 
                else 
                {
                    try 
                    {
                        geometry_msgs::msg::TransformStamped transform_stamped = tf_buffer_->lookupTransform(
                            main_frame_id_, current_frame.rgb_frame, tf2::TimePointZero);
                        Eigen::Quaterniond q(transform_stamped.transform.rotation.w, transform_stamped.transform.rotation.x,
                                            transform_stamped.transform.rotation.y, transform_stamped.transform.rotation.z);
                        Eigen::Vector3d t(transform_stamped.transform.translation.x, transform_stamped.transform.translation.y,
                                        transform_stamped.transform.translation.z);
                        T_base_opt_ = gtsam::Pose3(gtsam::Rot3(q.toRotationMatrix()), gtsam::Point3(t));
                        tf_received_ = true;
                    } 
                    catch (const tf2::TransformException & ex) 
                    { 
                        RCLCPP_WARN(this->get_logger(), "[Cam %d] Falha no TF: %s", camera_id, ex.what());
                        continue; 
                    }
                }
            }
           

            
            if (last_keyframe_.find(camera_id) == last_keyframe_.end()) 
            {
                last_keyframe_[camera_id] = current_frame; 
                global_pose_[camera_id] = cv::Mat::eye(4, 4, CV_64F);
                last_keyframe_pose_[camera_id] = cv::Mat::eye(4, 4, CV_64F);

                if (camera_id == 0) 
                {
                    keyframe_id_ = 0; 

                    // msg_copy->header.frame_id = std::to_string(keyframe_id_);
                            
                            
                    int current_kf_id;
                    try {
                        current_kf_id = std::stoi(msg_copy->header.frame_id);
                    } 
                    catch (...) 
                    {
                        RCLCPP_ERROR(this->get_logger(), "Erro ao converter frame_id para int!");
                        current_kf_id = -1; 
                    }

                    
                    cv_bridge::CvImagePtr cv_ptr;
                    try 
                    {
                        cv_ptr = cv_bridge::toCvCopy(msg_copy, sensor_msgs::image_encodings::RGB8);
                    } 
                    catch (cv_bridge::Exception& e) 
                    {
                        RCLCPP_ERROR(this->get_logger(), "Erro no cv_bridge: %s", e.what());
                        
                    }

                    std::vector<float> signature;
                    {
                        std::lock_guard<std::mutex> lock(compute_mutex);

                        if (cv_ptr) 
                        {
                            signature = dino_extractor_->process_image_and_find_loop(cv_ptr->image);
                        }
                    }

                    
                    slam_interfaces::msg::GtsamData init_msg;
                    init_msg.keyframe = keyframe_id_;

                    
                    init_msg.delta_base.pose.position.x = 0.0;
                    init_msg.delta_base.pose.position.y = 0.0;
                    init_msg.delta_base.pose.position.z = 0.0;
                    init_msg.estimate.position.x = 0.0;
                    init_msg.estimate.position.y = 0.0;
                    init_msg.estimate.position.z = 0.0;

                   
                    init_msg.delta_base.pose.orientation.x = 0.0;
                    init_msg.delta_base.pose.orientation.y = 0.0;
                    init_msg.delta_base.pose.orientation.z = 0.0;
                    init_msg.delta_base.pose.orientation.w = 1.0;
                    init_msg.estimate.orientation.x = 0.0;
                    init_msg.estimate.orientation.y = 0.0;
                    init_msg.estimate.orientation.z = 0.0;
                    init_msg.estimate.orientation.w = 1.0;
                    init_msg.signature = signature;

                    for (int r = 0; r < 6; ++r) 
                    {
                        for (int c = 0; c < 6; ++c) 
                        {
                            if (r == c) init_msg.delta_base.covariance[r * 6 + c] = 1e-6;
                            else        init_msg.delta_base.covariance[r * 6 + c] = 0.0;
                        }
                    }

                    init_msg.header.stamp = cam_data.stamp;
                    msg_copy->header.stamp = cam_data.stamp;
                    depth_msg_copy->header.stamp = cam_data.stamp;

                    if (factor_pub_) 
                    {
                        factor_pub_->publish(init_msg);
                        RCLCPP_INFO(this->get_logger(), "[Cam 0] Fator de inicializacao enviado para o no GTSAM.");
                    }
                   
                    image_transport_pub_.publish(msg_copy);
                    depth_image_transport_pub_.publish(depth_msg_copy);
                   
                    
                    
                    current_target_timestamp_ = cam_data.stamp; 
                    has_keyframe_ = true; 
                }

                last_processed_timestamp_[camera_id] = cam_data.stamp;
                continue; 
            }

            compute_translation_and_rotation(camera_id, current_frame, cam_data.stamp, msg_copy, depth_msg_copy);
            
            if (camera_id == 0) frame_count_++;
        }
    }

    
    void compute_translation_and_rotation(int camera_id, FrameData& current_frame, rclcpp::Time current_stamp, 
        std::shared_ptr<sensor_msgs::msg::Image> msg_copy, std::shared_ptr<sensor_msgs::msg::Image> depth_msg_copy)
    {
    

        sensor_msgs::msg::Image::ConstSharedPtr rgb_msg;
        sensor_msgs::msg::Image::ConstSharedPtr depth_msg;

        auto current_time = this->now(); 
        last_processed_time_ = current_time;
        
        double fx = camera_matrix_[camera_id].at<double>(0, 0);
        double fy = camera_matrix_[camera_id].at<double>(1, 1);
        double cx = camera_matrix_[camera_id].at<double>(0, 2);
        double cy = camera_matrix_[camera_id].at<double>(1, 2);

        std::vector<cv::Point2f> kp1, kp2;
        std::vector<cv::DMatch> matches;
        
        {
            std::lock_guard<std::mutex> lock(compute_mutex);
            lightglue_matcher_->compute_matches(current_frame.image, last_keyframe_[camera_id].image, kp1, kp2, matches);
        }

        if (!matches.empty())
        {
            cv::Mat debug_image;
            cv::hconcat(current_frame.image, last_keyframe_[camera_id].image, debug_image);
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
            match_header.stamp = current_time; 
            match_header.frame_id = current_frame.rgb_frame; 
            odometry_pub_->publish(*cv_bridge::CvImage(match_header, "bgr8", debug_image).toImageMsg());
        }
        
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
            if (cv::norm(dist_coeffs_[camera_id]) > 0.0001) 
            {
                cv::undistortPoints(train_pts, train_pts_undist, camera_matrix_[camera_id], dist_coeffs_[camera_id], cv::noArray(), camera_matrix_[camera_id]);
                cv::undistortPoints(query_pts, query_pts_undist, camera_matrix_[camera_id], dist_coeffs_[camera_id], cv::noArray(), camera_matrix_[camera_id]);
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

                float z_center = get_robust_depth(last_keyframe_[camera_id].depth_image, pt2d_train.x, pt2d_train.y);
                
                if (z_center <= 0.1f || z_center > 7.0) 
                {
                    edge_filter_rejected++; 
                    continue; 
                }

                float min_z = z_center, max_z = z_center;
                for (int dy = -1; dy <= 1; ++dy) 
                {
                    for (int dx = -1; dx <= 1; ++dx) 
                    {
                        if (dx == 0 && dy == 0) continue; 
                        
                        float z_neighbor = get_robust_depth(last_keyframe_[camera_id].depth_image, pt2d_train.x + dx, pt2d_train.y + dy);
                        
                        if (z_neighbor > 0.1f) 
                        {
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
                cv::cv2eigen(global_pose_[camera_id], global_pose_eigen);
                cv::cv2eigen(last_keyframe_pose_[camera_id], last_kf_pose_eigen);

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
                    camera_matrix_[camera_id],     
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
                        if (camera_id != 0) {
                            last_keyframe_[camera_id] = current_frame;
                            last_processed_timestamp_[camera_id] = current_stamp;
                        }
                        return;
                    }

                    tracking_success = true; 
                    Eigen::Matrix4d delta_base_eigen = delta_base.matrix();
                    cv::Mat delta_base_cv;
                    cv::eigen2cv(delta_base_eigen, delta_base_cv);
                    
                    global_pose_[camera_id] = last_keyframe_pose_[camera_id] * delta_base_cv;

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

                    Eigen::Matrix4d global_pose_eigen_out;
                    cv::cv2eigen(global_pose_[camera_id], global_pose_eigen_out);
                    gtsam::Pose3 current_global_pose(global_pose_eigen_out);

                    auto visual_noise = gtsam::noiseModel::Gaussian::Covariance(cov_eigen);

                    
                    if (camera_id == 0) 
                    {
                        if (translation_dist > 0.15 || rotation_dist > 0.1) 
                        {
                            keyframe_id_++; 

                            // msg_copy->header.frame_id = std::to_string(keyframe_id_);
                            
                            
                            int current_kf_id;
                            try {
                                current_kf_id = std::stoi(msg_copy->header.frame_id);
                            } 
                            catch (...) 
                            {
                                RCLCPP_ERROR(this->get_logger(), "Erro ao converter frame_id para int!");
                                current_kf_id = -1; 
                            }

                            
                            cv_bridge::CvImagePtr cv_ptr;
                            try 
                            {
                                cv_ptr = cv_bridge::toCvCopy(msg_copy, sensor_msgs::image_encodings::RGB8);
                            } 
                            catch (cv_bridge::Exception& e) 
                            {
                                RCLCPP_ERROR(this->get_logger(), "Erro no cv_bridge: %s", e.what());
                                
                            }

                            std::vector<float> signature;
                            {
                                std::lock_guard<std::mutex> lock(compute_mutex);

                                if (cv_ptr) 
                                {
                                    signature = dino_extractor_->process_image_and_find_loop(cv_ptr->image);
                                }
                            }


                           
                           
                            {
                                std::lock_guard<std::mutex> lock(frame_process_result_mutex);
                                
                                frame_process_result[camera_id].delta_base = delta_base;
                                frame_process_result[camera_id].visual_noise = visual_noise;
                                frame_process_result[camera_id].estimate = current_global_pose;
                                frame_process_result[camera_id].target_keyframe_id = keyframe_id_; 
                                frame_process_result[camera_id].is_new = true;                            
                                frame_process_result[camera_id].header = msg_copy->header; 
                                frame_process_result[camera_id].signature = signature;
                                frame_process_result[camera_id].image = msg_copy;
                                frame_process_result[camera_id].depth_image = depth_msg_copy;
                            }

                            last_keyframe_pose_[camera_id] = global_pose_[camera_id].clone();
                            last_keyframe_[camera_id] = current_frame;
                            keyframe_database_[camera_id][keyframe_id_] = current_frame;

                            current_target_timestamp_ = current_stamp;
                            last_processed_timestamp_[camera_id] = current_stamp;
                        }
                    }
                    else 
                    {
                        
                        // msg_copy->header.frame_id = std::to_string(keyframe_id_);
                            
                            
                        int current_kf_id;
                        try {
                            current_kf_id = std::stoi(msg_copy->header.frame_id);
                        } 
                        catch (...) 
                        {
                            RCLCPP_ERROR(this->get_logger(), "Erro ao converter frame_id para int!");
                            current_kf_id = -1; 
                        }

                        
                        cv_bridge::CvImagePtr cv_ptr;
                        try 
                        {
                            cv_ptr = cv_bridge::toCvCopy(msg_copy, sensor_msgs::image_encodings::RGB8);
                        } 
                        catch (cv_bridge::Exception& e) 
                        {
                            RCLCPP_ERROR(this->get_logger(), "Erro no cv_bridge: %s", e.what());
                            
                        }

                        std::vector<float> signature;
                        {
                            std::lock_guard<std::mutex> lock(compute_mutex);

                            if (cv_ptr) 
                            {
                                signature = dino_extractor_->process_image_and_find_loop(cv_ptr->image);
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lock(frame_process_result_mutex);
                            frame_process_result[camera_id].delta_base = delta_base;
                            frame_process_result[camera_id].visual_noise = visual_noise;
                            frame_process_result[camera_id].estimate = current_global_pose;
                            frame_process_result[camera_id].target_keyframe_id = keyframe_id_; 
                            frame_process_result[camera_id].is_new = true;
                            frame_process_result[camera_id].header = msg_copy->header; 
                            frame_process_result[camera_id].signature = signature;
                            frame_process_result[camera_id].image = msg_copy;
                            frame_process_result[camera_id].depth_image = depth_msg_copy;
                        }

                        last_keyframe_pose_[camera_id] = global_pose_[camera_id].clone();
                        last_keyframe_[camera_id] = current_frame;
                        keyframe_database_[camera_id][keyframe_id_] = current_frame;
                        
                        last_processed_timestamp_[camera_id] = current_stamp;
                    }
                }
                else 
                {
                    tracking_lost_counter_++;
                    RCLCPP_WARN(this->get_logger(), 
                        "[Rastreamento Perdido] PnP RANSAC falhou ou não obteve consenso. "
                        "Apenas %zu inliers de 8 necessarios.", inliers.size());
                    if (camera_id != 0) { last_processed_timestamp_[camera_id] = current_stamp; last_keyframe_[camera_id] = current_frame; }
                    return;
                }   
            }
            else 
            {
                tracking_lost_counter_++;
                RCLCPP_WARN(this->get_logger(), 
                "[Rastreamento Perdido] Pontos 3D validos insuficientes (%zu/8 minimos).", object_pts_3d.size());
                if (camera_id != 0) { last_processed_timestamp_[camera_id] = current_stamp; last_keyframe_[camera_id] = current_frame; }
                return;
            }
        }
    }



    void process_gtsam()
    {
        if (!has_keyframe_ || keyframe_id_ == 0) return; 

        std::lock_guard<std::mutex> lock(frame_process_result_mutex);

        for(int i = 0; i < frame_process_result.size(); i++)
        {
            if (frame_process_result[i].is_new) 
            {
                int kf_id = frame_process_result[i].target_keyframe_id;

                if (kf_id <= 0) 
                {
                    frame_process_result[i].is_new = false;
                    continue; 
                }

                slam_interfaces::msg::GtsamData msg;

                msg.header = frame_process_result[i].header;

                msg.keyframe = kf_id;

                gtsam::Point3 t = frame_process_result[i].delta_base.translation();
                gtsam::Quaternion q = frame_process_result[i].delta_base.rotation().toQuaternion();

                msg.delta_base.pose.position.x = t.x();
                msg.delta_base.pose.position.y = t.y();
                msg.delta_base.pose.position.z = t.z();
                msg.delta_base.pose.orientation.x = q.x();
                msg.delta_base.pose.orientation.y = q.y();
                msg.delta_base.pose.orientation.z = q.z();
                msg.delta_base.pose.orientation.w = q.w();

                auto gaussian_noise = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(frame_process_result[i].visual_noise);
                if (gaussian_noise) 
                {
                    Eigen::MatrixXd cov_matrix = gaussian_noise->covariance();
                    for (int r = 0; r < 6; ++r) 
                    {
                        for (int c = 0; c < 6; ++c) 
                        {
                            msg.delta_base.covariance[r * 6 + c] = cov_matrix(r, c);
                        }
                    }
                }

                gtsam::Point3 est_t = frame_process_result[i].estimate.translation();
                gtsam::Quaternion est_q = frame_process_result[i].estimate.rotation().toQuaternion();

                msg.estimate.position.x = est_t.x();
                msg.estimate.position.y = est_t.y();
                msg.estimate.position.z = est_t.z();
                msg.estimate.orientation.x = est_q.x();
                msg.estimate.orientation.y = est_q.y();
                msg.estimate.orientation.z = est_q.z();
                msg.estimate.orientation.w = est_q.w();

                msg.signature = frame_process_result[i].signature;

                if (factor_pub_)
                {
                    factor_pub_->publish(msg);
                }

                if (frame_process_result[i].image != nullptr) 
                {
                    frame_process_result[i].image->header = frame_process_result[i].header;
        
                    image_transport_pub_.publish(frame_process_result[i].image);
                    
                    RCLCPP_INFO(this->get_logger(), "Imagem da câmera %d publicada com sucesso!", i);
                }

                if (frame_process_result[i].depth_image != nullptr) 
                {
                    frame_process_result[i].depth_image->header = frame_process_result[i].header;
        
                    depth_image_transport_pub_.publish(frame_process_result[i].depth_image);
                    
                    RCLCPP_INFO(this->get_logger(), "Imagem Depth da câmera %d publicada com sucesso!", i);
                }

                RCLCPP_INFO(this->get_logger(), "[Pub] Fator enviado: Camera %d -> KF %d", i, kf_id);

                frame_process_result[i].is_new = false; 
            }
        }
    }

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions global_options;

    std::string current_robot_ns = "robot_0";
    int current_num_cameras = 1;

    
    {
        auto temp_node = std::make_shared<rclcpp::Node>("slam_core", global_options);
        current_robot_ns = temp_node->declare_parameter<std::string>("robot_namespace", "robot_0");
        current_num_cameras = temp_node->declare_parameter<int>("num_cameras", 1);
    } 


    rclcpp::NodeOptions mapping_opts;
    mapping_opts.arguments({"--ros-args", "-r", "__node:=mapping_node", "-p", "use_sim_time:=true"});
    mapping_opts.parameter_overrides({
        {"robot_namespace", current_robot_ns}
    });

    rclcpp::NodeOptions imu_opts;
    imu_opts.arguments({"--ros-args", "-r", "__node:=imu_integration_node", "-p", "use_sim_time:=true"});
    imu_opts.parameter_overrides({
        {"robot_namespace", current_robot_ns}
    });

    rclcpp::NodeOptions camera_opts;
    camera_opts.arguments({"--ros-args", "-r", "__node:=camera_integration_node", "-p", "use_sim_time:=true"});
    camera_opts.parameter_overrides({
        {"robot_namespace", current_robot_ns},
        {"num_cameras", current_num_cameras}
    });

    
    auto mapping_node = std::make_shared<slam_core::Mapping>(mapping_opts);
    auto imu_integration_node = std::make_shared<slam_core::ImuIntegration>(imu_opts);
    auto camera_integration_node = std::make_shared<slam_core::CameraIntegration>(camera_opts);

 
    auto server_node = std::make_shared<SlamCoreNode>(
        mapping_node, 
        imu_integration_node, 
        camera_integration_node,
        global_options 
    );

  
    rclcpp::executors::MultiThreadedExecutor executor;
    
    executor.add_node(mapping_node);
    executor.add_node(imu_integration_node);
    executor.add_node(camera_integration_node);
    executor.add_node(server_node);
    
    executor.spin();

    rclcpp::shutdown();
    return 0;
}