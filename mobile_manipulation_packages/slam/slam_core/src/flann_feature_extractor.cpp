#include <memory>
#include <vector>
#include <iostream>
#include <chrono> 
#include <map>
#include <cmath> 

// ROS 2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp" 
#include "geometry_msgs/msg/twist.hpp"
#include "cv_bridge/cv_bridge.hpp"

// OpenCV
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp" 

// DBoW3
#include "DBoW3/DBoW3.h"

struct FrameData {
    int frame_id;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point3f> points_3d; 
};

class BowLoopNode : public rclcpp::Node {
public:
    BowLoopNode() : Node("bow_loop_node") {
        std::string vocab_path = "/home/momesso/ORBvoc.txt"; 
        
        RCLCPP_INFO(this->get_logger(), "Carregando DBoW3...");
        voc_.load(vocab_path);
        
        if (voc_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "FALHA CRÍTICA: Vocabulário não encontrado.");
            throw std::runtime_error("Vocabulario ausente.");
        }   
        RCLCPP_INFO(this->get_logger(), "Vocabulário carregado! Palavras: %d", (int)voc_.size());

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
        
        // Mantive o publisher antigo, mas agora ele vai publicar os matches da odometria
        odometry_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/flann/odometry_matches", 10);

        orb_ = cv::ORB::create(1000);
        local_matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);

        camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
        dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);

        global_pose_ = cv::Mat::eye(4, 4, CV_64F);

        last_processed_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "--- NÓ DE ODOMETRIA VISUAL INICIADO ---");
        RCLCPP_INFO(this->get_logger(), "Aguardando tópico /camera/depth/camera_info...");
    }

private:
    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        is_moving_ = (std::abs(msg->linear.x) > 0.01 || std::abs(msg->angular.z) > 0.01);
    }

    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            last_depth_msg_ = cv_bridge::toCvCopy(msg, msg->encoding);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Erro na profundidade: %s", e.what());
        }
    }

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (camera_info_received_) return; 

        double fx = msg->k[0];
        double cx = msg->k[2];
        double fy = msg->k[4];
        double cy = msg->k[5];

        camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

        if (!msg->d.empty()) {
            dist_coeffs_ = cv::Mat(msg->d.size(), 1, CV_64F);
            for (size_t i = 0; i < msg->d.size(); ++i) {
                dist_coeffs_.at<double>(i) = msg->d[i];
            }
        }

        camera_info_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Matriz da Câmera Carregada! fx:%.1f, fy:%.1f, cx:%.1f, cy:%.1f", fx, fy, cx, cy);
    }

    float get_depth_meters(const cv::Mat& depth_img, int x, int y) {
        if (x < 0 || x >= depth_img.cols || y < 0 || y >= depth_img.rows) return 0.0f;

        if (depth_img.type() == CV_32FC1) {
            return depth_img.at<float>(y, x); 
        } else if (depth_img.type() == CV_16UC1) {
            return depth_img.at<uint16_t>(y, x) * 0.001f; 
        }
        return 0.0f;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        if (!camera_info_received_ || !is_moving_ || !last_depth_msg_) return;

        auto now = this->now();
        // if ((now - last_processed_time_).seconds() < 0.02) 
        // {
        //     return;
        // }

        last_processed_time_ = now;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) { return; }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb_->detectAndCompute(cv_ptr->image, cv::noArray(), keypoints, descriptors);

        if (descriptors.empty()) return;

        std::vector<cv::Point3f> points_3d;
        double fx = camera_matrix_.at<double>(0, 0);
        double fy = camera_matrix_.at<double>(1, 1);
        double cx = camera_matrix_.at<double>(0, 2);
        double cy = camera_matrix_.at<double>(1, 2);

        for (const auto& kp : keypoints) 
        {
            float z = get_depth_meters(last_depth_msg_->image, (int)kp.pt.x, (int)kp.pt.y);

            if (z > 0.1f && z < 4.0f) 
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
            last_keyframe_pose_ = cv::Mat::eye(4, 4, CV_64F); 
            global_pose_ = cv::Mat::eye(4, 4, CV_64F);
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
                    rvec, tvec, false, 100, 3.0f, 0.99, inliers);

                
                if (pnp_success && inliers.size() >= 15) 
                {
                    
                    double translation_dist = cv::norm(tvec);
                    double rotation_dist = cv::norm(rvec);

                   
                    if (translation_dist > 1.0) 
                    {
                        RCLCPP_WARN(this->get_logger(), "!!! EXPLOSÃO EVITADA !!! Pulo irreal de %.2f metros.", translation_dist);
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
                        global_pose_ = last_keyframe_pose_ * T_kf_curr;

                        double px = global_pose_.at<double>(0, 3);
                        double py = global_pose_.at<double>(1, 3);
                        double pz = global_pose_.at<double>(2, 3);

                        RCLCPP_INFO(this->get_logger(), "Pose: [X: %7.3f | Y: %7.3f | Z: %7.3f]", px, py, pz);

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

                        
                        double base_var_trans = 0.001; 
                        double base_var_rot   = 0.001; 

                        
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

                        
                        RCLCPP_INFO(this->get_logger(), 
                            "Incerteza (Covariância) -> Pos: [X: %.6f | Y: %.6f | Z: %.6f]  Rot: [R: %.6f | P: %.6f | Y: %.6f]", 
                            var_x, var_y, var_z, var_roll, var_pitch, var_yaw);

                        if (translation_dist > 0.2 || rotation_dist > 0.125 || inliers.size() < 20) 
                        {
                            last_keyframe_ = current_frame;


                            last_keyframe_pose_ = global_pose_.clone();
                            RCLCPP_INFO(this->get_logger(), "+++ NOVO KEYFRAME GERADO +++");
                        }
                    }
                }
            }
        }

       
        if (!tracking_success) 
        {
            RCLCPP_WARN(this->get_logger(), "!!! PERDA DE RASTREIO !!! Forçando Novo Keyframe para recuperar visão.");
      
            last_keyframe_ = current_frame;
            
            last_keyframe_pose_ = global_pose_.clone(); 
        }

        db_.add(descriptors); 
        frame_count_++;
    }

    cv::Mat last_keyframe_pose_;
    FrameData last_keyframe_;
    bool has_keyframe_ = false;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_; 
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_; 
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr current_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr odometry_pub_;
    
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::BFMatcher> local_matcher_; 
    
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv_bridge::CvImagePtr last_depth_msg_; 
    bool camera_info_received_ = false; 

    // Variável que guarda o seu (0,0,0) inicial e acumula o deslocamento
    cv::Mat global_pose_;

    DBoW3::Vocabulary voc_;
    DBoW3::Database db_;
    std::vector<FrameData> history_frames_;
    
    int frame_count_ = 0;
    bool is_moving_ = false;
    rclcpp::Time last_processed_time_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BowLoopNode>());
    rclcpp::shutdown();
    return 0;
}