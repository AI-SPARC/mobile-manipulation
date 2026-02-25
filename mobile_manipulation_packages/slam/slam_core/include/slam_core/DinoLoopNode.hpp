#ifndef SLAM_COMPONENTS__DINO_LOOP_NODE_HPP_
#define SLAM_COMPONENTS__DINO_LOOP_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>

#include <vector>
#include <string>
#include <memory>

namespace slam_core 
{

class DinoLoopNode : public rclcpp::Node 
{
public:
    explicit DinoLoopNode(const rclcpp::NodeOptions & options);
    ~DinoLoopNode(); 

    int keyframe_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    
    void compute_matches(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Point2f>& kp1, std::vector<cv::Point2f>& kp2, std::vector<cv::DMatch>& matches);

private:
    void normalize_vector(std::vector<float>& v);

    float similarity_threshold_;
    int min_frame_separation_;
    
    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "DinoLoopNode"};
    
    std::unique_ptr<Ort::Session> ort_session_dino_;
    std::unique_ptr<Ort::Session> ort_session_lightglue_;
    
    std::string dino_input_name_;
    std::string dino_output_name_;

    std::vector<std::string> lg_input_names_str_;
    std::vector<std::string> lg_output_names_str_;
    std::vector<const char*> lg_input_names_;
    std::vector<const char*> lg_output_names_;
    
    faiss::IndexFlatIP* inner_index_; 
    faiss::IndexIDMap* faiss_index_;  
};

} 

#endif