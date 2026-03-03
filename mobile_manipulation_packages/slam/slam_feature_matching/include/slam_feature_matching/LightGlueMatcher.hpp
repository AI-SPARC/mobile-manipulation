#ifndef SLAM_FEATURE_MATCHING_LIGHTGLUE_MATCHER_HPP
#define SLAM_FEATURE_MATCHING_LIGHTGLUE_MATCHER_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace slam_feature_matching 
{

class LightGlueMatcher 
{
public:
    LightGlueMatcher(const std::string& lightglue_onnx_path);
    ~LightGlueMatcher() = default;

    void compute_matches(const cv::Mat& img1, const cv::Mat& img2, 
                         std::vector<cv::Point2f>& kp1, std::vector<cv::Point2f>& kp2, 
                         std::vector<cv::DMatch>& matches);

private:
    Ort::Env ort_env_;
    std::unique_ptr<Ort::Session> ort_session_lightglue_;
    std::vector<std::string> lg_input_names_str_;
    std::vector<std::string> lg_output_names_str_;
    std::vector<const char*> lg_input_names_;
    std::vector<const char*> lg_output_names_;
};

} // namespace slam_feature_matching

#endif // SLAM_FEATURE_MATCHING_LIGHTGLUE_MATCHER_HPP