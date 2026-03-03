#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>

namespace slam_core 
{

class DinoLoop
{
public:
    DinoLoop(const std::string& dino_onnx_path, 
             const std::string& lightglue_onnx_path, 
             float similarity_threshold);
    
    ~DinoLoop();

    void compute_matches(const cv::Mat& img1, const cv::Mat& img2, 
                         std::vector<cv::Point2f>& kp1, std::vector<cv::Point2f>& kp2, 
                         std::vector<cv::DMatch>& matches);

    int process_image_and_find_loop(int current_kf_id, const cv::Mat& image);

private:
    void normalize_vector(std::vector<float>& v);

    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "DinoLoop"};
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
    float similarity_threshold_;
};

} // namespace slam_core