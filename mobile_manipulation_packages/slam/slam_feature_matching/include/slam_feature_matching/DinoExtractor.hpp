#ifndef SLAM_CORE_DINO_EXTRACTOR_HPP
#define SLAM_CORE_DINO_EXTRACTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace slam_core 
{

class DinoExtractor 
{
public:
    DinoExtractor(const std::string& dino_onnx_path);
    ~DinoExtractor() = default;

    std::vector<float> process_image_and_find_loop(const cv::Mat& image);

private:
    Ort::Env ort_env_;
    std::unique_ptr<Ort::Session> ort_session_dino_;
    std::string dino_input_name_;
    std::string dino_output_name_;
};

} // namespace slam_core

#endif // SLAM_CORE_DINO_EXTRACTOR_HPP