#include "slam_core/DinoExtractor.hpp"
#include <iostream>
#include <chrono>

namespace slam_core 
{

DinoExtractor::DinoExtractor(const std::string& dino_onnx_path)
{
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(0); 
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.arena_extend_strategy = 0; 
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        ort_session_dino_ = std::make_unique<Ort::Session>(ort_env_, dino_onnx_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        dino_input_name_ = ort_session_dino_->GetInputNameAllocated(0, allocator).get();
        dino_output_name_ = ort_session_dino_->GetOutputNameAllocated(0, allocator).get();
    } 
    catch (const Ort::Exception& e) 
    {
        std::cerr << "[DinoExtractor ERROR] Falha ao carregar modelo DINO: " << e.what() << std::endl;
    }
}

std::vector<float> DinoExtractor::process_image_and_find_loop(const cv::Mat& image)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!ort_session_dino_ || image.empty()) 
    {
        return {};
    }
    const int DINO_WIDTH = 644;  
    const int DINO_HEIGHT = 476; 
    
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(DINO_WIDTH, DINO_HEIGHT), cv::Scalar(), true, false, CV_32F);
    
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std_dev = {0.229f, 0.224f, 0.225f};
    float* blob_data = (float*)blob.data;
    
    int channel_size = DINO_WIDTH * DINO_HEIGHT;
    for (int c = 0; c < 3; ++c) {
        float* channel_ptr = blob_data + c * channel_size;
        for (int i = 0; i < channel_size; ++i) {
            channel_ptr[i] = (channel_ptr[i] / 255.0f - mean[c]) / std_dev[c];
        }
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, DINO_HEIGHT, DINO_WIDTH}; 
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob_data, blob.total(), 
        input_shape.data(), input_shape.size()
    );

    const char* input_names[] = {dino_input_name_.c_str()};
    const char* output_names[] = {dino_output_name_.c_str()};

    auto output_tensors = ort_session_dino_->Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );

    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> current_vector(floatarr, floatarr + 384);

    return current_vector;
}

} // namespace slam_core