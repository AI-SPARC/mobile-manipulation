#include "slam_feature_matching/LightGlueMatcher.hpp"
#include <iostream>
#include <chrono>

namespace slam_feature_matching 
{

LightGlueMatcher::LightGlueMatcher(const std::string& lightglue_onnx_path)
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

        ort_session_lightglue_ = std::make_unique<Ort::Session>(ort_env_, lightglue_onnx_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < ort_session_lightglue_->GetInputCount(); i++) 
        {
            lg_input_names_str_.push_back(ort_session_lightglue_->GetInputNameAllocated(i, allocator).get());
        }

        for (size_t i = 0; i < ort_session_lightglue_->GetOutputCount(); i++) 
        {
            lg_output_names_str_.push_back(ort_session_lightglue_->GetOutputNameAllocated(i, allocator).get());
        }

        for (const auto& str : lg_input_names_str_) lg_input_names_.push_back(str.c_str());
        for (const auto& str : lg_output_names_str_) lg_output_names_.push_back(str.c_str());

    } 
    catch (const Ort::Exception& e) 
    {
        std::cerr << "[LightGlueMatcher ERROR] Falha ao carregar modelo LightGlue: " << e.what() << std::endl;
    }
}

void LightGlueMatcher::compute_matches(const cv::Mat& img1, const cv::Mat& img2, 
                                       std::vector<cv::Point2f>& kp1, std::vector<cv::Point2f>& kp2, 
                                       std::vector<cv::DMatch>& matches)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    kp1.clear(); kp2.clear(); matches.clear();
    
    if (img1.empty() || img2.empty()) 
    {
        return;
    }

    cv::Mat mean1, stddev1, mean2, stddev2;
    cv::meanStdDev(img1, mean1, stddev1);
    cv::meanStdDev(img2, mean2, stddev2);

    if (stddev1.at<double>(0) < 5.0 || stddev2.at<double>(0) < 5.0) 
    {
        std::cout << "[LightGlue WARNING] Frame sem textura detectado. Ignorando match." << std::endl;
        return;
    }

    try 
    {
        int H_INFER = img1.rows;
        int W_INFER = img1.cols;

        cv::Mat gray1, gray2;
        if (img1.channels() == 3) cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY); else gray1 = img1.clone();
        if (img2.channels() == 3) cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY); else gray2 = img2.clone();

        cv::Mat float1, float2;
        gray1.convertTo(float1, CV_32FC1, 1.0 / 255.0);
        gray2.convertTo(float2, CV_32FC1, 1.0 / 255.0);

        if (!float1.isContinuous()) float1 = float1.clone();
        if (!float2.isContinuous()) float2 = float2.clone();

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        if (lg_input_names_.size() == 1) 
        {
            size_t image_size = float1.total();
            std::vector<float> batched_image(2 * image_size);
            
            std::memcpy(batched_image.data(), float1.data, image_size * sizeof(float));
            std::memcpy(batched_image.data() + image_size, float2.data, image_size * sizeof(float));

            std::vector<int64_t> shape = {2, 1, H_INFER, W_INFER};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, batched_image.data(), batched_image.size(), shape.data(), shape.size());

            auto outputs = ort_session_lightglue_->Run(
                Ort::RunOptions{nullptr}, 
                lg_input_names_.data(), &input_tensor, 1, 
                lg_output_names_.data(), lg_output_names_.size()
            );

            int kpts_idx = 0;    
            int matches_idx = 1; 

            auto kpt_shape = outputs[kpts_idx].GetTensorTypeAndShapeInfo().GetShape();
            int N = (kpt_shape.size() >= 2) ? kpt_shape[1] : 0; 
            auto kpt_type = outputs[kpts_idx].GetTensorTypeAndShapeInfo().GetElementType();

            if (kpt_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) 
            {
                float* k_ptr = outputs[kpts_idx].GetTensorMutableData<float>();
                for (int i=0; i<N; i++) kp1.push_back(cv::Point2f(k_ptr[i*2], k_ptr[i*2+1]));
                
                float* k1_ptr = k_ptr + (N*2);
                for (int i=0; i<N; i++) kp2.push_back(cv::Point2f(k1_ptr[i*2], k1_ptr[i*2+1]));
            } 
            else if (kpt_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) 
            {
                int64_t* k_ptr = outputs[kpts_idx].GetTensorMutableData<int64_t>();
                for (int i=0; i<N; i++) kp1.push_back(cv::Point2f((float)k_ptr[i*2], (float)k_ptr[i*2+1]));
                
                int64_t* k1_ptr = k_ptr + (N*2);
                for (int i=0; i<N; i++) kp2.push_back(cv::Point2f((float)k1_ptr[i*2], (float)k1_ptr[i*2+1]));
            }

            auto match_shape = outputs[matches_idx].GetTensorTypeAndShapeInfo().GetShape();
            int M = (match_shape.size() >= 1) ? match_shape[0] : 0; 
            auto match_type = outputs[matches_idx].GetTensorTypeAndShapeInfo().GetElementType();

            if (M > 0) 
            {
                if (match_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) 
                {
                    int64_t* m_ptr = outputs[matches_idx].GetTensorMutableData<int64_t>();
                    for (int i = 0; i < M; i++) {
                        matches.push_back(cv::DMatch((int)m_ptr[i*3 + 1], (int)m_ptr[i*3 + 2], 0.0f));
                    }
                } 
                else if (match_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) 
                {
                    int32_t* m_ptr = outputs[matches_idx].GetTensorMutableData<int32_t>();
                    for (int i = 0; i < M; i++) {
                        matches.push_back(cv::DMatch(m_ptr[i*3 + 1], m_ptr[i*3 + 2], 0.0f));
                    }
                }
            }
        }
        else 
        {
            // MODELO LEGADO (Separado)
            std::vector<int64_t> shape1 = {1, 1, H_INFER, W_INFER};
            std::vector<int64_t> shape2 = {1, 1, H_INFER, W_INFER};

            Ort::Value tensor1 = Ort::Value::CreateTensor<float>(memory_info, (float*)float1.data, float1.total(), shape1.data(), shape1.size());
            Ort::Value tensor2 = Ort::Value::CreateTensor<float>(memory_info, (float*)float2.data, float2.total(), shape2.data(), shape2.size());

            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(tensor1));
            inputs.push_back(std::move(tensor2));

            auto outputs = ort_session_lightglue_->Run(
                Ort::RunOptions{nullptr}, 
                lg_input_names_.data(), inputs.data(), lg_input_names_.size(), 
                lg_output_names_.data(), lg_output_names_.size()
            );

            auto shape_kpts0 = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            int N = (shape_kpts0.size() == 3) ? shape_kpts0[1] : shape_kpts0[0];
            auto type_kpts0 = outputs[0].GetTensorTypeAndShapeInfo().GetElementType();
            
            if(type_kpts0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                float* kpts0_ptr = outputs[0].GetTensorMutableData<float>();
                for(int i=0; i<N; i++) kp1.push_back(cv::Point2f(kpts0_ptr[i*2], kpts0_ptr[i*2+1]));
            } else if (type_kpts0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                int64_t* kpts0_ptr = outputs[0].GetTensorMutableData<int64_t>();
                for(int i=0; i<N; i++) kp1.push_back(cv::Point2f((float)kpts0_ptr[i*2], (float)kpts0_ptr[i*2+1]));
            }

            auto shape_kpts1 = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
            int M_kpts = (shape_kpts1.size() == 3) ? shape_kpts1[1] : shape_kpts1[0];
            auto type_kpts1 = outputs[1].GetTensorTypeAndShapeInfo().GetElementType();

            if(type_kpts1 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                float* kpts1_ptr = outputs[1].GetTensorMutableData<float>();
                for(int i=0; i<M_kpts; i++) kp2.push_back(cv::Point2f(kpts1_ptr[i*2], kpts1_ptr[i*2+1]));
            } else if (type_kpts1 == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                int64_t* kpts1_ptr = outputs[1].GetTensorMutableData<int64_t>();
                for(int i=0; i<M_kpts; i++) kp2.push_back(cv::Point2f((float)kpts1_ptr[i*2], (float)kpts1_ptr[i*2+1]));
            }

            auto match_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
            auto type_info = outputs[2].GetTensorTypeAndShapeInfo().GetElementType();

            if (match_shape.size() == 2 && match_shape[1] == 2) 
            {
                int K = match_shape[0];
                if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    int64_t* p = outputs[2].GetTensorMutableData<int64_t>();
                    for(int i=0; i<K; i++) matches.push_back(cv::DMatch((int)p[i*2], (int)p[i*2+1], 0.0f));
                } else {
                    int32_t* p = outputs[2].GetTensorMutableData<int32_t>();
                    for(int i=0; i<K; i++) matches.push_back(cv::DMatch(p[i*2], p[i*2+1], 0.0f));
                }
            } 
            else if (match_shape.size() >= 2) 
            {
                int N_matches = match_shape[1];
                if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                    int64_t* p = outputs[2].GetTensorMutableData<int64_t>();
                    for(int i=0; i<N_matches; i++) if(p[i] > -1) matches.push_back(cv::DMatch(i, (int)p[i], 0.0f));
                } else {
                    int32_t* p = outputs[2].GetTensorMutableData<int32_t>();
                    for(int i=0; i<N_matches; i++) if(p[i] > -1) matches.push_back(cv::DMatch(i, (int)p[i], 0.0f));
                }
            }
        }
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "[LightGlue ERROR] Erro critico C++: " << e.what() << std::endl;
        kp1.clear(); kp2.clear(); matches.clear();
        return; 
    }
}

} // namespace slam_feature_matching