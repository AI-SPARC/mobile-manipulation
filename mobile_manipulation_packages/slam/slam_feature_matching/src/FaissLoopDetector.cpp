#include "slam_core/FaissLoopDetector.hpp"
#include <cmath>
#include <iostream>

namespace slam_core 
{

FaissLoopDetector::FaissLoopDetector(float similarity_threshold)
: similarity_threshold_(similarity_threshold)
{
    int d = 384; 
    inner_index_ = new faiss::IndexFlatIP(d);
    faiss_index_ = new faiss::IndexIDMap(inner_index_);
}

FaissLoopDetector::~FaissLoopDetector()
{
    delete faiss_index_;
    delete inner_index_;
}

void FaissLoopDetector::normalize_vector(std::vector<float>& v) 
{
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& x : v) x /= norm;
    }
}

std::pair<int, int> FaissLoopDetector::process_feature_and_find_loop(int robot_id, int current_kf_id, std::vector<float> current_vector)
{
    normalize_vector(current_vector);

    int best_loop_robot_id = -1;
    int best_loop_kf_id = -1;
    float best_score = 0.0f;

    if (faiss_index_->ntotal > 0) 
    {
        int k = faiss_index_->ntotal; 
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> labels(k);

        faiss_index_->search(1, current_vector.data(), k, distances.data(), labels.data());

        for (int i = 0; i < k; ++i) 
        {
            if (labels[i] == -1) continue; 
            
            int cand_robot_id = static_cast<int>(labels[i] >> 32);
            int cand_kf_id = static_cast<int>(labels[i] & 0xFFFFFFFF);
            
            if (robot_id == cand_robot_id && (current_kf_id - cand_kf_id) < 20) continue; 
            
            if (distances[i] > similarity_threshold_) 
            {
                if (best_loop_kf_id == -1) 
                {
                    best_score = distances[i];
                    best_loop_robot_id = cand_robot_id;
                    best_loop_kf_id = cand_kf_id;
                }
            }
        }
    }

    faiss::idx_t global_id = (static_cast<int64_t>(robot_id) << 32) | static_cast<int64_t>(current_kf_id);
    faiss_index_->add_with_ids(1, current_vector.data(), &global_id);

    return {best_loop_robot_id, best_loop_kf_id};
}

} // namespace slam_core