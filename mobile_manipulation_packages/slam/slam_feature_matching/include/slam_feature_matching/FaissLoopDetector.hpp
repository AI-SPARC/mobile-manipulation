#ifndef SLAM_CORE_FAISS_LOOP_DETECTOR_HPP
#define SLAM_CORE_FAISS_LOOP_DETECTOR_HPP

#include <vector>
#include <utility>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>

namespace slam_core 
{

class FaissLoopDetector 
{
public:
    FaissLoopDetector(float similarity_threshold);
    ~FaissLoopDetector();

    std::pair<int, int> process_feature_and_find_loop(int robot_id, int current_kf_id, std::vector<float> current_vector);

private:
    void normalize_vector(std::vector<float>& v);

    float similarity_threshold_;
    faiss::IndexFlatIP* inner_index_;
    faiss::IndexIDMap* faiss_index_;
};

} // namespace slam_core

#endif // SLAM_CORE_FAISS_LOOP_DETECTOR_HPP