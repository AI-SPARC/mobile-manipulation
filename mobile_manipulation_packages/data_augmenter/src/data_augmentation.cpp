#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <omp.h>

namespace fs = std::filesystem;

// ==========================================
// CONFIGURAÇÕES
// ==========================================
const float CUBE_SIDE = 0.06f; 
const float WORKSPACE_LIMIT = CUBE_SIDE / 2.0f; 
const int NUM_AUGMENTATIONS_PER_FILE = 5000; 
const std::string DATASET_ROOT = "/home/momesso/isaac-sim/toma";
const std::string OUTPUT_ROOT = "/home/momesso/isaac-sim/toma_augmentation";

class DataAugmenter : public rclcpp::Node {
public:
    DataAugmenter() : Node("data_augmenter") {
        RCLCPP_INFO(this->get_logger(), "=== DATA AUGMENTER: HIERARCHICAL OUTPUT ===");
        RCLCPP_INFO(this->get_logger(), "Limite do Workspace: +/- %.4f m", WORKSPACE_LIMIT);
        RCLCPP_INFO(this->get_logger(), "Saída organizada em: %s/run_X/aug_Y", OUTPUT_ROOT.c_str());
        
        if (!fs::exists(OUTPUT_ROOT)) fs::create_directories(OUTPUT_ROOT);
        process_dataset();
    }

private:
    struct GraspEntry {
        int id;
        Eigen::Vector3f pos;
        Eigen::Quaternionf rot;
        bool success;
        float reward;
    };

    void process_dataset() {
        if (!fs::exists(DATASET_ROOT)) return;

        std::vector<fs::path> all_paths;
        for (const auto& entry : fs::directory_iterator(DATASET_ROOT)) {
            if (entry.is_directory() && entry.path().filename().string().find("run_") != std::string::npos) {
                all_paths.push_back(entry.path());
            }
        }
        std::sort(all_paths.begin(), all_paths.end());

        int total_folders = all_paths.size();
        std::atomic<int> total_generated{0};

        for (int idx = 0; idx < total_folders; idx++) {
            fs::path entry_path = all_paths[idx];
            std::string run_dir_name = entry_path.filename().string();
            std::string input_run_dir = entry_path.string();
            
            std::string npy_path = input_run_dir + "/object_pointcloud.npy";
            std::string yaml_path = input_run_dir + "/results_safe.yaml";

            if (!fs::exists(npy_path) || !fs::exists(yaml_path)) continue;

            // Carrega dados com detecção automática de Float/Double
            std::vector<Eigen::Vector3f> point_cloud = load_npy_pointcloud(npy_path);
            std::vector<GraspEntry> grasps = load_yaml_grasps(yaml_path);

            if (point_cloud.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Erro ao ler NPY em %s", run_dir_name.c_str());
                continue;
            }

            // Verifica a extensão real da nuvem carregada para debug
            float max_extent = 0.0f;
            for(const auto& p : point_cloud) {
                max_extent = std::max(max_extent, std::abs(p.x()));
                max_extent = std::max(max_extent, std::abs(p.y()));
                max_extent = std::max(max_extent, std::abs(p.z()));
            }

            RCLCPP_INFO(this->get_logger(), "[%d/%d] %s | Extensão: %.4f m", 
                        idx+1, total_folders, run_dir_name.c_str(), max_extent);

            int successes_in_folder = 0;

            #pragma omp parallel for schedule(dynamic) reduction(+:successes_in_folder)
            for (int i = 0; i < NUM_AUGMENTATIONS_PER_FILE; i++) {
                
                unsigned int seed = i + idx * 1000 + omp_get_thread_num();
                std::mt19937 rng(seed); 

                Eigen::Matrix3f R;
                float scale = 1.0f;
                bool valid_transform = false;
                
                // Fixa a escala em 1.0 (tamanho original)
                std::uniform_real_distribution<float> dist_scale(1.0f, 1.0f);

                for (int attempt = 0; attempt < 500; attempt++) {
                    R = generate_random_rotation(rng);
                    scale = dist_scale(rng);
                    
                    if (is_transform_safe(point_cloud, R, scale)) {
                        valid_transform = true;
                        break;
                    }
                }

                if (!valid_transform) continue; 

                // Aplica Transformação
                std::vector<Eigen::Vector3f> aug_cloud;
                aug_cloud.reserve(point_cloud.size());
                for (const auto& p : point_cloud) {
                    aug_cloud.push_back((R * p) * scale);
                }

                std::vector<GraspEntry> aug_grasps;
                Eigen::Quaternionf q_rot(R);

                for (const auto& g : grasps) {
                    Eigen::Vector3f new_pos = (R * g.pos) * scale;
                    
                    if (std::abs(new_pos.x()) > WORKSPACE_LIMIT || 
                        std::abs(new_pos.y()) > WORKSPACE_LIMIT || 
                        std::abs(new_pos.z()) > WORKSPACE_LIMIT) continue;

                    GraspEntry new_g = g;
                    new_g.pos = new_pos;
                    new_g.rot = (q_rot * g.rot).normalized();
                    aug_grasps.push_back(new_g);
                }

                if (aug_grasps.empty()) continue;

                // ========================================================
                // NOVA ESTRUTURA DE DIRETÓRIOS: OUTPUT / RUN_X / AUG_Y
                // ========================================================
                
                // 1. Pasta da Run (ex: .../toma_augmentation/run_0)
                std::string run_output_folder = OUTPUT_ROOT + "/" + run_dir_name;
                
                // 2. Pasta da Augmentation (ex: aug_0)
                std::string aug_subfolder = "aug_" + std::to_string(i);
                
                // 3. Caminho Final (ex: .../toma_augmentation/run_0/aug_0)
                std::string final_output_dir = run_output_folder + "/" + aug_subfolder;
                
                try {
                    // fs::create_directories cria toda a árvore necessária (run_X e aug_Y)
                    if (!fs::exists(final_output_dir)) fs::create_directories(final_output_dir);
                    
                    save_npy_pointcloud(final_output_dir + "/object_pointcloud.npy", aug_cloud);
                    save_yaml_grasps(final_output_dir + "/results_safe.yaml", aug_grasps);
                    successes_in_folder++;
                } catch (...) {}
            }
            
            if (successes_in_folder == 0) 
            {
                RCLCPP_WARN(this->get_logger(), "FALHA: Nenhuma augmentation gerada para %s.", run_dir_name.c_str());
            } else {
                total_generated += successes_in_folder;
            }
        }
        RCLCPP_INFO(this->get_logger(), "FINALIZADO! Total gerado: %d", total_generated.load());
    }

    // --- MANIPULAÇÃO DE DADOS ---

    std::vector<Eigen::Vector3f> load_npy_pointcloud(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return {};

        std::string header;
        char b;
        while (file.get(b)) {
            header += b;
            if (b == '\n') break;
        }

        bool is_double = (header.find("'descr': '<f8'") != std::string::npos);
        std::vector<Eigen::Vector3f> points;
        
        if (is_double) {
            double buffer[3];
            while (file.read(reinterpret_cast<char*>(buffer), 3 * sizeof(double))) {
                points.emplace_back((float)buffer[0], (float)buffer[1], (float)buffer[2]);
            }
        } else {
            float buffer[3];
            while (file.read(reinterpret_cast<char*>(buffer), 3 * sizeof(float))) {
                points.emplace_back(buffer[0], buffer[1], buffer[2]);
            }
        }
        return points;
    }

    bool is_transform_safe(const std::vector<Eigen::Vector3f>& cloud, const Eigen::Matrix3f& R, float scale) {
        float limit = WORKSPACE_LIMIT * 0.99f;
        size_t n = cloud.size();
        for (size_t k = 0; k < n; k += 50) { 
            Eigen::Vector3f p_trans = (R * cloud[k]) * scale;
            if (std::abs(p_trans.x()) > limit || 
                std::abs(p_trans.y()) > limit || 
                std::abs(p_trans.z()) > limit) {
                return false;
            }
        }
        return true;
    }

    Eigen::Matrix3f generate_random_rotation(std::mt19937& gen) {
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        float u1 = dis(gen);
        float u2 = dis(gen);
        float u3 = dis(gen);
        float sqrt1_u1 = std::sqrt(1 - u1);
        float sqrt_u1 = std::sqrt(u1);
        Eigen::Quaternionf q(
            sqrt1_u1 * std::sin(2 * M_PI * u2),
            sqrt1_u1 * std::cos(2 * M_PI * u2),
            sqrt_u1 * std::sin(2 * M_PI * u3),
            sqrt_u1 * std::cos(2 * M_PI * u3)
        );
        return q.toRotationMatrix();
    }

    void save_npy_pointcloud(const std::string& path, const std::vector<Eigen::Vector3f>& cloud) {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) return;
        std::string header_dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(cloud.size()) + ", 3), }";
        int padding = 64 - ((10 + header_dict.length() + 1) % 64);
        std::string pad_str(padding, ' ');
        header_dict += pad_str + "\n";
        const char magic[] = "\x93NUMPY";
        file.write(magic, 6);
        char ver_major = 1, ver_minor = 0;
        file.write(&ver_major, 1);
        file.write(&ver_minor, 1);
        unsigned short header_len = static_cast<unsigned short>(header_dict.length());
        file.write(reinterpret_cast<const char*>(&header_len), 2);
        file.write(header_dict.c_str(), header_len);
        for (const auto& p : cloud) {
            float data[3] = {p.x(), p.y(), p.z()};
            file.write(reinterpret_cast<const char*>(data), 3 * sizeof(float));
        }
    }

    std::vector<GraspEntry> load_yaml_grasps(const std::string& path) {
        std::vector<GraspEntry> entries;
        try {
            YAML::Node config = YAML::LoadFile(path);
            for (std::size_t i = 0; i < config.size(); i++) {
                GraspEntry e;
                e.id = config[i]["grasp_id"].as<int>();
                auto pos = config[i]["pose_pos"];
                e.pos = Eigen::Vector3f(pos[0].as<float>(), pos[1].as<float>(), pos[2].as<float>());
                auto rot = config[i]["pose_rot"];
                e.rot = Eigen::Quaternionf(rot[0].as<float>(), rot[1].as<float>(), rot[2].as<float>(), rot[3].as<float>());
                e.success = config[i]["contact_success"].as<bool>();
                e.reward = config[i]["reward"].as<float>();
                entries.push_back(e);
            }
        } catch (...) {}
        return entries;
    }

    void save_yaml_grasps(const std::string& path, const std::vector<GraspEntry>& entries) {
        YAML::Emitter out;
        out << YAML::BeginSeq;
        for (const auto& e : entries) {
            out << YAML::BeginMap;
            out << YAML::Key << "grasp_id" << YAML::Value << e.id;
            out << YAML::Key << "pose_pos";
            out << YAML::Flow << YAML::BeginSeq << e.pos.x() << e.pos.y() << e.pos.z() << YAML::EndSeq;
            out << YAML::Key << "pose_rot";
            out << YAML::Flow << YAML::BeginSeq << e.rot.w() << e.rot.x() << e.rot.y() << e.rot.z() << YAML::EndSeq;
            out << YAML::Key << "contact_success" << YAML::Value << e.success;
            out << YAML::Key << "reward" << YAML::Value << e.reward;
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
        std::ofstream fout(path);
        fout << out.c_str();
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DataAugmenter>();
    rclcpp::shutdown();
    return 0;
}