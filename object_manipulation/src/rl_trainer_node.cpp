#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <mutex>
#include <vector>
#include <functional>
#include <filesystem>
#include <string>

// Usar namespaces para simplificar
using namespace std::chrono_literals;
using PointCloud2 = sensor_msgs::msg::PointCloud2;

// Definição da Arquitetura da Rede Neural em C++
struct PolicyNet : torch::nn::Module 
{
    PolicyNet() 
    {
        l1 = register_module("l1", torch::nn::Linear(3000, 512));
        l2 = register_module("l2", torch::nn::Linear(512, 256));
        l3 = register_module("l3", torch::nn::Linear(256, 128));
        l4 = register_module("l4", torch::nn::Linear(128, 64));
        l5 = register_module("l5", torch::nn::Linear(64, 3));
    }

    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(l1->forward(x));
        x = torch::relu(l2->forward(x));
        x = torch::relu(l3->forward(x));
        x = torch::relu(l4->forward(x));
        x = l5->forward(x);
        return x;
    }

    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr}, l5{nullptr};
};

// Definição da Classe do Nó
class RLTrainerNode : public rclcpp::Node 
{
    public:
        RLTrainerNode();
        ~RLTrainerNode();

    private:
        void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
        void create_canonical_grid();
        torch::Tensor process_pointcloud_to_vector(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

        // Modelo C++, otimizadores e tensores
        std::shared_ptr<PolicyNet> model_;
        torch::Device device_;
        std::unique_ptr<torch::optim::Optimizer> optimizer_;
        torch::Tensor canonical_grid_;
        
        // ROS 2
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_publisher_;
        
        // Mutex
        std::mutex training_mutex_;
        
        // Parâmetros
        long input_dim_, output_dim_;
        float learning_rate_;
};

// Implementação das Funções do Nó
RLTrainerNode::RLTrainerNode() : Node("rl_trainer_node"), device_(torch::kCPU) 
{
    // Parâmetros
    input_dim_ = 3000;
    output_dim_ = 3;
    learning_rate_ = 0.0005e-1;

    if (torch::cuda::is_available()) {
        RCLCPP_INFO(this->get_logger(), "CUDA está disponível! Usando a GPU.");
        device_ = torch::kCUDA;
    } else {
        RCLCPP_INFO(this->get_logger(), "CUDA não encontrada. Usando a CPU.");
    }
    
    std::string model_path = "/home/momesso/pibic/src/object_manipulation/models/policy_net_state.pt";

    // 1. Sempre criamos uma instância da arquitetura primeiro.
    model_ = std::make_shared<PolicyNet>();
    model_->to(device_);

    // 2. Se um arquivo de pesos salvo existir, carregamos o estado NELE.
    if (std::filesystem::exists(model_path)) {
        RCLCPP_INFO(this->get_logger(), "Arquivo de pesos encontrado. Carregando estado do modelo...");
        try {
            torch::load(model_, model_path);
            RCLCPP_INFO(this->get_logger(), "Pesos carregados com sucesso.");
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Falha ao carregar pesos: %s", e.what());
        }
    } else {
        RCLCPP_WARN(this->get_logger(), "Arquivo de pesos não encontrado. Iniciando modelo com pesos zerados.");
        torch::NoGradGuard no_grad;
        for (auto& param : model_->parameters()) {
            param.zero_();
        }
    }

    model_->train(); // Garante que o modelo está em modo de treinamento

    // 3. O otimizador é criado com os parâmetros do modelo já carregados ou zerados.
    optimizer_ = std::make_unique<torch::optim::Adam>(model_->parameters(), torch::optim::AdamOptions(learning_rate_));
    
    subscription_ = this->create_subscription<PointCloud2>(
        "/points", 10, std::bind(&RLTrainerNode::pointcloud_callback, this, std::placeholders::_1));

    debug_publisher_ = this->create_publisher<PointCloud2>("/debug_points", 10);
    
    create_canonical_grid();
    RCLCPP_INFO(this->get_logger(), "Nó de Treinamento RL (C++) iniciado.");
}

RLTrainerNode::~RLTrainerNode() 
{
    std::string save_path = "/home/momesso/pibic/src/object_manipulation/models/policy_net_state.pt";
    RCLCPP_INFO(this->get_logger(), "Salvando estado do modelo em: %s", save_path.c_str());
    try {
        // Salva o estado do objeto C++ diretamente.
        torch::save(model_, save_path);
        RCLCPP_INFO(this->get_logger(), "Modelo salvo com sucesso.");
    } catch (const c10::Error& e) {
        RCLCPP_ERROR(this->get_logger(), "Falha ao salvar o modelo: %s", e.what());
    }
}

void RLTrainerNode::pointcloud_callback(const PointCloud2::SharedPtr msg) 
{
    debug_publisher_->publish(*msg);
    std::unique_lock<std::mutex> lock(training_mutex_, std::try_to_lock);

    if (!lock.owns_lock()) { return; }

    torch::Tensor obs_vector = process_pointcloud_to_vector(msg).to(device_);
    torch::Tensor action_mean = model_->forward(obs_vector);

    auto std_dev = torch::ones_like(action_mean) * 0.000001;
    auto sampled_action = action_mean + torch::randn_like(action_mean) * std_dev;
    auto log_std = torch::log(std_dev);
    auto squared_diff = torch::pow(sampled_action - action_mean, 2);
    auto logprob = -0.5 * squared_diff / (torch::pow(std_dev, 2)) - log_std - std::log(std::sqrt(2 * M_PI));
    auto logprob_sum = logprob.sum();
    auto target = torch::tensor({0.044, -0.055, 20.088}, device_);
    auto reward = -torch::norm(sampled_action - target);
    auto loss = -logprob_sum * reward;

    RCLCPP_INFO_STREAM(this->get_logger(), "Saída da Rede (action_mean): " << action_mean);
    
    optimizer_->zero_grad();
    loss.backward();
    optimizer_->step();

    RCLCPP_INFO(this->get_logger(), "Ciclo de treinamento concluído! Recompensa: %.4f, Loss: %.4f", 
                reward.item().toFloat(), loss.item().toFloat());
}

void RLTrainerNode::create_canonical_grid() 
{
    float spacing = 0.01; int size = 10;
    canonical_grid_ = torch::zeros({1000, 3});
    int index = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                canonical_grid_[index][0] = i * spacing;
                canonical_grid_[index][1] = j * spacing;
                canonical_grid_[index][2] = k * spacing;
                index++;
            }
        }
    }
    canonical_grid_ = canonical_grid_.to(device_);
}

torch::Tensor RLTrainerNode::process_pointcloud_to_vector(const PointCloud2::SharedPtr msg) 
{
    float spacing = 0.01;
    int size = 10;
    
    auto occupancy_tensor = torch::zeros({1000}, torch::kFloat32).to(device_);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x"), iter_y(*msg, "y"), iter_z(*msg, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        float x = *iter_x;
        float y = *iter_y;
        float z = *iter_z;

        int ix = static_cast<int>(std::round(x / spacing));
        int iy = static_cast<int>(std::round(y / spacing));
        int iz = static_cast<int>(std::round(z / spacing));

        if (ix >= 0 && ix < size && iy >= 0 && iy < size && iz >= 0 && iz < size) {
            int index = ix * (size * size) + iy * size + iz;
            occupancy_tensor[index] = 1.0;
        }
    }
    
    auto final_tensor = canonical_grid_ * occupancy_tensor.unsqueeze(1);

    return final_tensor.flatten();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RLTrainerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}