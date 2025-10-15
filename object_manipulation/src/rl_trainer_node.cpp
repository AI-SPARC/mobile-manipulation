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
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

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

// Definição da Classe do Nó com funções implementadas DENTRO da classe
class RLTrainerNode : public rclcpp::Node
{
public:
    RLTrainerNode() : Node("rl_trainer_node"), device_(torch::kCPU)
    {
        // Parâmetros
        input_dim_ = 3000;
        output_dim_ = 3;

        // --- CORREÇÃO DE TAXA DE APRENDIZADO ---
        // 1e-3 (0.001) é um valor padrão muito comum e eficaz para Adam.
        // 1e-4 (0.0001) também é bom, 1e-7 é muito baixo.
        learning_rate_ = 1e-7; 

        add_ground_plane();

        if (torch::cuda::is_available()) {
            RCLCPP_INFO(this->get_logger(), "CUDA está disponível! Usando a GPU.");
            device_ = torch::kCUDA;
        } else {
            RCLCPP_INFO(this->get_logger(), "CUDA não encontrada. Usando a CPU.");
        }

        std::string model_path = "/home/momesso/pibic/src/object_manipulation/models/policy_net_state.pt";
        model_ = std::make_shared<PolicyNet>();
        model_->to(device_);

        if (std::filesystem::exists(model_path)) {
            RCLCPP_INFO(this->get_logger(), "Arquivo de pesos encontrado. Carregando estado do modelo...");
            try {
                torch::load(model_, model_path);
                RCLCPP_INFO(this->get_logger(), "Pesos carregados com sucesso.");
            } catch (const c10::Error& e) {
                RCLCPP_ERROR(this->get_logger(), "Falha ao carregar pesos: %s", e.what());
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "Arquivo de pesos não encontrado. Usando pesos com inicialização padrão.");
        }

        model_->train();

        optimizer_ = std::make_unique<torch::optim::Adam>(model_->parameters(), torch::optim::AdamOptions(learning_rate_));

        subscription_ = this->create_subscription<PointCloud2>(
            "/points", 10, std::bind(&RLTrainerNode::pointcloud_callback, this, std::placeholders::_1));

        debug_publisher_ = this->create_publisher<PointCloud2>("/debug_points", 10);
        create_canonical_grid();

        RCLCPP_INFO(this->get_logger(), "Nó de Treinamento RL (C++) iniciado.");
    }

    ~RLTrainerNode()
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

private:
    
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        debug_publisher_->publish(*msg);
        std::unique_lock<std::mutex> lock(training_mutex_, std::try_to_lock);

        if (!lock.owns_lock()) { return; }

        torch::Tensor current_object_center;
        float sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        size_t point_count_in_cloud = 0;

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x"), iter_y(*msg, "y"), iter_z(*msg, "z");
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            sum_x += *iter_x;
            sum_y += *iter_y;
            sum_z += *iter_z;
            point_count_in_cloud++;
        }

        if (point_count_in_cloud > 0)
        {
            current_object_center = torch::tensor({sum_x / point_count_in_cloud,
                                                    sum_y / point_count_in_cloud,
                                                    sum_z / point_count_in_cloud}, device_);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Nuvem de pontos recebida está vazia. Usando centro padrão da grade {0.045, 0.045, 0.045} como fallback.");
            current_object_center = torch::tensor({0.045, 0.045, 0.045}, device_);
        }

        torch::Tensor obs_vector = process_pointcloud_to_vector(msg).to(device_);
        torch::Tensor action_mean = model_->forward(obs_vector);

        auto std_dev = torch::ones_like(action_mean) * 0.0000001; 
        
        auto sampled_action = action_mean + torch::randn_like(action_mean) * std_dev;
        auto log_std = torch::log(std_dev);
        auto squared_diff = torch::pow(sampled_action - action_mean, 2);
        auto logprob = -0.5 * squared_diff / (torch::pow(std_dev, 2)) - log_std - std::log(std::sqrt(2 * M_PI));
        auto logprob_sum = logprob.sum();

        // --- INÍCIO DA NOVA LÓGICA DE RECOMPENSA ---
        torch::Tensor reward;
        auto dist_to_object_center = torch::norm(sampled_action - current_object_center);
        
        // Pega o valor da distância para usar na condição
        float distance_val = dist_to_object_center.item<float>();
        // Adiciona um valor pequeno para evitar divisão por zero
        const float epsilon = 1e-6; 

        if (distance_val > 0.15)
        {
            // Se a distância for grande, a recompensa é uma penalidade linear (distância * -1)
            reward = -dist_to_object_center;
        }
        else
        {
            // Se a distância for pequena, a recompensa é o inverso da distância.
            // Quanto mais perto de zero a distância, maior a recompensa.
            reward = 1.0 / (dist_to_object_center + epsilon);
        }
        // --- FIM DA NOVA LÓGICA DE RECOMPENSA ---

        auto loss = -logprob_sum * reward;

        RCLCPP_INFO_STREAM(this->get_logger(), "Saída da Rede (action_mean): " << action_mean);
        RCLCPP_INFO_STREAM(this->get_logger(), "Centro do Objeto (calculado da nuvem): " << current_object_center);

        optimizer_->zero_grad();    
        loss.backward();
        optimizer_->step();

        RCLCPP_INFO(this->get_logger(), "Ciclo de treinamento concluído! Recompensa: %.4f, Loss: %.4f, Distância ao Centro: %.4f",
                    reward.item().toFloat(), loss.item().toFloat(), dist_to_object_center.item<float>());
    }

    void create_canonical_grid()
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

    torch::Tensor process_pointcloud_to_vector(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        float spacing = 0.01;
        int size = 10;

        // 1. Cria um tensor de "ocupação" com 1000 zeros.
        //    Este tensor vai marcar quais pontos da grade foram recebidos.
        auto occupancy_tensor = torch::zeros({1000}, torch::kFloat32).to(device_);

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x"), iter_y(*msg, "y"), iter_z(*msg, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;

            int ix = static_cast<int>(std::round(x / spacing));
            int iy = static_cast<int>(std::round(y / spacing));
            int iz = static_cast<int>(std::round(z / spacing));

            // 2. Se um ponto recebido pertence à grade, marca sua posição com '1.0'.
            if (ix >= 0 && ix < size && iy >= 0 && iy < size && iz >= 0 && iz < size) {
                int index = ix * (size * size) + iy * size + iz;
                occupancy_tensor[index] = 1.0;
            }
        }

        // 3. Multiplica a grade canônica completa pelo tensor de ocupação.
        //    - Se o ponto existia (valor 1.0), a coordenada é mantida (ex: {0.5, 0.2, 0.1} * 1.0 = {0.5, 0.2, 0.1}).
        //    - Se o ponto estava faltando (valor 0.0), a coordenada se torna zero (ex: {0.8, 0.3, 0.4} * 0.0 = {0.0, 0.0, 0.0}).
        auto final_tensor = canonical_grid_ * occupancy_tensor.unsqueeze(1);

        return final_tensor.flatten();
    }

    void remove_collision_box(const std::string &id)
    {
        static moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
        planning_scene_interface.removeCollisionObjects({id});
        RCLCPP_INFO(rclcpp::get_logger("remove_collision_box"),
                    "Objeto %s removido do planning scene.", id.c_str());
    }

    void initMoveGroup() {
        try {
            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "arm");

            move_group_gripper = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "gripper");

            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface inicializado com sucesso.");
            init_timer_->cancel();
        } catch (const std::exception &e)
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }
    }

    float positions_for_arm(const geometry_msgs::msg::Pose &target_pose)
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return 0;
        }

        move_group_arm->setWorkspace(-2.0, -2.0, 0.01, 2.0,  2.0, 2.0);
        move_group_arm->setStartStateToCurrentState();
        move_group_arm->setPlannerId("RRTConnectkConfigDefault");
        move_group_arm->setPoseTarget(target_pose);
        move_group_arm->setPlanningTime(2.0);
        move_group_arm->setNumPlanningAttempts(40);
        move_group_arm->setMaxVelocityScalingFactor(1.0);
        move_group_arm->setMaxAccelerationScalingFactor(1.0);
        move_group_arm->setGoalTolerance(0.002);
        move_group_arm->setGoalJointTolerance(0.002);
        move_group_arm->setGoalPositionTolerance(0.002);
        move_group_arm->setGoalOrientationTolerance(0.002);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_arm->execute(plan);
            rclcpp::sleep_for(std::chrono::milliseconds(50));
        }

        if (plan.trajectory.joint_trajectory.points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Trajetória vazia retornada pelo planejador");
            return -100;
        }
        return 0;
    }

    void return_to_origin()
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do arm não inicializado.");
            return;
        }
        move_group_arm->setJointValueTarget({
            {"panda_joint1", 0.0},
            {"panda_joint2", -0.7853981633974483},
            {"panda_joint3", 0.0},
            {"panda_joint4", -2.356194490192345},
            {"panda_joint5", 0.0},
            {"panda_joint6", 1.5707963267948966},
            {"panda_joint7", 0.7853981633974483},
        });

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_arm->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            if (move_group_arm->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
                 RCLCPP_INFO(this->get_logger(), "Braço retornou à posição original.");
            }
        }
    }

    void close_gripper()
    {
        if (!move_group_gripper) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do gripper não inicializado.");
            return;
        }
        move_group_gripper->setStartStateToCurrentState();
        move_group_gripper->setJointValueTarget({
            {"panda_finger_joint1", 0.01},
            {"panda_finger_joint2", 0.01}
        });
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_gripper->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            if (move_group_gripper->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
                rclcpp::sleep_for(std::chrono::milliseconds(50));
                RCLCPP_INFO(this->get_logger(), "Gripper fechou (MoveIt).");
            }
        }
    }

    void open_gripper()
    {
        if (!move_group_gripper) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do gripper não inicializado.");
            return;
        }
        move_group_gripper->setStartStateToCurrentState();
        move_group_gripper->setJointValueTarget({
            {"panda_finger_joint1", 0.037},
            {"panda_finger_joint2", 0.037}
        });
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_gripper->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            if (move_group_gripper->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
                rclcpp::sleep_for(std::chrono::milliseconds(50));
                RCLCPP_INFO(this->get_logger(), "Gripper abriu (MoveIt).");
            }
        }
    }

    void add_ground_plane()
    {
        static moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
        moveit_msgs::msg::CollisionObject ground;
        ground.id = "ground_plane";
        ground.header.frame_id = "world";
        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions = {10.0, 10.0, 0.01};
        geometry_msgs::msg::Pose pose;
        pose.position.x = 0.0;
        pose.position.y = 0.0;
        pose.position.z = 0.01;
        pose.orientation.w = 1.0;
        ground.primitives.push_back(primitive);
        ground.primitive_poses.push_back(pose);
        ground.operation = ground.ADD;
        planning_scene_interface.applyCollisionObjects({ground});
    }


    // Modelo C++, otimizadores e tensores
    std::shared_ptr<PolicyNet> model_;
    torch::Device device_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    torch::Tensor canonical_grid_;

    // ROS 2
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_publisher_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;

    rclcpp::TimerBase::SharedPtr init_timer_;

    // Mutex
    std::mutex training_mutex_;

    // Parâmetros
    long input_dim_, output_dim_;
    float learning_rate_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RLTrainerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}