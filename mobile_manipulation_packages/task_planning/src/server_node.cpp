/**
 * @file server_node.cpp
 * @brief Nó central de controle (Task Planner)
 */

#include <memory>
#include <vector>
#include <string>
#include <unordered_set>
#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <cmath>
#include <atomic>
#include <mutex>
#include <map>
#include <chrono>

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/xml_parsing.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

// Mensagens ROS
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <yaml-cpp/yaml.h>

// Interfaces Customizadas
#include "mobile_manipulation_interfaces/action/pick_object.hpp"
#include "mobile_manipulation_interfaces/action/path.hpp"
#include "mobile_manipulation_interfaces/action/controller.hpp"

// Classes Auxiliares
#include <manipulation/IsGripperHolding.hpp>
#include <manipulation/ProjectedReachabilityAnalysis.hpp>
#include <manipulation/IKValidator.hpp>
#include <manipulation/CloudBoxFilter.hpp>
#include <storage_manager/GetStorageInfo.hpp>
#include <storage_manager/Organize.hpp>
#include <navigation/SharedObstacleGraph.hpp>

#include <drl_to_pick_cpp/BridgeToInference.hpp>

namespace BT
{
    // DOC-START: convertFromString
    // Especialização de template para converter strings do XML (ex: "1.0;2.0;3.0")
    // para o tipo complexo geometry_msgs::msg::Pose.
    // O BehaviorTree.CPP exige isso para tipos não primitivos nas Portas de Entrada.
    template <>
    inline geometry_msgs::msg::Pose convertFromString(StringView)
    {
        // Retorna uma pose zerada por padrão.
        // Em uma implementação real, aqui faríamos o parse da string "x;y;z;..."
        return geometry_msgs::msg::Pose();
    }
    // DOC-END: convertFromString
}

// Estados possíveis para uma tarefa assíncrona (Action Client)
// Usado para sincronizar o tick da BT com o callback do ROS
enum class TaskState
{
    IDLE,    // Nenhuma ação rodando
    RUNNING, // Action enviada, aguardando resultado
    SUCCESS, // Action terminou com sucesso
    FAILURE  // Action abortada ou falhou
};

// DOC-START: ParallelAny
// Nó de Controle Personalizado: "Parallel Any" (Paralelo "Ou")
// Executa todos os filhos simultaneamente (no mesmo tick).
// Retorna SUCESSO se *pelo menos um* filho retornar sucesso.
// Retorna FALHA se *pelo menos um* filho retornar falha.
// Caso contrário, retorna RUNNING.
class ParallelAny : public BT::ControlNode
{
public:
    ParallelAny(const std::string& name, const BT::NodeConfig& config)
        : BT::ControlNode(name, config) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override
    {
        // Itera sobre todos os nós filhos registrados neste controle
        for (size_t i = 0; i < children_nodes_.size(); i++)
        {
            BT::TreeNode* child = children_nodes_[i];
            // Executa o tick do filho
            BT::NodeStatus status = child->executeTick();

            // Lógica Short-Circuit: Se um acabou bem, todos acabam bem.
            if (status == BT::NodeStatus::SUCCESS)
            {
                haltChildren(); // Para os outros que ainda estão rodando
                return BT::NodeStatus::SUCCESS;
            }

            // Lógica Short-Circuit de Falha: Se um falhou, o grupo todo falha.
            if (status == BT::NodeStatus::FAILURE)
            {
                haltChildren();
                return BT::NodeStatus::FAILURE;
            }
        }
        // Se ninguém terminou ainda, continuamos rodando.
        return BT::NodeStatus::RUNNING;
    }

    void halt() override
    {
        haltChildren();
        BT::ControlNode::halt();
    }
};
// DOC-END: ParallelAny

// DOC-START: AsyncAction
// Wrapper para criar Actions Stateful (Assíncronas) de forma rápida usando Lambdas.
// Evita ter que criar uma classe .h/.cpp separada para cada nó simples da árvore.
class AsyncAction : public BT::StatefulActionNode
{
public:
    AsyncAction(const std::string& name, const BT::NodeConfig& config,
                std::function<BT::NodeStatus(BT::TreeNode&)> tick_fun,
                std::function<void(BT::TreeNode&)> halt_fun = nullptr) 
        : BT::StatefulActionNode(name, config), tick_fun_(tick_fun), halt_fun_(halt_fun) {}

    BT::NodeStatus onStart() override { return tick_fun_(*this); }

    BT::NodeStatus onRunning() override { return tick_fun_(*this); }

    void onHalted() override 
    {
        if (halt_fun_)
        {
            halt_fun_(*this);
        }
    }

private:
    std::function<BT::NodeStatus(BT::TreeNode&)> tick_fun_;
    std::function<void(BT::TreeNode&)> halt_fun_; 
};
// DOC-END: AsyncAction
// DOC-END: AsyncAction

// ============================================================================
// CLASSE PRINCIPAL DO SERVER NODE
// ============================================================================

class ServerNode : public rclcpp::Node
{
public:
    // DOC-START: ServerNode
    // Construtor: Configura toda a infraestrutura do nó.
    // Recebe referências compartilhadas para os nós auxiliares (Gripper, Storage, Organize)
    // para permitir comunicação direta em memória, sem latência de tópicos.
    ServerNode(
        std::shared_ptr<manipulation::IsGripperHolding> gripper_node,
        std::shared_ptr<storage_manager::StorageNode> storage_node,
        std::shared_ptr<storage_manager::OrganizeNode> organize_node,
        std::shared_ptr<manipulation::ProjectedReachabilityAnalysis> reachability_node,
        std::shared_ptr<navigation::SharedObstacleGraph> obstacle_graph_node,
        std::shared_ptr<manipulation::IKValidator> ik_validator_node,
        std::shared_ptr<manipulation::CloudBoxFilter> cloud_box_filter_node,
        std::shared_ptr<drl_to_pick_cpp::BridgeToInference> bridge_to_inference_node
    )
    : Node("server_node"),
    gripper_monitor_node_(gripper_node),
    storage_node_(storage_node),
    organize_node_(organize_node),
    reachability_node_(reachability_node),
    obstacle_graph_node_(obstacle_graph_node),
    ik_validator_node_(ik_validator_node),
    cloud_box_filter_node_(cloud_box_filter_node),
    bridge_to_inference_node_(bridge_to_inference_node)
    {
        // Declaração de parâmetros (caminhos de arquivos)
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("bt_xml_path", "");
        this->declare_parameter<bool>("use_graspnet", false);

        yaml_file = this->get_parameter("yaml_file").as_string();
        std::string bt_xml_path = this->get_parameter("bt_xml_path").as_string();
        use_graspnet = this->get_parameter("use_graspnet").as_bool();
        
        // 1. Subscribers:
        // Ouve as detecções do YOLO ("vision_msgs")
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&ServerNode::detection_callback, this, std::placeholders::_1));

        // Ouve a posição do robô ("nav_msgs")
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&ServerNode::odom_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("object_pose", 10);
        
        // 2. Action Clients (Clientes de Ação):
        // Conecta com os servidores de Manipulação, Planejamento de Caminho e Controle.
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(this, "pick_object");
        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(this, "path");
        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(this, "controller");

        // Inicializa máquina de estados interna das ações
        path_state_ = TaskState::IDLE;
        nav_state_ = TaskState::IDLE;
        manipulation_state_ = TaskState::IDLE;

        // 3. Behavior Tree:
        // Registra os nós e carrega o arquivo XML
        setup_behavior_tree(bt_xml_path);

        // 4. Thread Dedicada:
        // O BT loop roda em uma thread separada para não bloquear o executor do ROS (spin).
        bt_thread_ = std::thread(&ServerNode::bt_loop, this);

        RCLCPP_INFO(this->get_logger(), "ServerNode iniciado.");

        // Timer para debug (publica pose do alvo)
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&ServerNode::publish_pose, this));

        // Carrega lista de objetos permitidos do YAML
        if(!yaml_file.empty())
        {
            loadLocationsFromYaml(yaml_file);
        }

        

        // Publishers

    }
    // DOC-END: ServerNode

    // DOC-START: ~ServerNode
    ~ServerNode()
    {
        // Garante que a thread da BT seja encerrada corretamente ao fechar o nó
        if (bt_thread_.joinable()) bt_thread_.join();
    }
    // DOC-END: ~ServerNode

private:
    // DOC-START: internal_structs
    // Estrutura auxiliar para agrupar informações de um objeto detectado pela visão computacional.
    struct ObjectInfo
    {
        std::string id;                 // ID único ou classe do objeto (ex: "garrafa_1")
        geometry_msgs::msg::Pose pose;  // Posição e orientação espacial do objeto
        geometry_msgs::msg::Vector3 size; // Tamanho da bounding box (x, y, z)
    };
    // DOC-END: internal_structs

    // DOC-START: member_variables
    // --- Injeção de Dependências ---
    // Ponteiro para o nó que envia a point cloud via msgpack para o arquivo python que faz a inferência no graspnet.
     std::shared_ptr<drl_to_pick_cpp::BridgeToInference> bridge_to_inference_node_;
    // Ponteiro para o nó que retira a point cloud da bounding box do objeto e depois aumenta a quantidade de pontos para melhor inferência.
    std::shared_ptr<manipulation::CloudBoxFilter> cloud_box_filter_node_;
    // Ponteiro para o nó que verifica se o robô consegue achar uma IK para uma série de pontos passados.
    std::shared_ptr<manipulation::IKValidator> ik_validator_node_;
    // Ponteiro para o nó que modifica o grafo de obstáculos.
    std::shared_ptr<navigation::SharedObstacleGraph> obstacle_graph_node_;
    // Ponteiro para o nó que verifica o ponto ideal para pegar o objeto.
    std::shared_ptr<manipulation::ProjectedReachabilityAnalysis> reachability_node_;
    // Ponteiro para o nó que monitora o sensor da garra
    std::shared_ptr<manipulation::IsGripperHolding> gripper_monitor_node_;
    // Ponteiro para o gerenciador de banco de dados de posições (Storage)
    std::shared_ptr<storage_manager::StorageNode> storage_node_;
    // Ponteiro para o algoritmo de organização (Bin Packing)
    std::shared_ptr<storage_manager::OrganizeNode> organize_node_;
    // Ponteiro para o publicador de logs do Groot2 (Visualizador da Behavior Tree)
    std::unique_ptr<BT::Groot2Publisher> groot_publisher_;

    // --- Comunicação ROS 2 ---

    // Publicador para enviar a pose do objeto em tempo real para o nó de manipulação
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
    // Subscriber para receber as detecções do YOLO (Bounding Boxes 3D)
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    // Subscriber para receber a odometria e atualizar a posição do robô
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    // --- Clientes de Ação (Action Clients) ---
    // Cliente para a ação de Pegar/Largar objetos (Manipulação)
    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    // Cliente para o planejador de caminho global (A* / D*)
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    // Cliente para o controlador de trajetória local (Pure Pursuit)
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;

    // --- Handles de Ação ---
    // Handle para controlar a ação de controle ativa (permite cancelar a navegação)
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::SharedPtr active_controller_goal_handle_;
    // Handle para controlar a ação de planejamento ativa
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::SharedPtr active_path_goal_handle_;

    // --- Configuração e Estado Lógico ---
    // Caminho do arquivo YAML com objetos permitidos
    std::string yaml_file;
    // Lista de nomes de objetos que o robô tem permissão para pegar
    std::unordered_set<std::string> authorized_labels;
    // Lista de IDs únicos de objetos que já foram pegos para evitar repetição
    std::unordered_set<std::string> picked;

    // Variáveis temporárias para armazenar dados de objetos
    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;
    // Cache do último objeto válido detectado pela câmera
    ObjectInfo cached_object_;

    // --- Estado do Alvo Atual ---
    // ID do objeto que está sendo processado pela Behavior Tree (vazio se ocioso)
    std::string current_target_id_ = "";
    // Posição do alvo atual
    geometry_msgs::msg::Pose current_target_pose_;

    // Timer para publicar dados de debug periodicamente
    rclcpp::TimerBase::SharedPtr timer_;

    // --- Infraestrutura da Behavior Tree ---
    // Thread dedicada para rodar o tick da árvore sem bloquear o ROS
    std::thread bt_thread_;
    // Mutex para proteger variáveis compartilhadas entre a thread ROS e a thread BT
    std::mutex bt_mutex_;
    // Mutex para proteger a posição atual do robô.
    std::mutex odom_mutex;
    // Objeto principal da árvore de comportamento
    BT::Tree bt_tree_;

    // --- Estados das Tarefas Assíncronas ---
    // Estado atual da tarefa de planejamento de caminho
    TaskState path_state_;
    // Estado atual da tarefa de navegação
    TaskState nav_state_;
    // Estado atual da tarefa de manipulação
    TaskState manipulation_state_;

    // --- Sincronização ---
    // Mutex crítico para proteger transições de estado das Actions
    std::mutex state_mutex_;
    // Mutex para proteger a leitura e escrita do caminho calculado
    std::mutex path_mutex_;

    // Armazena o último caminho recebido do planejador
    nav_msgs::msg::Path last_calculated_path_;
    nav_msgs::msg::Path last_no_filter_calculated_path_;

    // --- Odometria e Flags ---
    // Posição atual do robô no mapa
    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0;
    // Flag atômica para indicar à BT que um novo objeto foi visto
    bool has_new_object_ = false;

    bool use_graspnet = false;
    // DOC-END: member_variables

    // DOC-START: check_task_status
    // Helper para converter o enum interno 'TaskState' para 'BT::NodeStatus'.
    // Também reseta o estado para IDLE automaticamente quando a tarefa termina.
    BT::NodeStatus check_task_status(TaskState &state)
    {
        if (state == TaskState::SUCCESS)
        {
            state = TaskState::IDLE; // Reset para próxima execução
            return BT::NodeStatus::SUCCESS;
        }
        else if (state == TaskState::FAILURE)
        {
            state = TaskState::IDLE; // Reset
            return BT::NodeStatus::FAILURE;
        }
        return BT::NodeStatus::RUNNING; // Ainda processando
    }
    // DOC-END: check_task_status

    // DOC-START: setup_behavior_tree
    // Configura a fábrica da Behavior Tree, registra os nós e carrega o XML.
    // Aqui está definida a lógica de cada nó (Action/Condition) usando Lambdas C++.
    void setup_behavior_tree(const std::string &xml_path)
    {
        BT::BehaviorTreeFactory factory;

        // DOC-START: BT_ParallelAny
        // Registra o nó customizado
        factory.registerNodeType<ParallelAny>("ParallelAny");
        // DOC-END: BT_ParallelAny

        // DOC-START: BT_IsReachable
        // --- Condition: IsReachable ---
        // Verifica se a distância euclidiana entre o robô e o alvo está dentro de um limite.
        factory.registerSimpleCondition("IsReachable", [&](BT::TreeNode &self)
        {
            auto target_pose_opt = self.getInput<geometry_msgs::msg::Pose>("target_pose");
            auto authorized_id_opt = self.getInput<std::string>("object_id");
            auto robot_base_z_opt = self.getInput<double>("robot_base_z");
            auto max_reach_3d_opt = self.getInput<double>("max_reach_3d");

            if (!target_pose_opt) return BT::NodeStatus::FAILURE;
            if (!authorized_id_opt) return BT::NodeStatus::FAILURE;
            if (!robot_base_z_opt) return BT::NodeStatus::FAILURE;
            if (!max_reach_3d_opt) return BT::NodeStatus::FAILURE;


            geometry_msgs::msg::Pose target = target_pose_opt.value();
            std::string authorized_id = authorized_id_opt.value();
            double robot_base_z = robot_base_z_opt.value();
            double max_reach_3d = max_reach_3d_opt.value();


            std::vector<std::pair<float, float>> viable_points;
          
            this->reachability_node_->get_reachable_points(target, robot_base_z, max_reach_3d, viable_points);

            if (viable_points.empty()) 
            {
                RCLCPP_WARN(this->get_logger(), "O alvo é inalcançável.");
                return BT::NodeStatus::FAILURE;
            }

            std::optional<std::tuple<float, float, float>> best_base_opt;
            
            std::vector<std::tuple<float, float, float>> viable_points_3d;
            viable_points_3d.reserve(viable_points.size());

            for (const auto& p : viable_points) 
            {
                viable_points_3d.emplace_back(p.first, p.second, static_cast<float>(robot_base_z));
            }

            std::tuple<float, float, float> actual_robot_position;

            {
                std::lock_guard<std::mutex> lock(odom_mutex);
            
                std::get<0>(actual_robot_position) = pose_x;
                std::get<1>(actual_robot_position) = pose_y;
                std::get<2>(actual_robot_position) = robot_base_z;
            }

            auto start_total = std::chrono::high_resolution_clock::now();
            
            std::sort(viable_points_3d.begin(), viable_points_3d.end(), 
            [&actual_robot_position](const std::tuple<float, float, float>& a, const std::tuple<float, float, float>& b) 
            {
                float rx = std::get<0>(actual_robot_position);
                float ry = std::get<1>(actual_robot_position);
                float rz = std::get<2>(actual_robot_position);

                float dax = std::get<0>(a) - rx;
                float day = std::get<1>(a) - ry;
                float daz = std::get<2>(a) - rz;
                float dist_sq_a = (dax * dax) + (day * day) + (daz * daz);

                float dbx = std::get<0>(b) - rx;
                float dby = std::get<1>(b) - ry;
                float dbz = std::get<2>(b) - rz;
                float dist_sq_b = (dbx * dbx) + (dby * dby) + (dbz * dbz);

                return dist_sq_a < dist_sq_b;
            });

            auto end_total = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count() / 1000.0;
            
       

            
            best_base_opt = this->ik_validator_node_->find_best_base_position(
                viable_points_3d, 
                target, 
                true, 
                this->obstacle_graph_node_,
                authorized_id
            );



            if (best_base_opt.has_value())
            {
                auto p = best_base_opt.value(); 
                std::get<2>(p) = 0.0;

                float px = std::get<0>(p); 
                float py = std::get<1>(p); 
                float pz = std::get<2>(p); 

                float dx_curr = px - this->pose_x;
                float dy_curr = py - this->pose_y;
                float dist_sq = std::sqrt((dx_curr * dx_curr) + (dy_curr * dy_curr));

                const float threshold_sq = 0.07f; 

                if (dist_sq <= threshold_sq)
                {
                    RCLCPP_INFO(this->get_logger(), "O robô JÁ ESTÁ na posição ideal (Dist: %.4f).", dist_sq);
                    return BT::NodeStatus::SUCCESS;
                }
                else
                {
                    
                    geometry_msgs::msg::Pose final_pose;
                    final_pose.position.x = px;
                    final_pose.position.y = py;
                    final_pose.position.z = 0.0; 


                    double target_dx = target.position.x - px;
                    double target_dy = target.position.y - py;


                    double yaw = std::atan2(target_dy, target_dx);

                    tf2::Quaternion q;
                    q.setRPY(0.0, 0.0, yaw); 

                    final_pose.orientation.x = q.x();
                    final_pose.orientation.y = q.y();
                    final_pose.orientation.z = q.z();
                    final_pose.orientation.w = q.w();
                    
                   
                    self.setOutput("adjustment_pose", final_pose);
                    RCLCPP_INFO(this->get_logger(), "Ajuste necessário. Indo para (%.2f, %.2f) virado para o objeto.", px, py);
                    return BT::NodeStatus::FAILURE;
                }
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Falha. O robô não conseguirá pegar o objeto.");
                return BT::NodeStatus::FAILURE;
            }
        },
        {
            BT::InputPort<geometry_msgs::msg::Pose>("target_pose"),
            BT::InputPort<geometry_msgs::msg::Vector3>("target_size"),
            BT::InputPort<std::string>("object_id"),
            BT::InputPort<double>("robot_base_z"),
            BT::InputPort<double>("max_reach_3d"),
            BT::OutputPort<geometry_msgs::msg::Pose>("adjustment_pose")
        });
        // DOC-END: BT_IsReachable

        // DOC-START: BT_IsStillReachable
        factory.registerSimpleCondition("IsStillReachable", [&](BT::TreeNode &self)
        {
            bool reachable = this->ik_validator_node_->is_still_reachable(this->obstacle_graph_node_);

            if(reachable == true)
            {
                return BT::NodeStatus::SUCCESS;
            }
            else if(reachable == false)
            {
                return BT::NodeStatus::FAILURE;
            }
            
            return BT::NodeStatus::FAILURE;
        });
        // DOC-END: BT_IsStillReachable



        // DOC-START: BT_DetectObject
        // --- Action: DetectObject ---
        // Verifica se há um objeto novo detectado pelo callback de visão.
        factory.registerSimpleAction("DetectObject", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_); 

            // Se já temos um alvo travado, retornamos ele (persistencia de alvo)
            if (!current_target_id_.empty())
            {
                self.setOutput("output_pose", current_target_pose_);
                self.setOutput("output_id", current_target_id_);
                self.setOutput("output_size", cached_object_.size);
                return BT::NodeStatus::SUCCESS;
            }

            // Se não, verifica a flag setada pelo detection_callback
            if (!has_new_object_)
            {
                return BT::NodeStatus::RUNNING;
            }

            // Promove o objeto cacheado para alvo atual
            current_target_id_ = cached_object_.id;
            current_target_pose_ = cached_object_.pose;

            self.setOutput("output_pose", current_target_pose_);
            self.setOutput("output_id", current_target_id_);
            self.setOutput("output_size", cached_object_.size);

            // Marca o ID como 'picked' para evitar pegar o mesmo objeto em loop
            picked.insert(current_target_id_);
            has_new_object_ = false;

            RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' travado.", current_target_id_.c_str());
            return BT::NodeStatus::SUCCESS;
        },
        {
            BT::OutputPort<geometry_msgs::msg::Pose>("output_pose"),
            BT::OutputPort<std::string>("output_id"),
            BT::OutputPort<geometry_msgs::msg::Vector3>("output_size")
        });
        // DOC-END: BT_DetectObject

        // DOC-START: BT_ClearTarget
        // --- Action: ClearTarget ---
        // Limpa o alvo atual, permitindo que a detecção busque um novo objeto.
        factory.registerSimpleAction("ClearTarget", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' liberado.", current_target_id_.c_str());
            current_target_id_ = ""; 
            return BT::NodeStatus::SUCCESS;
        });
         // DOC-END: BT_ClearTarget

        // DOC-START BT_IsPathClear
        factory.registerSimpleCondition("IsPathClear", [&](BT::TreeNode& self)
        {
            
            auto map_snapshot = this->obstacle_graph_node_->get_current_map();
            std::pair<float, float> pair_point;


            
            {
                std::lock_guard<std::mutex> lock(path_mutex_);
                

                for(const auto& point : last_no_filter_calculated_path_.poses)
                {
                    pair_point.first = static_cast<float>(point.pose.position.x);
                    pair_point.second = static_cast<float>(point.pose.position.y);

                    if (map_snapshot->find(pair_point) != map_snapshot->end())
                    {
                        cancel_controller_goal();
                        return BT::NodeStatus::FAILURE; 
                    } 
                }
            }
            

          
            return BT::NodeStatus::SUCCESS;
            
        });
        // DOC-END: BT_IsPathClear

         // DOC-START: BT_GetStorageInfo
        // --- Action: GetStorageInfo ---
        // Consulta o banco de dados (StorageNode) para achar uma vaga livre.
        factory.registerSimpleAction("GetStorageInfo", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("object_id");
            if (!id_opt) return BT::NodeStatus::FAILURE;

            // Limpa o ID (ex: "can_34" -> "can") para buscar categoria genérica
            std::string full_id = id_opt.value();
            std::string label = full_id;
            size_t pos = full_id.find('_');
            if (pos != std::string::npos) label = full_id.substr(0, pos);

            geometry_msgs::msg::Pose current_obj_pose;
            {
                std::lock_guard<std::mutex> lock(bt_mutex_);
                current_obj_pose = current_target_pose_;
            }

            // Chama o StorageNode
            auto result = storage_node_->getBestStorage(label, current_obj_pose);

            if (result.success)
            {
                // Exporta os dados da caixa encontrada para a Blackboard
                self.setOutput("storage_pose", result.pose);
                self.setOutput("storage_limits", result.limits);
                self.setOutput("storage_id", result.storage_name);
                self.setOutput("indexes", result.indexes);
                self.setOutput("storage_size", result.size);
                return BT::NodeStatus::SUCCESS;
            }

            RCLCPP_WARN(this->get_logger(), "Storage cheio ou não encontrado para %s", label.c_str());
            return BT::NodeStatus::FAILURE;
        },
        {
            BT::InputPort<std::string>("object_id"),
            BT::OutputPort<geometry_msgs::msg::Pose>("storage_pose"),
            BT::OutputPort<std::vector<double>>("storage_limits"),
            BT::OutputPort<std::string>("storage_id"),
            BT::OutputPort<std::vector<int>>("indexes"),
            BT::OutputPort<geometry_msgs::msg::Vector3>("storage_size")
        });
        // DOC-END: BT_GetStorageInfo

        // DOC-START: BT_ComputePoseToOrganize
        // --- Action: ComputePoseToOrganize ---
        // Calcula a posição exata dentro da caixa usando algoritmo de Bin Packing (OrganizeNode).
        factory.registerSimpleAction("ComputePoseToOrganize", [&](BT::TreeNode &self)
        {
            auto storagePose = self.getInput<geometry_msgs::msg::Pose>("storage_pose");
            auto storageSize = self.getInput<geometry_msgs::msg::Vector3>("storage_size");
            auto objectSize = self.getInput<geometry_msgs::msg::Vector3>("object_size");
            auto indexes = self.getInput<std::vector<int>>("indexes");
            auto objectPadding = self.getInput<float>("object_padding");
            auto zLiftOffset = self.getInput<float>("z_lift_offset");

            if (!storagePose || !storageSize || !objectSize || !indexes || !objectPadding || !zLiftOffset)
            {
                RCLCPP_ERROR(this->get_logger(), "BT: Parâmetros de organização faltando.");
                return BT::NodeStatus::FAILURE;
            }

            std::vector<int> idx_vec = indexes.value();
            if (idx_vec.size() != 3) return BT::NodeStatus::FAILURE;

            // Chama o algoritmo de cálculo geométrico
            std::pair<geometry_msgs::msg::Pose, std::vector<int>> result = organize_node_->placeObjectInBox(
                storagePose.value(), storageSize.value(), objectSize.value(),
                objectPadding.value(), zLiftOffset.value(),
                idx_vec[0], idx_vec[1], idx_vec[2]
            );

            self.setOutput("output_final_pose", std::get<0>(result));
            self.setOutput("new_indexes", std::get<1>(result));

            RCLCPP_INFO(this->get_logger(), "Nova posição de organização calculada.");
            return BT::NodeStatus::SUCCESS;
        },
        {
            BT::InputPort<geometry_msgs::msg::Pose>("storage_pose"),
            BT::InputPort<geometry_msgs::msg::Vector3>("storage_size"),
            BT::InputPort<geometry_msgs::msg::Vector3>("object_size"),
            BT::InputPort<std::vector<int>>("indexes"),
            BT::InputPort<float>("object_padding"),
            BT::InputPort<float>("z_lift_offset"),
            BT::OutputPort<std::vector<int>>("new_indexes"),
            BT::OutputPort<geometry_msgs::msg::Pose>("output_final_pose")
        });
        // DOC-END: BT_ComputePoseToOrganize

        // DOC-START: BT_ComputePoseToStore
        // --- Action: ComputePoseToStore ---
        // Versão simples: Apenas calcula uma pose no topo da caixa (para empilhamento simples).
        factory.registerSimpleAction("ComputePoseToStore", [&](BT::TreeNode &self)
        {
            auto storagePose = self.getInput<geometry_msgs::msg::Pose>("storage_pose");
            auto storageSize = self.getInput<geometry_msgs::msg::Vector3>("storage_size");
            auto zLiftOffset = self.getInput<float>("z_lift_offset");

            if (!storagePose || !storageSize || !zLiftOffset) return BT::NodeStatus::FAILURE;

            geometry_msgs::msg::Pose output_final_pose = storagePose.value();
            output_final_pose.position.z += storageSize.value().z + zLiftOffset.value();

            self.setOutput("output_final_pose", output_final_pose);
            return BT::NodeStatus::SUCCESS;
        },
        {
            BT::InputPort<geometry_msgs::msg::Pose>("storage_pose"),
            BT::InputPort<geometry_msgs::msg::Vector3>("storage_size"),
            BT::InputPort<float>("z_lift_offset"),
            BT::OutputPort<geometry_msgs::msg::Pose>("output_final_pose")
        });
        // DOC-END: BT_ComputePoseToStore

        // DOC-START: BT_IncrementOrganizedStorageIndexes
        // --- Gerenciamento de Estoque ---
        factory.registerSimpleAction("IncrementOrganizedStorageIndexes", [&](BT::TreeNode &self)
        {
            // Persiste a ocupação do espaço no banco de dados
            auto id_opt = self.getInput<std::string>("storage_id");
            auto newIndexes = self.getInput<std::vector<int>>("new_indexes");
            if (!id_opt || !newIndexes) return BT::NodeStatus::FAILURE;

            storage_node_->addNewIndexes(id_opt.value(), newIndexes.value());
            RCLCPP_WARN(this->get_logger(), "Storage '%s' atualizado.", id_opt.value().c_str());
            return BT::NodeStatus::SUCCESS;
        },
        { BT::InputPort<std::string>("storage_id"), BT::InputPort<std::vector<int>>("new_indexes") });
        // DOC-END: BT_IncrementOrganizedStorageIndexes

        // DOC-START: BT_DecrementStorageCount
        factory.registerSimpleAction("DecrementStorageCount", [&](BT::TreeNode &self)
        {
            // Rollback: Libera o espaço se algo der errado na manipulação
            auto id_opt = self.getInput<std::string>("storage_id");
            if (!id_opt) return BT::NodeStatus::FAILURE;

            storage_node_->incrementStorageCount(id_opt.value(), -1);
            RCLCPP_WARN(this->get_logger(), "ROLLBACK: Espaço liberado em '%s'.", id_opt.value().c_str());
            return BT::NodeStatus::SUCCESS;
        },
        { BT::InputPort<std::string>("storage_id") });
        // DOC-END: BT_DecrementStorageCount

        // DOC-START: BT_IsGripperHoldingObject
        // --- Condition: IsGripperHoldingObject ---
        // Verifica sensor físico da garra (carga/contato).
        factory.registerSimpleCondition("IsGripperHoldingObject",
            [this](BT::TreeNode& self) -> BT::NodeStatus
            {
                std::lock_guard<std::mutex> lock(bt_mutex_); 
                if (this->gripper_monitor_node_->checkIsHolding()) 
                {
                    return BT::NodeStatus::SUCCESS;
                }    
                else
                {
                    // Se perdeu o objeto, para o robô imediatamente!
                    cancel_controller_goal();
                    return BT::NodeStatus::FAILURE;
                }
            }
        );
        // DOC-END: BT_IsGripperHoldingObject

        // DOC-START: BT_ComputePath
        // --- Action: ComputePath (Assíncrona) ---
        // Envia requisição para o planejador de caminho global (A* / D*).
        BT::NodeBuilder builder_compute = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                // 1. Monitoramento de Estado
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    
                    // Se já terminou (Sucesso ou Falha), retorna e reseta para IDLE
                    if (path_state_ == TaskState::SUCCESS) { 
                        path_state_ = TaskState::IDLE; 
                        return BT::NodeStatus::SUCCESS; 
                    }
                    if (path_state_ == TaskState::FAILURE) { 
                        path_state_ = TaskState::IDLE; 
                        return BT::NodeStatus::FAILURE; 
                    }
                    // Se já está rodando, continua retornando RUNNING
                    if (path_state_ == TaskState::RUNNING) return BT::NodeStatus::RUNNING;
                }

                // 2. Se está IDLE, inicia o processo
                auto target = self.getInput<geometry_msgs::msg::Pose>("target");
                if (!target) 
                {
                    RCLCPP_ERROR(this->get_logger(), "ComputePath: Target inválido na Blackboard.");
                    
                    rclcpp::sleep_for(std::chrono::milliseconds(2000)); 

                    return BT::NodeStatus::FAILURE;
                }

                // Tenta enviar o goal
                this->send_path_goal(target.value());

                // Define estado como RUNNING
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    path_state_ = TaskState::RUNNING;
                }
                
                return BT::NodeStatus::RUNNING;
            });
        };
        factory.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "ComputePath", { BT::InputPort<geometry_msgs::msg::Pose>("target"), BT::InputPort<std::string>("planner") }, {} }, builder_compute);
        // DOC-END: BT_ComputePath

        // DOC-START: BT_FollowPath
        // --- Action: FollowPath (Assíncrona) ---
        // Envia o caminho calculado para o controlador local (Pure Pursuit).
        factory.registerBuilder<AsyncAction>("FollowPath", [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, 
           
                [&](BT::TreeNode &self)
                {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        if (nav_state_ == TaskState::SUCCESS) { nav_state_ = TaskState::IDLE; return BT::NodeStatus::SUCCESS; }
                        if (nav_state_ == TaskState::FAILURE) { nav_state_ = TaskState::IDLE; return BT::NodeStatus::FAILURE; }
                        if (nav_state_ == TaskState::RUNNING) return BT::NodeStatus::RUNNING;
                    }

                    nav_msgs::msg::Path path_to_send;
                    bool has_path = false;
                    
                    {
                        std::lock_guard<std::mutex> lock(path_mutex_);
                        if (!last_calculated_path_.poses.empty())
                        {
                            path_to_send = last_calculated_path_;
                            has_path = true;
                        }
                    }

                    if (has_path)
                    {
                        if(this->send_controller_goal(path_to_send))
                        {
                            std::lock_guard<std::mutex> lock(state_mutex_);
                            nav_state_ = TaskState::RUNNING;
                            return BT::NodeStatus::RUNNING;
                        }
                        else 
                        {
                            RCLCPP_ERROR(this->get_logger(), "FollowPath: Falha ao enviar goal.");
                            return BT::NodeStatus::FAILURE;
                        }
                    }
                    else 
                    {
                        return BT::NodeStatus::FAILURE; 
                    }
                },
                
                
                [&](BT::TreeNode &self)
                {
                    RCLCPP_WARN(this->get_logger(), "FollowPath: HALT recebido! Cancelando Action...");
                    
                    // 1. Cancela a Action do ROS
                    this->cancel_controller_goal();

                    // 2. Reseta o estado interno para IDLE para permitir nova execução futura
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    nav_state_ = TaskState::IDLE;
                }
            );
        });
        // DOC-END: BT_FollowPath


        // DOC-START: BT_PickObject
        // --- Action: PickObject ---
        // Envia comando para o braço pegar o objeto.
        BT::NodeBuilder builder_pick = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE)
                {
                    auto object_pose = self.getInput<geometry_msgs::msg::Pose>("object_pose");
                    auto object_size = self.getInput<geometry_msgs::msg::Vector3>("object_size");
                    auto id = self.getInput<std::string>("id");

                    if (!object_pose || !id || !object_size) 
                    {
                        RCLCPP_ERROR(this->get_logger(), "PickObject: Parâmetros faltando");
                        return BT::NodeStatus::FAILURE;
                    }

                    geometry_msgs::msg::Pose target = object_pose.value();
                    geometry_msgs::msg::Vector3 target_size = object_size.value();

                    cached_object_.id = id.value();
                    cached_object_.pose = target;
                    cached_object_.size = target_size;
            
                    if(use_graspnet == true)
                    {
                        
                        target_size.x += 0.005;
                        target_size.y += 0.005;
                        target_size.z += 0.005;

                        this->cloud_box_filter_node_->set_bounding_box(target, target_size);

                        rclcpp::sleep_for(std::chrono::milliseconds(1000));
                        std::vector<geometry_msgs::msg::Pose> result;

                        if (this->cloud_box_filter_node_->has_points()) 
                        {
                            pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_points = this->cloud_box_filter_node_->get_filtered_points();
                            result = this->bridge_to_inference_node_->process_point_cloud(filtered_points);
                            RCLCPP_INFO(get_logger(), "Recebidos %zu grasps", result.size());
                        }
                        else
                        {
                            RCLCPP_WARN(get_logger(), "Sem pontos para grasp");
                            return BT::NodeStatus::FAILURE;
                        }

                        if (result.empty())
                        {
                            RCLCPP_ERROR(get_logger(), "Nenhum grasp válido encontrado");
                            return BT::NodeStatus::FAILURE;
                        }

                        this->send_goal(id.value(), result[0], true);
                        manipulation_state_ = TaskState::RUNNING;
                        
                    }
                    else
                    {
                        this->send_goal(id.value(), object_pose.value(), true);
                        manipulation_state_ = TaskState::RUNNING; 
                    }

                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory.registerBuilder(BT::TreeNodeManifest{
            BT::NodeType::ACTION, "PickObject", 
            { 
                BT::InputPort<geometry_msgs::msg::Pose>("object_pose"), 
                BT::InputPort<geometry_msgs::msg::Vector3>("object_size"), 
                BT::InputPort<std::string>("id") 
            }, 
            {} 
        }, builder_pick);
        // DOC-END: BT_PickObject

        // DOC-START: BT_PlaceObject
        // --- Action: PlaceObject ---
        // Envia comando para o braço largar o objeto.
        BT::NodeBuilder builder_place = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE)
                {
                    auto pose = self.getInput<geometry_msgs::msg::Pose>("pose");
                    if (!pose) return BT::NodeStatus::FAILURE;

                    std::string id_dummy = cached_object_.id;
                    this->send_goal(id_dummy, pose.value(), false); // false = Place
                    manipulation_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "PlaceObject", { BT::InputPort<geometry_msgs::msg::Pose>("pose"), BT::InputPort<std::vector<double>>("limits") }, {} }, builder_place);
        // DOC-END: BT_PlaceObject
        
        // Inicialização do Groot2 para visualização remota
        try
        {
            bt_tree_ = factory.createTreeFromFile(xml_path);
            groot_publisher_ = std::make_unique<BT::Groot2Publisher>(bt_tree_, 1666);
            RCLCPP_INFO(this->get_logger(), "Groot 2 Publisher iniciado na porta 1666");
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Erro Fatal ao criar Tree: %s", e.what());
        }
    }
    // DOC-END: setup_behavior_tree

    // DOC-START: bt_loop
    // Loop principal da thread da Behavior Tree.
    // Roda a 50Hz, verifica novos objetos e chama tick() da árvore.
    void bt_loop()
    {
        rclcpp::Rate rate(50);
        while (rclcpp::ok())
        {
            if (!bt_tree_.rootNode())
            {
                rate.sleep();
                continue;
            }

            BT::NodeStatus status = bt_tree_.rootNode()->status();

            // Verifica se chegou um objeto novo protegido por mutex
            bool new_obj = false;
            {
                std::lock_guard<std::mutex> lock(bt_mutex_);
                new_obj = has_new_object_;
            }

            // Condição de Gatilho:
            // Roda se a árvore já está rodando, se tem objeto novo ou se já tem um alvo fixo.
            if (status == BT::NodeStatus::RUNNING || new_obj || !current_target_id_.empty())
            {
                BT::NodeStatus result = bt_tree_.tickOnce();

                // Se a árvore terminar (sucesso ou falha total):
                if (result == BT::NodeStatus::SUCCESS || result == BT::NodeStatus::FAILURE)
                {
                    std::lock_guard<std::mutex> lock(bt_mutex_);
                    has_new_object_ = false;

                    // Se falhou, libera o ID para tentar de novo no futuro
                    if (result == BT::NodeStatus::FAILURE)
                    {
                         picked.erase(cached_object_.id); 
                         current_target_id_ = "";
                    }

                    // Reset geral de estados
                    {
                        std::lock_guard<std::mutex> slock(state_mutex_);
                        path_state_ = TaskState::IDLE;
                    }
                    nav_state_ = TaskState::IDLE;
                    manipulation_state_ = TaskState::IDLE;
                }
            }
            rate.sleep();
        }
    }
    // DOC-END: bt_loop

    // DOC-START: loadLocationsFromYaml
    // Carrega do YAML a lista de classes de objetos (labels) que o robô está autorizado a pegar.
    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);
            for (const auto &label_node : config) {
                authorized_labels.insert(label_node.first.as<std::string>());
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load YAML: %s", e.what());
        }
    }
    // DOC-END: loadLocationsFromYaml

    // DOC-START: odom_callback
    // Callback de Odometria: Atualiza a posição (x, y) do robô.
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex>lock(odom_mutex);

        pose_x = msg->pose.pose.position.x;
        pose_y = msg->pose.pose.position.y;
        pose_z = 0.0; // Assume robô em plano 2D
    }
    // DOC-END: odom_callback

    // DOC-START: detection_callback
    // Callback de Visão (YOLO/Depth).
    // Filtra detecções, verifica se o objeto é autorizado e se é novo.
    void detection_callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(bt_mutex_);

        // 1. Modo de Rastreamento: Se já temos um alvo, apenas atualiza a posição dele.
        if (!current_target_id_.empty() || has_new_object_)
        {
            for (const auto &det : msg->detections)
            {
                if (det.results.empty()) continue;
                std::string raw_id = det.results[0].hypothesis.class_id;

                if (raw_id == current_target_id_)
                {
                    current_target_pose_.position = det.bbox.center.position;
                    current_target_pose_.orientation = det.bbox.center.orientation;
                    cached_object_.pose = current_target_pose_;
                    cached_object_.size = det.bbox.size;
                    return;
                }
            }
            return;
        }

        // 2. Modo de Busca: Procura um novo objeto válido.
        for (const auto &det : msg->detections)
        {
            if (det.results.empty()) continue;

            std::string raw_id = det.results[0].hypothesis.class_id;
            std::string label = raw_id;
            size_t pos = raw_id.find('_');
            if (pos != std::string::npos) label = raw_id.substr(0, pos);

            // Verifica lista de autorização e lista de 'já pegos'
            if (authorized_labels.find(label) == authorized_labels.end()) continue;
            if (picked.find(raw_id) != picked.end()) continue;

            // Novo objeto encontrado!
            geometry_msgs::msg::Pose pose;
            pose.position = det.bbox.center.position;
            pose.orientation = det.bbox.center.orientation;

            cached_object_.id = raw_id;
            cached_object_.pose = pose;
            cached_object_.size = det.bbox.size;
            has_new_object_ = true; // Acorda a Behavior Tree

            RCLCPP_INFO(this->get_logger(), "Nova detecção salva: '%s'", raw_id.c_str());
            break;
        }
    }
    // DOC-END: detection_callback


    // DOC-START: cancel_controller_goal
    // Cancela o movimento do robô se necessário (ex: recálculo de rota).
    void cancel_controller_goal()
    {
        if (this->active_controller_goal_handle_)
        {
            RCLCPP_WARN(this->get_logger(), "Solicitando PARADA IMEDIATA...");
            this->controller_client->async_cancel_goal(this->active_controller_goal_handle_);
        }
    }
    // DOC-END: cancel_controller_goal
    
    // DOC-START: send_path_goal
    void send_path_goal(const geometry_msgs::msg::Pose & target_pose)
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            this->active_path_goal_handle_.reset();
        }

        if (!this->path_client->wait_for_action_server(std::chrono::seconds(2)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server 'path' indisponível! Abortando envio.");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            last_calculated_path_.poses.clear();
            last_no_filter_calculated_path_.poses.clear();
        }

        auto goal_msg = mobile_manipulation_interfaces::action::Path::Goal();
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "Enviando solicitação de Path Planning...");

        // 5. Configura as opções (Apenas Response e Result, SEM Feedback)
        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SendGoalOptions();
        
        send_goal_options.goal_response_callback = 
            std::bind(&ServerNode::path_goal_response_callback, this, std::placeholders::_1);
        
        send_goal_options.result_callback = 
            std::bind(&ServerNode::path_result_callback, this, std::placeholders::_1);

        // Envia de forma assíncrona
        this->path_client->async_send_goal(goal_msg, send_goal_options);
    }
    // DOC-END: send_path_goal


    // DOC-START: path_goal_response_callback
    // Apenas confirma se o servidor aceitou processar o pedido
    void path_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>> & goal_handle)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (!goal_handle) 
        {
            RCLCPP_ERROR(this->get_logger(), "O servidor REJEITOU o pedido de Path Planning.");
        } 
        else 
        {
            this->active_path_goal_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Pedido aceito pelo servidor. Calculando...");
        }
    }
    // DOC-END: path_goal_response_callback


    // DOC-START: path_result_callback
    // Aqui é onde o Caminho (Path) chega quando o cálculo termina
    void path_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::WrappedResult & result)
    {
        std::lock_guard<std::mutex> lock(state_mutex_); // Protege a transição de estado

        // Verifica se o handle é do goal ativo
        if (!this->active_path_goal_handle_ || result.goal_id != this->active_path_goal_handle_->get_goal_id()) {
            return;
        }
        this->active_path_goal_handle_.reset();

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            // Verifica se o caminho retornado é válido e não vazio
            if (result.result->success && !result.result->path.poses.empty())
            {
                std::lock_guard<std::mutex> p_lock(path_mutex_);
                this->last_calculated_path_ = result.result->path;
                this->last_no_filter_calculated_path_ = result.result->path_without_filter;
                RCLCPP_INFO(this->get_logger(), "Path Calculation: SUCCESS (%zu poses)", this->last_calculated_path_.poses.size());
                path_state_ = TaskState::SUCCESS; // Sinaliza sucesso para a BT
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Path Calculation: Server retornou true, mas o caminho está VAZIO.");
                path_state_ = TaskState::FAILURE;
            }
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Path Calculation: ABORTED/CANCELED");
            path_state_ = TaskState::FAILURE;
        }
    }
    // DOC-END: path_result_callback


    // DOC-START: send_controller_goal
    // Envia o caminho (Path) para o controlador local seguir.
    bool send_controller_goal(const nav_msgs::msg::Path &target_path)
    {
        if (!this->controller_client->wait_for_action_server(std::chrono::seconds(2))) 
        {
            RCLCPP_ERROR(this->get_logger(), "Action server 'controller' not available");
            return false;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::Controller::Goal();
        goal_msg.path = target_path;

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SendGoalOptions();

        send_goal_options.goal_response_callback = std::bind(&ServerNode::controller_goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::controller_result_callback, this, std::placeholders::_1);

        this->controller_client->async_send_goal(goal_msg, send_goal_options);
        return true;
    }
    // DOC-END: send_controller_goal

    // DOC-START: controller_goal_response_callback
    void controller_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>> & goal_handle)
    {
        if (!goal_handle) 
        {
            RCLCPP_ERROR(this->get_logger(), "Goal CONTROLLER rejeitado");
            nav_state_ = TaskState::FAILURE;
        } 
        else 
        {
            this->active_controller_goal_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Goal CONTROLLER aceito.");
        }
    }
    // DOC-END: controller_goal_response_callback

    // DOC-START: controller_result_callback
    // Resultado da navegação (Chegou ou Falhou).
    void controller_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::WrappedResult & result)
    {
        std::lock_guard<std::mutex> s_lock(state_mutex_);

        // Verifica se o resultado pertence ao goal atual
        if (this->active_controller_goal_handle_ && result.goal_id != this->active_controller_goal_handle_->get_goal_id()) 
        {
            return; // Resultado antigo, ignora
        }
        this->active_controller_goal_handle_.reset();

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            {
                std::lock_guard<std::mutex> p_lock(path_mutex_);
                last_calculated_path_.poses.clear(); 
            }
            RCLCPP_INFO(this->get_logger(), "Controller: Chegou ao destino.");
            nav_state_ = TaskState::SUCCESS;
        }
        else if (result.code == rclcpp_action::ResultCode::CANCELED)
        {
            RCLCPP_WARN(this->get_logger(), "Controller: Cancelado.");
            nav_state_ = TaskState::IDLE; // Reset suave
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Controller: FALHOU (Aborted).");
            nav_state_ = TaskState::FAILURE;
        }
    }
    // DOC-END: controller_result_callback

    // DOC-START: send_goal
    // Envia Action de Manipulação (Pick ou Place).
    void send_goal(const std::string id, const geometry_msgs::msg::Pose & target_pose, bool pick)
    {
        if (!this->client_ptr_->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server manipulação not available");
            manipulation_state_ = TaskState::FAILURE;
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::PickObject::Goal();
        goal_msg.obstacle_id = id;
        goal_msg.pick = pick;
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "BT: Enviando Goal para MANIPULATION...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SendGoalOptions();
        send_goal_options.goal_response_callback = std::bind(&ServerNode::goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::result_callback, this, std::placeholders::_1);

        this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
    }
    // DOC-END: send_goal

    // DOC-START: goal_response_callback
    void goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>> & goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal PICK rejeitado");
            manipulation_state_ = TaskState::FAILURE;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Goal PICK aceito.");
        }
    }
    // DOC-END: goal_response_callback

    // DOC-START: result_callback
    void result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success)
        {
            manipulation_state_ = TaskState::SUCCESS;
            RCLCPP_INFO(this->get_logger(), "PICK SUCCESS");
        }
        else
        {
            manipulation_state_ = TaskState::FAILURE;
            RCLCPP_ERROR(this->get_logger(), "PICK FAILED");
        }
    }
    // DOC-END: result_callback


    // DOC-START: publish_pose
    void publish_pose()
    {
        auto message = geometry_msgs::msg::Pose();
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            message = cached_object_.pose;
        }
        publisher_->publish(message);
    }
    // DOC-END: publish_pose

};

// DOC-START: has_flag
// Utilitário para verificar flags de terminal (ex: --no-gripper)
bool has_flag(const std::vector<std::string>& args, const std::string& flag) 
{
    return std::find(args.begin(), args.end(), flag) != args.end();
}
// DOC-END: has_flag

// DOC-START: main
// Função Principal: Inicializa ROS, nós auxiliares e o ServerNode.
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    std::vector<std::string> args(argv, argv + argc);

    // Flags para controle modular
    bool enable_organize     = !has_flag(args, "--no-organize");
    bool enable_storage      = !has_flag(args, "--no-storage");
    bool enable_gripper      = !has_flag(args, "--no-gripper");

    // Configuração dos nós auxiliares (remapeamento de nomes)
    rclcpp::NodeOptions organize_opts;
    organize_opts.arguments({"--ros-args", "-r", "__node:=organize_node"});

    rclcpp::NodeOptions storage_opts;
    storage_opts.arguments({"--ros-args", "-r", "__node:=storage_node"});

    rclcpp::NodeOptions gripper_opts;
    gripper_opts.arguments({"--ros-args", "-r", "__node:=gripper_monitor_node"});
    
    rclcpp::NodeOptions reachability_opts;
    reachability_opts.arguments({"--ros-args", "-r", "__node:=reachability_node"});

    rclcpp::NodeOptions obstacle_graph_opts;
    obstacle_graph_opts.arguments({"--ros-args", "-r", "__node:=shared_obstacle_graph_node"});

    rclcpp::NodeOptions ik_validator_opts;
    ik_validator_opts.arguments({"--ros-args", "-r", "__node:=ik_validator_node"});

    rclcpp::NodeOptions cloud_box_filter_opts;
    cloud_box_filter_opts.arguments({"--ros-args", "-r", "__node:=cloud_box_filter"});

    rclcpp::NodeOptions bridge_to_inference_opts;
    bridge_to_inference_opts.arguments({"--ros-args", "-r", "__node:=bridge_to_inference"});

    std::shared_ptr<storage_manager::OrganizeNode> organize_node = nullptr;
    std::shared_ptr<storage_manager::StorageNode> storage_node   = nullptr;

    std::shared_ptr<manipulation::IsGripperHolding> gripper_node = nullptr;
    std::shared_ptr<manipulation::ProjectedReachabilityAnalysis> reachability_node = nullptr; 
    std::shared_ptr<manipulation::IKValidator> ik_validator_node = nullptr; 
    std::shared_ptr<manipulation::CloudBoxFilter> cloud_box_filter_node = nullptr; 

    std::shared_ptr<navigation::SharedObstacleGraph> obstacle_graph_node = nullptr; 

    std::shared_ptr<drl_to_pick_cpp::BridgeToInference> bridge_to_inference_node = nullptr; 


    rclcpp::executors::MultiThreadedExecutor executor;

    if (enable_organize)
    {
        organize_node = std::make_shared<storage_manager::OrganizeNode>(organize_opts);
        executor.add_node(organize_node);
    }

    if (enable_storage)
    {
        storage_node = std::make_shared<storage_manager::StorageNode>(storage_opts);
        executor.add_node(storage_node);
    }

    if (enable_gripper)
    {
        gripper_node = std::make_shared<manipulation::IsGripperHolding>(gripper_opts);
        executor.add_node(gripper_node);
    }

  
    reachability_node = std::make_shared<manipulation::ProjectedReachabilityAnalysis>(reachability_opts);
    executor.add_node(reachability_node);

    obstacle_graph_node = std::make_shared<navigation::SharedObstacleGraph>(obstacle_graph_opts);
    executor.add_node(obstacle_graph_node);

    ik_validator_node = std::make_shared<manipulation::IKValidator>(ik_validator_opts);
    executor.add_node(ik_validator_node);

    cloud_box_filter_node = std::make_shared<manipulation::CloudBoxFilter>(cloud_box_filter_opts);
    executor.add_node(cloud_box_filter_node);

    bridge_to_inference_node = std::make_shared<drl_to_pick_cpp::BridgeToInference>(bridge_to_inference_opts);
    executor.add_node(bridge_to_inference_node);

    auto server_node = std::make_shared<ServerNode>(gripper_node, storage_node, organize_node, reachability_node, 
        obstacle_graph_node, ik_validator_node, cloud_box_filter_node, bridge_to_inference_node);

    executor.add_node(server_node);

    executor.spin();

    rclcpp::shutdown();
    return 0;
}
// DOC-END: main