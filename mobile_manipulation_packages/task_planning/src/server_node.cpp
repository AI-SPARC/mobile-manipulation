/**
 * @file server_node.cpp
 * @brief Nó central de controle para manipulação móvel utilizando Behavior Trees e ROS 2.
 * @author Lucas Momesso Alves
 * @date 06/12/2025
 *
 * @details Este arquivo implementa o `ServerNode`, que atua como o cérebro do robô.
 * Ele recebe a parte de visão do ambiente e comanda a execução da navegação (Path Planning e Controle),
 * manipulação (Pick and Place) e gerenciamento de estoque (Storage Manager).
 *
 * A lógica de decisão é governada pela biblioteca BehaviorTree.CPP v4.
 * O nó gerencia a conversão de callbacks assíncronos do ROS 2 (Actions/Topics)
 * para o fluxo síncrono da Behavior Tree (Ticks).
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
#include <storage_manager/GetStorageInfo.hpp>
#include <storage_manager/Organize.hpp>

namespace BT
{
    /**
     * @brief Especialização de template necessária para o BT.CPP converter strings para geometry_msgs::msg::Pose.
     *
     * @details O BehaviorTree.CPP precisa saber como converter tipos de portas XML para tipos C++.
     * Como geometry_msgs::msg::Pose é um tipo complexo, fornecemos uma implementação padrão.
     *
     * @param str String de entrada (não utilizada nesta implementação básica).
     * @return geometry_msgs::msg::Pose Uma pose vazia/zerada por padrão.
     */
    template <>
    inline geometry_msgs::msg::Pose convertFromString(StringView)
    {
        return geometry_msgs::msg::Pose();
    }
}

/**
 * @enum TaskState
 * @brief Enumeração para rastrear o estado de execução de ações assíncronas do ROS.
 *
 * @details Usado para fazer a ponte entre o loop de feedback do ROS (Callbacks) e o tick da Behavior Tree.
 */
enum class TaskState
{
    IDLE,    /**< @brief Nenhuma tarefa está sendo executada ou a tarefa anterior foi resetada. */
    RUNNING, /**< @brief A Action do ROS foi enviada e o servidor está processando. */
    SUCCESS, /**< @brief A Action do ROS retornou sucesso no result callback. */
    FAILURE  /**< @brief A Action falhou, foi abortada ou rejeitada. */
};

/**
 * @class ParallelAny
 * @brief Nó de Controle customizado para a Behavior Tree.
 *
 * @details Diferente dos nós `Parallel` padrão (que esperam N sucessos ou N falhas),
 * este nó implementa uma lógica "short-circuit". Ele retorna SUCESSO imediatamente se
 * *qualquer* filho retornar sucesso, ou FALHA se *qualquer* filho retornar falha.
 * Caso contrário, retorna RUNNING.
 */
class ParallelAny : public BT::ControlNode
{
public:
    /**
     * @brief Construtor do nó ParallelAny.
     * @param name Nome do nó na árvore.
     * @param config Configuração do nó.
     */
    ParallelAny(const std::string& name, const BT::NodeConfig& config)
        : BT::ControlNode(name, config) {}

    /**
     * @brief Define as portas fornecidas pelo nó.
     * @return BT::PortsList Lista vazia, pois este nó não possui portas de entrada/saída.
     */
    static BT::PortsList providedPorts() { return {}; }

    /**
     * @brief Executa o tick do nó.
     *
     * @details Itera sobre todos os filhos.
     * 1. Se um filho retorna SUCCESS -> Interrompe os outros e retorna SUCCESS.
     * 2. Se um filho retorna FAILURE -> Interrompe os outros e retorna FAILURE.
     * 3. Se nenhum retornar status terminal, retorna RUNNING.
     *
     * @return BT::NodeStatus O estado agregado dos filhos.
     */
    BT::NodeStatus tick() override
    {
        for (size_t i = 0; i < children_nodes_.size(); i++)
        {
            BT::TreeNode* child = children_nodes_[i];
            BT::NodeStatus status = child->executeTick();

            if (status == BT::NodeStatus::SUCCESS)
            {
                haltChildren();
                return BT::NodeStatus::SUCCESS;
            }

            if (status == BT::NodeStatus::FAILURE)
            {
                haltChildren();
                return BT::NodeStatus::FAILURE;
            }
        }
        return BT::NodeStatus::RUNNING;
    }

    /**
     * @brief Interrompe a execução do nó e de todos os seus filhos.
     */
    void halt() override
    {
        haltChildren();
        BT::ControlNode::halt();
    }
};

/**
 * @class AsyncAction
 * @brief Wrapper genérico para criar Actions assíncronas na Behavior Tree.
 *
 * @details Permite passar uma função lambda (`tick_fun`) que define o comportamento do nó
 * tanto na inicialização (`onStart`) quanto durante a execução (`onRunning`).
 * Isso simplifica a criação de nós Stateful sem precisar herdar classes complexas repetidamente.
 */
class AsyncAction : public BT::StatefulActionNode
{
public:
    /**
     * @brief Construtor do AsyncAction.
     * @param name Nome da Action.
     * @param config Configuração da BT.
     * @param tick_fun Função lambda que contém a lógica do Tick. Recebe uma referência ao próprio TreeNode.
     */
    AsyncAction(const std::string& name, const BT::NodeConfig& config,
                std::function<BT::NodeStatus(BT::TreeNode&)> tick_fun)
        : BT::StatefulActionNode(name, config), tick_fun_(tick_fun) {}

    /**
     * @brief Chamado quando o nó sai do estado IDLE.
     * @return BT::NodeStatus Resultado da execução da lambda.
     */
    BT::NodeStatus onStart() override { return tick_fun_(*this); }

    /**
     * @brief Chamado a cada tick enquanto o nó está no estado RUNNING.
     * @return BT::NodeStatus Resultado da execução da lambda.
     */
    BT::NodeStatus onRunning() override { return tick_fun_(*this); }

    /**
     * @brief Chamado quando o nó é interrompido (halt).
     */
    void onHalted() override {}

private:
    /** @brief Armazena a função lógica injetada via construtor. */
    std::function<BT::NodeStatus(BT::TreeNode&)> tick_fun_;
};

// ============================================================================
// CLASSE PRINCIPAL DO SERVER NODE
// ============================================================================

/**
 * @class ServerNode
 * @brief Nó principal (Main Node) do sistema de robótica.
 *
 * @details Este nó herda de `rclcpp::Node` e é responsável por:
 * 1. Assinar tópicos de odometria e visão computacional.
 * 2. Gerenciar clientes de Action para Path Planning, Controle e Manipulação.
 * 3. Carregar e executar a Behavior Tree.
 * 4. Manter o estado global do sistema (posição do robô, objeto detectado, estado das tarefas).
 */
class ServerNode : public rclcpp::Node
{
public:
    /**
     * @name Ciclo de Vida e Inicialização
     * Construtores, destrutores e carregamento de parâmetros.
     * @{
    */

    /**
     * @brief Construtor do ServerNode.
     *
     * @details Inicializa parâmetros, subscribers, clients e a thread da Behavior Tree.
     *
     * @param gripper_node Ponteiro compartilhado para o nó de monitoramento da garra.
     * @param storage_node Ponteiro compartilhado para o nó de gerenciamento de armazenamento.
     * @param organize_node Ponteiro compartilhado para o nó de lógica de organização (Bin Packing).
     */
    ServerNode(
        std::shared_ptr<manipulation::IsGripperHolding> gripper_node,
        std::shared_ptr<storage_manager::StorageNode> storage_node,
        std::shared_ptr<storage_manager::OrganizeNode> organize_node
    )
     : Node("server_node"),
       gripper_monitor_node_(gripper_node),
       storage_node_(storage_node),
       organize_node_(organize_node)
    {
        // Declaração e obtenção de parâmetros do ROS
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("bt_xml_path", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        std::string bt_xml_path = this->get_parameter("bt_xml_path").as_string();

        // Inicialização dos Subscribers
        // Detection3DArray: Recebe caixas delimitadoras (bounding boxes) dos objetos detectados.
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>("/bbox_3d_with_labels", 10,
            std::bind(&ServerNode::detection_callback, this, std::placeholders::_1));

        // Odometria: Recebe a posição atual do robô.
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/odom", 10, std::bind(&ServerNode::odom_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("object_pose", 10);
        
        // Inicialização dos Action Clients
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(this, "pick_object");
        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(this, "path");
        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(this, "controller");

        // Inicialização dos estados internos
        path_state_ = TaskState::IDLE;
        nav_state_ = TaskState::IDLE;
        manipulation_state_ = TaskState::IDLE;

        // Configuração da Árvore de Comportamento
        setup_behavior_tree(bt_xml_path);

        // Inicia a thread dedicada para o tick da Behavior Tree para não bloquear o executor do ROS
        bt_thread_ = std::thread(&ServerNode::bt_loop, this);

        RCLCPP_INFO(this->get_logger(), "ServerNode iniciado.");

        timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&ServerNode::publish_pose, this));

        if(!yaml_file.empty())
        {
            loadLocationsFromYaml(yaml_file);
        }
    }

    /**
     * @brief Destrutor do ServerNode.
     * @details Garante que a thread da Behavior Tree seja finalizada corretamente (join).
     */
    ~ServerNode()
    {
        if (bt_thread_.joinable()) bt_thread_.join();
    }

private:
    /**
     * @struct ObjectInfo
     * @brief Estrutura auxiliar para armazenar dados de um objeto detectado.
     */
    struct ObjectInfo
    {
        std::string id;                 /**< @brief ID único ou classe do objeto (ex: "garrafa_1"). */
        geometry_msgs::msg::Pose pose;  /**< @brief Pose 3D (posição e orientação) do objeto no mundo. */
        geometry_msgs::msg::Vector3 size; /**< @brief Dimensões da Bounding Box (x, y, z). */
    };

    // --- Membros de Ponteiros para Nós Auxiliares ---
    std::shared_ptr<manipulation::IsGripperHolding> gripper_monitor_node_; /**< Referência ao nó que verifica sensor de carga/garra. */
    std::shared_ptr<storage_manager::StorageNode> storage_node_;           /**< Referência ao banco de dados de posições de armazenamento. */
    std::shared_ptr<storage_manager::OrganizeNode> organize_node_;         /**< Referência à lógica de cálculo de posição dentro da caixa. */
    std::unique_ptr<BT::Groot2Publisher> groot_publisher_;                 /**< Publicador para visualização em tempo real no Groot2. */

    // Publishers.
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;

    // --- Subscribers ROS ---
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_; /**< Assinante de detecções visuais. */
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;       /**< Assinante de odometria. */

    // --- Action Clients ROS ---
    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_; /**< Cliente para ação de Pegar/Largar. */
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;       /**< Cliente para planejador de caminho (A* / Nav2). */
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client; /**< Cliente para controlador de trajetória (Pure Pursuit/MPC). */

    // --- Handles de Action Ativos ---
    // Usados para poder cancelar a ação em execução se necessário.
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::SharedPtr active_controller_goal_handle_;
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::SharedPtr active_path_goal_handle_;

    // --- Variáveis de Configuração e Estado de Objetos ---
    std::string yaml_file; /**< Caminho para o arquivo YAML de objetos autorizados. */
    std::unordered_set<std::string> authorized_labels; /**< Conjunto de labels de objetos que o robô tem permissão para pegar. */
    std::unordered_set<std::string> picked; /**< Conjunto de IDs de objetos que já foram pegos/processados para evitar loops. */

    // --- Estado do Alvo Atual ---
    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose; /**< (Não usado ativamente no código atual, possivelmente legado). */
    ObjectInfo cached_object_; /**< Armazena temporariamente o objeto detectado mais recente e válido. */

    std::string current_target_id_ = ""; /**< ID do objeto que está sendo ativamente processado pela Behavior Tree. Vazio se IDLE. */
    geometry_msgs::msg::Pose current_target_pose_; /**< Pose do objeto alvo atual. */

    rclcpp::TimerBase::SharedPtr timer_;

    // --- Infraestrutura da Behavior Tree ---
    std::thread bt_thread_; /**< Thread separada para o loop `tick` da árvore. */
    std::mutex bt_mutex_;   /**< Mutex para proteger leitura/escrita de variáveis compartilhadas entre thread ROS e thread BT (ex: current_target_id_). */
    BT::Tree bt_tree_;      /**< O objeto da árvore de comportamento carregada. */

    // --- Máquina de Estados e Sincronização ---
    TaskState path_state_;        /**< Estado atual da ação de planejamento de caminho. */
    TaskState nav_state_;         /**< Estado atual da ação do controlador (movimento). */
    TaskState manipulation_state_;/**< Estado atual da ação de manipulação (braço). */

    std::mutex state_mutex_; /**< Mutex CRÍTICO. Protege as transições de estado (path_state_, etc) para evitar condições de corrida entre callbacks e ticks. */
    std::mutex path_mutex_;  /**< Mutex para proteger o acesso à variável `last_calculated_path_`. */

    nav_msgs::msg::Path last_calculated_path_; /**< Armazena o último caminho recebido do planner antes de enviá-lo ao controlador. */

    // --- Odometria e Flags ---
    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0; /**< Posição atual do robô atualizada pelo odom_callback. */
    bool has_new_object_ = false; /**< Flag atômica (logicamente) indicando que o detection_callback encontrou um candidato válido. */

    /**
     * @name Utilitários Internos
     * Funções auxiliares para verificação de estado e carregamento de arquivos.
     * @{
     */

    /**
     * @brief Converte o enum TaskState interno para BT::NodeStatus.
     *
     * @details Também reseta o estado interno para IDLE se a tarefa tiver terminado (SUCCESS ou FAILURE),
     * preparando-o para a próxima execução.
     *
     * @param state Referência para a variável de estado (ex: nav_state_).
     * @return BT::NodeStatus O status correspondente para a Behavior Tree.
     */
    BT::NodeStatus check_task_status(TaskState &state)
    {
        if (state == TaskState::SUCCESS)
        {
            state = TaskState::IDLE;
            return BT::NodeStatus::SUCCESS;
        }
        else if (state == TaskState::FAILURE)
        {
            state = TaskState::IDLE;
            return BT::NodeStatus::FAILURE;
        }
        return BT::NodeStatus::RUNNING;
    }

    /** @} */

    /**
     * @name Núcleo da Behavior Tree
     * Configuração, registro de nós e loop principal de execução (Tick).
     * @{
     */

    /**
     * @brief Configura e registra todos os nós da Behavior Tree.
     * @details Esta função é responsável por mapear as strings do XML para a lógica C++.
     * Registra nós customizados (ParallelAny) e Actions/Conditions simples via lambdas.
     *
     * Nós registrados:
     * - **IsRobotNear**: Verifica distância euclidiana entre robô e alvo. Falha se longe e retorna pose de ajuste.
     * - **DetectObject**: Retorna SUCESSO se houver um alvo travado ou um novo objeto válido detectado. Preenche as portas de saída com dados do objeto.
     * - **ClearTarget**: Limpa o alvo atual (current_target_id_), permitindo buscar novos objetos.
     * - **GetStorageInfo**: Consulta o nó de Storage para encontrar a melhor caixa/prateleira para o objeto atual.
     * - **ComputePoseToOrganize**: Chama o algoritmo de bin-packing para calcular onde exatamente colocar o objeto dentro da caixa.
     * - **ComputePoseToStore**: Calcula pose simples de armazenamento.
     * - **IncrementOrganizedStorageIndexes**: Persiste os novos índices ocupados no banco de dados de armazenamento.
     * - **DecrementStorageCount**: Rollback em caso de falha, libera o espaço reservado.
     * - **IsGripperHoldingObject**: Verifica sensor de força/carga da garra.
     * - **ComputePath**: Action Assíncrona. Envia goal para o planejador de caminho.
     * - **NavigateTo**: Action Assíncrona. Envia o caminho calculado para o controlador.
     * - **PickObject**: Action Assíncrona. Controla o braço para pegar.
     * - **PlaceObject**: Action Assíncrona. Controla o braço para largar.
     *
     * @param xml_path Caminho absoluto para o arquivo XML da árvore.
     */
    void setup_behavior_tree(const std::string &xml_path)
    {
        BT::BehaviorTreeFactory factory;

        factory.registerNodeType<ParallelAny>("ParallelAny");

        // --- Condition: IsRobotNear ---
        factory.registerSimpleCondition("IsRobotNear", [&](BT::TreeNode &self)
        {
            auto target_pose_opt = self.getInput<geometry_msgs::msg::Pose>("target");
            if (!target_pose_opt) return BT::NodeStatus::FAILURE;
            geometry_msgs::msg::Pose target = target_pose_opt.value();

            auto max_dist_opt = self.getInput<double>("max_dist");
            auto min_dist_opt = self.getInput<double>("min_dist");

            double max_dist = max_dist_opt ? max_dist_opt.value() : 0.5;
            double min_dist = min_dist_opt ? min_dist_opt.value() : 0.35;

            double dx = this->pose_x - target.position.x;
            double dy = this->pose_y - target.position.y;
            double current_dist = std::sqrt(dx*dx + dy*dy);

            if (current_dist >= min_dist && current_dist <= max_dist)
            {
                return BT::NodeStatus::SUCCESS;
            }

            RCLCPP_WARN(this->get_logger(), "BT: Robô longe (%.2fm). Ajustando...", current_dist);
            // Saída para que o robô possa se mover para esta pose
            self.setOutput("adjustment_pose", target);
            return BT::NodeStatus::FAILURE;
        },
        {
            BT::InputPort<geometry_msgs::msg::Pose>("target"),
            BT::InputPort<double>("max_dist"),
            BT::InputPort<double>("min_dist"),
            BT::OutputPort<geometry_msgs::msg::Pose>("adjustment_pose")
        });

        // --- Action: DetectObject ---
        factory.registerSimpleAction("DetectObject", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_); // Protege current_target_id_ e cached_object_

            // Se já temos um alvo travado, retornamos ele (persistencia)
            if (!current_target_id_.empty())
            {
                self.setOutput("output_pose", current_target_pose_);
                self.setOutput("output_id", current_target_id_);
                self.setOutput("object_size", cached_object_.size);
                return BT::NodeStatus::SUCCESS;
            }

            // Se não temos alvo, verificamos se o callback de visão encontrou algo novo
            if (!has_new_object_)
            {
                return BT::NodeStatus::RUNNING;
            }

            // Promove o objeto cacheado para alvo atual
            current_target_id_ = cached_object_.id;
            current_target_pose_ = cached_object_.pose;

            self.setOutput("output_pose", current_target_pose_);
            self.setOutput("output_id", current_target_id_);
            self.setOutput("object_size", cached_object_.size);

            // Marca como 'pego' para não processar novamente o mesmo ID imediatamente
            picked.insert(current_target_id_);
            has_new_object_ = false;

            RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' processado.", current_target_id_.c_str());
            return BT::NodeStatus::SUCCESS;
        },
        {
            BT::OutputPort<geometry_msgs::msg::Pose>("output_pose"),
            BT::OutputPort<std::string>("output_id"),
            BT::OutputPort<geometry_msgs::msg::Vector3>("object_size")
        });


        // --- Action: ClearTarget ---
        factory.registerSimpleAction("ClearTarget", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' finalizado.", current_target_id_.c_str());
            current_target_id_ = ""; // Libera o sistema para pegar o próximo objeto
            return BT::NodeStatus::SUCCESS;
        });

        // --- Action: GetStorageInfo ---
        factory.registerSimpleAction("GetStorageInfo", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("object_id");
            if (!id_opt) return BT::NodeStatus::FAILURE;

            // Lógica para extrair label genérico do ID (ex: "cola_123" -> "cola")
            std::string full_id = id_opt.value();
            std::string label = full_id;
            size_t pos = full_id.find('_');
            if (pos != std::string::npos) label = full_id.substr(0, pos);

            geometry_msgs::msg::Pose current_obj_pose;
            {
                std::lock_guard<std::mutex> lock(bt_mutex_);
                current_obj_pose = current_target_pose_;
            }

            // Chama o Storage Node para achar onde guardar
            auto result = storage_node_->getBestStorage(label, current_obj_pose);

            if (result.success)
            {
                self.setOutput("storage_pose", result.pose); // Pose da caixa/estante
                self.setOutput("storage_limits", result.limits);
                self.setOutput("storage_id", result.storage_name);
                self.setOutput("indexes", result.indexes); // Índices de grade ocupados atualmente
                self.setOutput("storage_size", result.size);
                return BT::NodeStatus::SUCCESS;
            }

            RCLCPP_WARN(this->get_logger(), "Falha ao encontrar storage para %s", label.c_str());
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

        // --- Action: ComputePoseToOrganize ---
        factory.registerSimpleAction("ComputePoseToOrganize", [&](BT::TreeNode &self)
        {
            // Obtém todos os parâmetros geométricos necessários para o bin packing
            auto storagePose = self.getInput<geometry_msgs::msg::Pose>("storage_pose");
            auto storageSize = self.getInput<geometry_msgs::msg::Vector3>("storage_size");
            auto objectSize = self.getInput<geometry_msgs::msg::Vector3>("object_size");
            auto indexes = self.getInput<std::vector<int>>("indexes");
            auto objectPadding = self.getInput<float>("object_padding");
            auto zLiftOffset = self.getInput<float>("z_lift_offset");

            if (!storagePose || !storageSize || !objectSize || !indexes || !objectPadding || !zLiftOffset)
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: Parâmetros faltando em ComputePoseToOrganize.");
                return BT::NodeStatus::FAILURE;
            }

            std::vector<int> idx_vec = indexes.value();
            if (idx_vec.size() != 3) return BT::NodeStatus::FAILURE;

            // Calcula a posição exata (x,y,z) dentro da caixa usando a grade (i,j,k)
            std::pair<geometry_msgs::msg::Pose, std::vector<int>> result = organize_node_->placeObjectInBox(
                storagePose.value(),
                storageSize.value(),
                objectSize.value(),
                objectPadding.value(),
                zLiftOffset.value(),
                idx_vec[0], idx_vec[1], idx_vec[2]
            );

            self.setOutput("output_final_pose", std::get<0>(result)); // Pose final de place
            self.setOutput("new_indexes", std::get<1>(result)); // Próximos índices livres

            RCLCPP_INFO(this->get_logger(), "Indexes atualizados. Novo IDX: [%d, %d, %d]",
                std::get<1>(result)[0], std::get<1>(result)[1], std::get<1>(result)[2]);

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

        // --- Action: ComputePoseToStore (Simplificada) ---
        factory.registerSimpleAction("ComputePoseToStore", [&](BT::TreeNode &self)
        {
            auto storagePose = self.getInput<geometry_msgs::msg::Pose>("storage_pose");
            auto storageSize = self.getInput<geometry_msgs::msg::Vector3>("storage_size");
            auto zLiftOffset = self.getInput<float>("z_lift_offset");

            if (!storagePose || !storageSize || !zLiftOffset) return BT::NodeStatus::FAILURE;

            geometry_msgs::msg::Pose output_final_pose = storagePose.value();
            // Simplesmente coloca no topo da caixa + offset
            output_final_pose.position.z = output_final_pose.position.z + storageSize.value().z + zLiftOffset.value();

            self.setOutput("output_final_pose", output_final_pose);
            return BT::NodeStatus::SUCCESS;
        },
        {
            BT::InputPort<geometry_msgs::msg::Pose>("storage_pose"),
            BT::InputPort<geometry_msgs::msg::Vector3>("storage_size"),
            BT::InputPort<float>("z_lift_offset"),
            BT::OutputPort<geometry_msgs::msg::Pose>("output_final_pose")
        });


        // --- Action: IncrementOrganizedStorageIndexes ---
        factory.registerSimpleAction("IncrementOrganizedStorageIndexes", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("storage_id");
            auto newIndexes = self.getInput<std::vector<int>>("new_indexes");
            if (!id_opt || !newIndexes) return BT::NodeStatus::FAILURE;

            // Atualiza o banco de dados do storage para marcar o espaço como ocupado
            storage_node_->addNewIndexes(id_opt.value(), newIndexes.value());
            RCLCPP_WARN(this->get_logger(), "Adicionando novos índices ocupados no armazém '%s'.", id_opt.value().c_str());
            return BT::NodeStatus::SUCCESS;
        },
        { BT::InputPort<std::string>("storage_id"), BT::InputPort<std::vector<int>>("new_indexes") });

        // --- Action: DecrementStorageCount ---
        factory.registerSimpleAction("DecrementStorageCount", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("storage_id");
            if (!id_opt) return BT::NodeStatus::FAILURE;

            // Libera o espaço se a operação falhar (Rollback)
            storage_node_->incrementStorageCount(id_opt.value(), -1);
            RCLCPP_WARN(this->get_logger(), "ROLLBACK: Liberando vaga no storage '%s' devido a falha.", id_opt.value().c_str());
            return BT::NodeStatus::SUCCESS;
        },
        { BT::InputPort<std::string>("storage_id") });


        // --- Condition: IsGripperHoldingObject ---
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
                    cancel_controller_goal();
                    return BT::NodeStatus::FAILURE;
                }
            }
        );

        // --- Action Builder: ComputePath (Async) ---
        // Utiliza AsyncAction e mutexes cuidadosamente para evitar Deadlocks entre a thread BT e callbacks ROS.
        BT::NodeBuilder builder_compute = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                // 1. Verificação inicial rápida COM lock para checar estado anterior
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);

                    if (self.status() == BT::NodeStatus::IDLE && path_state_ != TaskState::IDLE) 
                    {
                        path_state_ = TaskState::IDLE; // Reset se reiniciado
                    }

                    if (path_state_ == TaskState::SUCCESS) 
                    {
                        path_state_ = TaskState::IDLE;
                        return BT::NodeStatus::SUCCESS;
                    }
                    if (path_state_ == TaskState::FAILURE) 
                    {
                        path_state_ = TaskState::IDLE;
                        return BT::NodeStatus::FAILURE;
                    }

                    if (path_state_ == TaskState::RUNNING) 
                    {
                        return BT::NodeStatus::RUNNING;
                    }
                } // <--- IMPORTANTE: LIBERA O MUTEX AQUI

                // 2. Envio do Goal.
                // Fazemos isso SEM segurar o mutex, para não travar o ROS Executor no wait_for_server dentro de send_path_goal
                auto target = self.getInput<geometry_msgs::msg::Pose>("target");
                if (!target) return BT::NodeStatus::FAILURE;

                this->send_path_goal(target.value());

                // 3. Atualiza estado para RUNNING (pegando o lock novamente para segurança)
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    path_state_ = TaskState::RUNNING;
                }

                return BT::NodeStatus::RUNNING;
            });
        };
        factory.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "ComputePath", { BT::InputPort<geometry_msgs::msg::Pose>("target"), BT::InputPort<std::string>("planner") }, {} }, builder_compute);


        // --- Action Builder: NavigateTo (Async) ---
        factory.registerBuilder<AsyncAction>("NavigateTo", [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (self.status() == BT::NodeStatus::IDLE && nav_state_ != TaskState::IDLE)
                {
                    nav_state_ = TaskState::IDLE;
                }

                if (nav_state_ == TaskState::IDLE )
                {
                    nav_msgs::msg::Path path_to_send;
                    bool has_path = false;
                    {
                        std::lock_guard<std::mutex> lock(path_mutex_); // Protege leitura do caminho
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
                            nav_state_ = TaskState::RUNNING;
                            return BT::NodeStatus::RUNNING;
                        }
                        else
                        {
                            return BT::NodeStatus::FAILURE;
                        }
                    }
                    else
                    {
                        // Espera o caminho chegar (pode haver um delay entre ComputePath e o feedback)
                        return BT::NodeStatus::RUNNING;
                    }
                }
                return check_task_status(nav_state_);
            });
        });

        // --- Action Builder: PickObject ---
        BT::NodeBuilder builder_pick = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE)
                {
                    auto pose = self.getInput<geometry_msgs::msg::Pose>("pose");
                    auto id = self.getInput<std::string>("id");
                    if (!pose || !id) return BT::NodeStatus::FAILURE;

                    this->send_goal(id.value(), pose.value(), true); // true = pick
                    manipulation_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "PickObject", { BT::InputPort<geometry_msgs::msg::Pose>("pose"), BT::InputPort<std::string>("id") }, {} }, builder_pick);


        // --- Action Builder: PlaceObject ---
        BT::NodeBuilder builder_place = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE)
                {
                    auto pose = self.getInput<geometry_msgs::msg::Pose>("pose");
                    if (!pose) return BT::NodeStatus::FAILURE;

                    std::string id_dummy = cached_object_.id;
                    this->send_goal(id_dummy, pose.value(), false); // false = place
                    manipulation_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "PlaceObject", { BT::InputPort<geometry_msgs::msg::Pose>("pose"), BT::InputPort<std::vector<double>>("limits") }, {} }, builder_place);

        // --- Carregamento Final do XML ---
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
    

    // DOC-START: bt_loop
    /**
     * @brief Loop principal da Behavior Tree rodando em uma thread dedicada.
     *
     * @details Roda a 50Hz.
     * 1. Verifica se há novos objetos detectados (`has_new_object_`).
     * 2. Executa `bt_tree_.tickOnce()`.
     * 3. Se a árvore terminar (SUCCESS ou FAILURE), reseta flags e estados para permitir nova execução.
     */
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

            bool new_obj = false;
            {
                std::lock_guard<std::mutex> lock(bt_mutex_);
                new_obj = has_new_object_;
            }

            // Só roda a árvore se ela já estiver rodando, se houver novo objeto ou se um alvo já estiver travado
            if (status == BT::NodeStatus::RUNNING || new_obj || !current_target_id_.empty())
            {
                BT::NodeStatus result = bt_tree_.tickOnce();

                if (result == BT::NodeStatus::SUCCESS || result == BT::NodeStatus::FAILURE)
                {
                    std::lock_guard<std::mutex> lock(bt_mutex_);
                    has_new_object_ = false;

                    if (result == BT::NodeStatus::FAILURE)
                    {
                         picked.erase(cached_object_.id); // Permite tentar novamente o mesmo ID se falhar
                         current_target_id_ = "";
                    }

                    // Reseta estados para garantir limpeza
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



    /** @} */

    /**
     * @brief Carrega lista de labels autorizados de um arquivo YAML.
     * @param yaml_path Caminho do arquivo.
     */
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

    /** @} */

    /**
     * @name Callbacks de Sensores
     * Recepção de dados da Odometria e Visão Computacional.
     * @{
    */

    /**
     * @brief Callback de odometria.
     * @details Atualiza a posição X, Y do robô. Assume-se Z=0 para navegação 2D.
     * @param msg Mensagem de odometria recebida.
     */
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        pose_x = msg->pose.pose.position.x;
        pose_y = msg->pose.pose.position.y;
        pose_z = 0.0;
    }

    /**
     * @brief Callback de Detecção 3D (YOLO + Depth).
     *
     * @details
     * Lógica:
     * 1. Caso o robô já esteja fixo em um alvo ('current_target_id_'), apenas atualiza a pose dele para que os nós da BT que precisem da posição do objeto recuperem a posição atual dele.
     * É necessário mutex para não haver condição de corrida entre a thread do ROS2 e a thread da BT.
     * 2. Caso o robô agora ocioso, então procura na lista de detecções um objeto que:
     * - Esteja na lista `authorized_labels` (YAML).
     * - Não esteja na lista `picked` (já coletados).
     * 3. Se encontrar, salva em `cached_object_` e seta `has_new_object_` para acordar a Behavior Tree.
     *
     * @param msg Array de detecções 3D.
    */
    void detection_callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(bt_mutex_);

        if (!current_target_id_.empty() || has_new_object_)
        {
            // Atualização de tracking
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

        // Busca de novo alvo
        for (const auto &det : msg->detections)
        {
            if (det.results.empty()) continue;

            std::string raw_id = det.results[0].hypothesis.class_id;
            // Extrai label base (ex: "box_1" -> "box")
            std::string label = raw_id;
            size_t pos = raw_id.find('_');
            if (pos != std::string::npos) label = raw_id.substr(0, pos);

            if (authorized_labels.find(label) == authorized_labels.end()) continue;
            if (picked.find(raw_id) != picked.end()) continue;

            geometry_msgs::msg::Pose pose;
            pose.position = det.bbox.center.position;
            pose.orientation = det.bbox.center.orientation;

            cached_object_.id = raw_id;
            cached_object_.pose = pose;
            cached_object_.size = det.bbox.size;
            has_new_object_ = true;

            RCLCPP_INFO(this->get_logger(), "Nova detecção salva: '%s'", raw_id.c_str());
            break;
        }
    }

    // --- Path Callbacks (Action Client) ---

    /** @} */

    /**
     * @name Cliente de Ação: Path Planning
     * Funções para requisitar o cálculo de rotas (A*, D*, etc.) e tratar feedback.
     * @{
     */

    /**
     * @brief Envia um goal para a Action de Path Planning (ex: A*).
     * @details Limpa caminhos anteriores e configura callbacks.
     * @note Usa mutex `state_mutex_` para limpar handle e setar falha se servidor offline.
     * @param target_pose Pose alvo para onde o caminho deve ser planejado.
     */
    void send_path_goal(const geometry_msgs::msg::Pose & target_pose)
    {
        // 1. Limpa o handle ativo (Protegido por mutex)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            this->active_path_goal_handle_.reset();
        }

        // 2. Espera o servidor SEM segurar o mutex do estado
        if (!this->path_client->wait_for_action_server(std::chrono::seconds(2)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server 'path' not available");
            std::lock_guard<std::mutex> lock(state_mutex_);
            path_state_ = TaskState::FAILURE;
            return;
        }

        // 3. Limpa o path anterior
        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            last_calculated_path_.poses.clear();
        }

        auto goal_msg = mobile_manipulation_interfaces::action::Path::Goal();
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "BT: Enviando Goal (Pose) para A*...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SendGoalOptions();
        send_goal_options.goal_response_callback = std::bind(&ServerNode::path_goal_response_callback, this, std::placeholders::_1);
        send_goal_options.feedback_callback = std::bind(&ServerNode::path_feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
        send_goal_options.result_callback = std::bind(&ServerNode::path_result_callback, this, std::placeholders::_1);

        this->path_client->async_send_goal(goal_msg, send_goal_options);
    }

    /**
     * @brief Função auxiliar para cancelar o controlador de movimento.
     * @details Chamada quando o planejador de caminho envia feedback de "recalculando",
     * indicando que o caminho atual é inválido.
     */
    void cancel_controller_goal()
    {
        if (this->active_controller_goal_handle_)
        {
            RCLCPP_WARN(this->get_logger(), "Solicitando PARADA IMEDIATA (Cancelando Action Controller)...");
            this->controller_client->async_cancel_goal(this->active_controller_goal_handle_);
        }
    }

    /**
     * @brief Callback de feedback da Action de Path.
     * @details Recebe trechos do caminho calculado ou sinalização de recalculo.
     * Se `recalculating_path` for true, cancela o movimento atual do robô.
     */
    void path_feedback_callback(
        rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::SharedPtr,
        const std::shared_ptr<const mobile_manipulation_interfaces::action::Path::Feedback> feedback)
    {
        if (!feedback->path.poses.empty())
        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            std::cout << "caminho recebido." << std::endl;
            this->last_calculated_path_ = feedback->path;
        }

        std::cout << "opa." << std::endl;

        if (feedback->recalculating_path)
        {
            {
                std::lock_guard<std::mutex> lock(path_mutex_);
                last_calculated_path_.poses.clear();
            }
            RCLCPP_INFO(this->get_logger(), "Planner recalculando: cancelando controller atual...");
            cancel_controller_goal();
        }
    }

    /**
     * @brief Callback de aceitação/rejeição do Goal de Path.
     */
    void path_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>> & goal_handle)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (!goal_handle) 
        {
            RCLCPP_ERROR(this->get_logger(), "Goal PATH rejeitado");
            path_state_ = TaskState::FAILURE;
        } 
        else 
        {
            this->active_path_goal_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Goal PATH aceito (ID Interno: %s)", rclcpp_action::to_string(goal_handle->get_goal_id()).c_str());
        }
    }

    /**
     * @brief Callback de resultado final da Action de Path.
     * @details Atualiza `path_state_` para SUCCESS ou FAILURE.
     */
    void path_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::WrappedResult & result)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (!this->active_path_goal_handle_ || result.goal_id != this->active_path_goal_handle_->get_goal_id()) {
            return;
        }

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            if (result.result->success)
            {
                path_state_ = TaskState::SUCCESS;
                RCLCPP_INFO(this->get_logger(), "PATH RESULT SUCCESS");
            }
            else
            {
                path_state_ = TaskState::FAILURE;
                RCLCPP_WARN(this->get_logger(), "PATH RESULT LOGICAL FAILURE (Success = false)");
            }
        }
        else if (result.code == rclcpp_action::ResultCode::ABORTED)
        {
            path_state_ = TaskState::FAILURE;
            RCLCPP_WARN(this->get_logger(), "PATH ACTION ABORTED");
        }
        else if (result.code == rclcpp_action::ResultCode::CANCELED)
        {
            path_state_ = TaskState::IDLE;
            RCLCPP_WARN(this->get_logger(), "PATH ACTION CANCELED");
        }
        else
        {
            path_state_ = TaskState::FAILURE;
        }

        this->active_path_goal_handle_.reset();
    }

    // --- Controller Callbacks ---

    /** @} */

    /**
     * @name Cliente de Ação: Controller
     * Funções para execução de trajetórias e controle de movimento.
     * @{
     */

    /**
     * @brief Envia um goal para o Controlador (seguidor de caminho).
     * @param target_path O caminho (lista de poses) a ser seguido.
     * @return true Se o servidor aceitou enviar, false se servidor offline.
     */
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

    /** @brief Callback de aceitação do Goal do Controller. */
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
            RCLCPP_INFO(this->get_logger(), "Goal CONTROLLER aceito, executando...");
        }
    }

    /** @brief Callback de resultado do Controller. Limpa o path salvo se sucesso. */
    void controller_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::WrappedResult & result)
    {
        if (this->active_controller_goal_handle_ && result.goal_id != this->active_controller_goal_handle_->get_goal_id()) 
        {
            return;
        }

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            {
                std::lock_guard<std::mutex> lock(path_mutex_);
                last_calculated_path_.poses.clear(); // Limpa caminho já executado
            }
            nav_state_ = TaskState::SUCCESS;
            RCLCPP_INFO(this->get_logger(), "Navegação concluída!");
        }
        else if (result.code == rclcpp_action::ResultCode::CANCELED)
        {
            nav_state_ = TaskState::IDLE;
        }
        else
        {
            nav_state_ = TaskState::FAILURE;
        }

        if (this->active_controller_goal_handle_ && result.goal_id == this->active_controller_goal_handle_->get_goal_id()) 
        {
            this->active_controller_goal_handle_.reset();
        }
    }

    // --- Pick/Place Callbacks ---

    /** @} */

    /**
     * @name Cliente de Ação: Manipulation
     * Funções para controle do braço robótico (Pick & Place) via MoveIt.
     * @{
     */

    /**
     * @brief Envia goal de manipulação (Pick ou Place).
     * @param id ID do objeto (necessário para planejamento de colisão no MoveIt).
     * @param target_pose Pose alvo do End Effector.
     * @param pick Booleano: true para Pegar, false para Largar.
     */
    void send_goal(const std::string id, const geometry_msgs::msg::Pose & target_pose, bool pick)
    {
        if (!this->client_ptr_->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            manipulation_state_ = TaskState::FAILURE;
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::PickObject::Goal();
        goal_msg.obstacle_id = id;
        goal_msg.pick = pick;
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "BT: Enviando Goal (Pose) para MANIPULATION...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SendGoalOptions();
        send_goal_options.goal_response_callback = std::bind(&ServerNode::goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::result_callback, this, std::placeholders::_1);

        this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
    }

    /** @brief Callback de aceitação da manipulação. */
    void goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>> & goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal PICK rejeitado");
            manipulation_state_ = TaskState::FAILURE;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Goal PICK aceito, executando...");
        }
    }

    /** @brief Callback de resultado da manipulação. */
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
            RCLCPP_ERROR(this->get_logger(), "PICK FAILED or ABORTED");
        }
    }


    // Publishers

    void publish_pose()
    {
        auto message = geometry_msgs::msg::Pose();

        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            message = cached_object_.pose;
        }

        publisher_->publish(message);
    }

};

/**
 * @brief Helper function para verificar se uma flag existe nos argumentos da linha de comando.
 * @param args Vetor de argumentos.
 * @param flag Flag a ser buscada (ex: "--no-organize").
 * @return true se encontrada, false caso contrário.
 */
bool has_flag(const std::vector<std::string>& args, const std::string& flag) 
{
    return std::find(args.begin(), args.end(), flag) != args.end();
}

/**
 * @brief Função principal (Entry Point).
 *
 * @details
 * 1. Inicializa o ROS 2.
 * 2. Processa argumentos de linha de comando para ativar/desativar nós opcionais (organize, storage, gripper).
 * 3. Cria instâncias dos nós auxiliares.
 * 4. Adiciona todos os nós a um MultiThreadedExecutor para permitir execução paralela de callbacks.
 * 5. Inicia o spin.
 *
 * @param argc Número de argumentos.
 * @param argv Vetor de argumentos.
 * @return int Código de saída (0 sucesso).
 */
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  std::vector<std::string> args(argv, argv + argc);

  // Flags para desenvolvimento/teste modular
  bool enable_organize = !has_flag(args, "--no-organize");
  bool enable_storage  = !has_flag(args, "--no-storage");
  bool enable_gripper  = !has_flag(args, "--no-gripper");


  // Opções para renomear nós internos se necessário para evitar colisão de nomes
  rclcpp::NodeOptions organize_opts;
  organize_opts.arguments({"--ros-args", "-r", "__node:=organize_node"});

  rclcpp::NodeOptions storage_opts;
  storage_opts.arguments({"--ros-args", "-r", "__node:=storage_node"});

  rclcpp::NodeOptions gripper_opts;
  gripper_opts.arguments({"--ros-args", "-r", "__node:=gripper_monitor_node"});


  std::shared_ptr<storage_manager::OrganizeNode> organize_node = nullptr;
  std::shared_ptr<storage_manager::StorageNode> storage_node   = nullptr;
  std::shared_ptr<manipulation::IsGripperHolding> gripper_node = nullptr;

  // Executor MultiThreaded essencial para que callbacks de Actions não bloqueiem uns aos outros
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

  // Injeção de dependência dos nós auxiliares no ServerNode
  auto server_node = std::make_shared<ServerNode>(gripper_node, storage_node, organize_node);
  executor.add_node(server_node);

  executor.spin();

  rclcpp::shutdown();
  return 0;
}