#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <thread>
#include <unordered_map>
#include <atomic>
#include <mutex>

// Inclui o header da classe definida
#include <manipulation/SimpleManipulation.hpp>

// Biblioteca para leitura de arquivos YAML
#include <yaml-cpp/yaml.h>

// Bibliotecas para transformações geométricas e matemática (quaterniões/matrizes)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// Bibliotecas do MoveIt para controle do braço e cena de planejamento
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/allowed_collision_matrix.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

using namespace std::chrono_literals;

namespace manipulation {

// DOC-START: SimpleManipulation
// Construtor da classe: Inicializa o nó e configura a comunicação ROS 2
SimpleManipulation::SimpleManipulation()
 : Node("simple_manipulation")
{
    // Carrega o caminho do arquivo YAML dos parâmetros do ROS
    this->declare_parameter<std::string>("yaml_file", "");
    yaml_file = this->get_parameter("yaml_file").as_string();

    // Cria um nó separado apenas para o MoveIt. Isso é necessário para evitar 
    // travamentos (deadlocks), pois o MoveIt precisa de um executor rodando em paralelo.
    moveit_node_ = std::make_shared<rclcpp::Node>("simple_manipulation_moveit_node");

    // Configura um executor MultiThreaded para gerenciar o nó do MoveIt em uma thread separada
    executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    executor_->add_node(moveit_node_);
    executor_thread_ = std::thread([this]() { this->executor_->spin(); });

    // Assina o tópico do sensor de contato (simulado ou real) para saber se pegou o objeto
    subscription_ = this->create_subscription<std_msgs::msg::Float32>(
            "contact_sensor", 10, std::bind(&SimpleManipulation::topic_callback, this, std::placeholders::_1));

    // Cria o servidor da Action "pick_object", que recebe os comandos de pegar/largar
    this->action_server_ = rclcpp_action::create_server<mobile_manipulation_interfaces::action::PickObject>(
        this, 
        "pick_object",
        std::bind(&SimpleManipulation::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&SimpleManipulation::handle_cancel, this, std::placeholders::_1),
        std::bind(&SimpleManipulation::handle_accepted, this, std::placeholders::_1));

    // Cliente para chamar serviços externos de colisão (ex: ativar/desativar colisão da base)
    client_ = this->create_client<mobile_manipulation_interfaces::srv::MobileObjectCollision>(
        "/object_collision");
    
    // Publishers e Clients para manipular a Cena de Planejamento do MoveIt (adicionar/remover obstáculos)
    planning_scene_publisher_ = this->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 1);        
    get_planning_scene_client_ = this->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");

    // Inicializa o sistema de transformadas (TF2) para converter coordenadas entre frames
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Cria um timer para tentar inicializar o MoveIt após 1 segundo (garante que o sistema subiu)
    init_timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&SimpleManipulation::initMoveGroup, this));

    // Carrega as poses de "pega" (offsets) do arquivo YAML
    loadLocationsFromYaml(yaml_file);
}   
// DOC-END: SimpleManipulation

// DOC-START: ~SimpleManipulation
// Destrutor: Garante que a thread do executor pare corretamente ao fechar o nó
SimpleManipulation::~SimpleManipulation()
{
    executor_->cancel();
    if (executor_thread_.joinable())
    {
        executor_thread_.join();
    }
}
// DOC-END: ~SimpleManipulation

// DOC-START: loadLocationsFromYaml
// Função para ler o arquivo YAML
// O YAML contém offsets (posições relativas) de como pegar cada tipo de objeto (ex: "cola", "box")
void SimpleManipulation::loadLocationsFromYaml(const std::string &yaml_path)
{
    try
    {
        YAML::Node config = YAML::LoadFile(yaml_path);

        // Itera sobre cada tipo de objeto no arquivo
        for (const auto &label_node : config)
        {
            const std::string label = label_node.first.as<std::string>();
            const YAML::Node &locations_node = label_node.second;

            std::vector<geometry_msgs::msg::Pose> locations;

            // Lê posição (x,y,z) e orientação (roll,pitch,yaw)
            for (const auto &loc_item : locations_node)
            {
                if (!loc_item.IsMap() || loc_item.size() != 1)
                {
                    RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                "[%s] Ignorando entrada inválida de localização.", label.c_str());
                    continue;
                }

                const auto &loc_name = loc_item.begin()->first.as<std::string>();
                const YAML::Node &loc_data = loc_item.begin()->second;

                if (!loc_data["position"] || !loc_data["orientation"])
                {
                    RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                "[%s] '%s' faltando dados", label.c_str(), loc_name.c_str());
                    continue;
                }

                const YAML::Node &pos = loc_data["position"];
                const YAML::Node &ori = loc_data["orientation"];

                geometry_msgs::msg::Pose pose;

                pose.position.x = pos[0].as<double>();
                pose.position.y = pos[1].as<double>();
                pose.position.z = pos[2].as<double>();

                // Converte Roll, Pitch, Yaw para Quaternion (formato usado pelo ROS)
                double roll  = ori[0].as<double>();
                double pitch = ori[1].as<double>();
                double yaw   = ori[2].as<double>();

                tf2::Quaternion q;
                q.setRPY(roll, pitch, yaw);
                q.normalize();

                pose.orientation.x = q.x();
                pose.orientation.y = q.y();
                pose.orientation.z = q.z();
                pose.orientation.w = q.w();

                locations.push_back(pose);

                RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                            "Carregado [%s - %s]", label.c_str(), loc_name.c_str());
            }

            // Salva no mapa global da classe
            pick_and_place_poses[label] = locations;
        }
    }
    catch (const YAML::Exception &e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                    "Erro ao carregar YAML '%s': %s", yaml_path.c_str(), e.what());
    }
}
// DOC-END: loadLocationsFromYaml

// DOC-START: initMoveGroup
// Inicializa as interfaces do MoveIt (Braço e Garra)
// É chamado pelo Timer 1 segundo após o boot para evitar conflitos de inicialização
void SimpleManipulation::initMoveGroup() 
{
    try 
    {
        // Interface para controlar o braço ("panda_arm")
        move_group_arm = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            moveit_node_, "panda_arm"); 
        
        // Interface para controlar a garra ("hand")
        move_group_gripper = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            moveit_node_, "hand"); 

        std::cout << "alo" << std::endl; // Debug simples

        // Inicia monitoramento do estado atual do robô
        move_group_arm->startStateMonitor(); 

        RCLCPP_INFO(this->get_logger(), "MoveGroup (arm e gripper) inicializados com sucesso.");

        // Marca como pronto e desliga o timer
        moveit_ready_ = true; 
        init_timer_->cancel();  
    } 
    catch (const std::exception &e) 
    {
        RCLCPP_WARN(this->get_logger(), "Tentando inicializar MoveGroupInterface: %s...", e.what());
    }
}
// DOC-END: initMoveGroup

// DOC-START: ready
// Move o braço para uma posição segura/padrão ("Home") definida por valores de juntas
void SimpleManipulation::ready()
{
    if (!move_group_arm) {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do arm não inicializado.");
        return;
    }

    // Define os ângulos exatos de cada junta para a posição "ready"
    move_group_arm->setJointValueTarget({
        {"panda_joint1", 0.0},
        {"panda_joint2", -0.5934},
        {"panda_joint3", 0.0},
        {"panda_joint4", -1.17},
        {"panda_joint5", 0.0},
        {"panda_joint6", 0.576},
        {"panda_joint7", 0.8552},
    });

    // Planeja e Executa o movimento
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto result = move_group_arm->plan(plan);

    if (result == moveit::core::MoveItErrorCode::SUCCESS) 
    {
        auto exec_result = move_group_arm->execute(plan);
        if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            RCLCPP_INFO(this->get_logger(), "Braço na posição padrão (Ready).");
        }
    }
}
// DOC-END: ready

// DOC-START: close_gripper
// Comando para fechar a garra (movendo as juntas dos dedos para próximo de 0)
void SimpleManipulation::close_gripper() 
{
    if (!move_group_gripper) {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do GRIPPER não inicializado.");
        return;
    }
    move_group_gripper->setStartStateToCurrentState();
    
    // Valor muito pequeno indica garra fechada
    move_group_gripper->setJointValueTarget({
        {"panda_finger_joint1", 0.003},
        {"panda_finger_joint2", 0.003},
    });

    move_group_gripper->allowReplanning(true);
    
    auto result = move_group_gripper->move();

    if (result == moveit::core::MoveItErrorCode::SUCCESS) 
    {
        RCLCPP_INFO(this->get_logger(), "Gripper fechou (MoveIt).");
    } 
    else 
    {
        RCLCPP_ERROR(this->get_logger(), "Falha ao fechar o gripper.");
    }
}
// DOC-END: close_gripper

// DOC-START: open_gripper
// Comando para abrir a garra (movendo as juntas dos dedos para fora)
void SimpleManipulation::open_gripper() 
{
    if (!move_group_gripper) {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do GRIPPER não inicializado.");
        return;
    }

    move_group_gripper->setStartStateToCurrentState();

    // Valor maior indica garra aberta
    move_group_gripper->setJointValueTarget({
            {"panda_finger_joint1", 0.038},
            {"panda_finger_joint2", 0.038},
    });
    move_group_gripper->allowReplanning(true);

    auto result = move_group_gripper->move();

    if (result == moveit::core::MoveItErrorCode::SUCCESS) 
    {
        RCLCPP_INFO(this->get_logger(), "Gripper abriu (MoveIt).");
    } 
    else 
    {
        RCLCPP_ERROR(this->get_logger(), "Falha ao abrir o gripper.");
    }
}
// DOC-END: open_gripper

// DOC-START: attempt_cartesian_move
// Tenta mover o braço em LINHA RETA (Cartesiano)
// Retorna true se conseguiu calcular pelo menos 99% do caminho
bool SimpleManipulation::attempt_cartesian_move(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool avoid_collisions)
{
    const int MAX_CARTESIAN_ATTEMPTS = 5;
    const double MIN_CARTESIAN_FRACTION = 0.99; // Exige 99% do caminho planejado
    const double eef_step = 0.01; // Resolução do passo (1cm)

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;

    move_group_arm->setStartStateToCurrentState();
    move_group_arm->setMaxVelocityScalingFactor(maxVelocity);
    move_group_arm->setMaxAccelerationScalingFactor(maxVelocity);

    // Tenta algumas vezes caso o planejador falhe
    for (int cart_attempt = 1; cart_attempt <= MAX_CARTESIAN_ATTEMPTS; ++cart_attempt)
    {
        double fraction = move_group_arm->computeCartesianPath(waypoints, eef_step, trajectory, avoid_collisions);

        if (fraction >= MIN_CARTESIAN_FRACTION)
        {
            RCLCPP_INFO(this->get_logger(), "Planejamento Cartesiano OK (%.1f%%). Executando...", fraction * 100.0);
            
            auto exec_result = move_group_arm->execute(trajectory);

            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS)
            {
                return true; 
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Execução Cartesiana falhou. Tentando novamente...");
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Planejamento Cartesiano incompleto (fração: %.2f).", fraction);
        }
    }

    RCLCPP_ERROR(this->get_logger(), "Falha no planejamento Cartesiano.");
    return false; 
}
// DOC-END: attempt_cartesian_move

// DOC-START: positions_for_arm
// Move o braço usando planejadores convencionais (RRTConnect, etc.)
// Isso permite movimentos curvos para desviar de obstáculos (ao contrário do cartesiano)
bool SimpleManipulation::positions_for_arm(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool computeCartesian)
{
    if (!move_group_arm)
    {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
        return false;
    }

    const int MAX_PLANNING_CYCLES = 4; 
    bool task_success = false; 

    // Opcional: Tentar movimento reto primeiro se solicitado
    if (computeCartesian)
    {
        if (attempt_cartesian_move(target_pose, maxVelocity, true))
        {
            return true;
        }
    }

    RCLCPP_INFO(this->get_logger(), "Iniciando planejamento Free-Space (RRTConnect)...");

    move_group_arm->setStartStateToCurrentState();
    move_group_arm->setPlannerId("RRTConnectkConfigDefault"); 
    move_group_arm->setPoseTarget(target_pose, "panda_link8"); // Define o alvo para a ponta do braço

    // Parâmetros de tolerância e velocidade
    move_group_arm->setPlanningTime(5.0); 
    move_group_arm->setNumPlanningAttempts(20); 
    move_group_arm->setMaxVelocityScalingFactor(maxVelocity);
    move_group_arm->setMaxAccelerationScalingFactor(maxVelocity);

    double tolerance = 0.005; 
    move_group_arm->setGoalPositionTolerance(tolerance);
    move_group_arm->setGoalOrientationTolerance(0.01); 

    // Tenta planejar e executar algumas vezes
    for (int cycle = 1; cycle <= MAX_PLANNING_CYCLES; ++cycle)
    {
        RCLCPP_INFO(this->get_logger(), "Ciclo de Planejamento %d/%d", cycle, MAX_PLANNING_CYCLES);

        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        auto plan_result = move_group_arm->plan(my_plan);

        if (plan_result == moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_INFO(this->get_logger(), "Planejamento bem-sucedido. Executando...");
            
            auto exec_result = move_group_arm->execute(my_plan);

            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS)
            {
                RCLCPP_INFO(this->get_logger(), "Execução bem-sucedida.");
                task_success = true;
                break; 
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Execução falhou.");
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Falha ao encontrar plano válido.");
        }
    }

    if (!task_success)
    {
        RCLCPP_ERROR(this->get_logger(), "FALHA FINAL: Não foi possível atingir a pose.");
    }

    return task_success;
}
// DOC-END: positions_for_arm

// DOC-START: calculate_global_pose
// Função Central da Lógica de Manipulação.
// 1. Identifica o objeto.
// 2. Calcula a pose exata no mundo baseada no offset do YAML.
// 3. Decide se é Pick (Pegar) ou Place (Largar) e executa a sequência.
bool SimpleManipulation::calculate_global_pose(std::string received_id, geometry_msgs::msg::Pose pose, bool pick)
{
    // Limpa o ID (ex: "box_1" vira "box") para buscar no YAML genérico
    std::string id = received_id;
    size_t pos = received_id.find('_');
    if (pos != std::string::npos) 
    {
        id = received_id.substr(0, pos);
    }

    // Verifica se temos instruções de como pegar esse objeto
    if (pick_and_place_poses.find(id) == pick_and_place_poses.end()) 
    {
        RCLCPP_WARN(this->get_logger(), "ID '%s' não encontrado no YAML.", id.c_str());
        return false;
    }

    const auto &poses = pick_and_place_poses[id];
    geometry_msgs::msg::Pose target_pose_world;

    // Itera sobre as configurações possíveis (geralmente apenas 1)
    for (size_t i = 0; i < poses.size(); ++i) 
    {
        const auto &pose_local = poses[i];

        // --- Matemática de Transformação ---
        // Converte a pose local (offset) para coordenadas globais
        // baseando-se na posição atual detectada do objeto.
        
        tf2::Vector3 local_point(pose_local.position.x, pose_local.position.y, pose_local.position.z);
        tf2::Quaternion q_object(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
        tf2::Vector3 t_object(pose.position.x, pose.position.y, pose.position.z);

        tf2::Transform object_transform(q_object, t_object);
        tf2::Vector3 world_point = object_transform * local_point;

        target_pose_world.position.x = world_point.x();
        target_pose_world.position.y = world_point.y();
        target_pose_world.position.z = world_point.z();

        // Cálculo da orientação final somando RPY do objeto + RPY do offset
        double obj_r, obj_p, obj_y;
        tf2::Matrix3x3(q_object).getRPY(obj_r, obj_p, obj_y);
        
        double off_r, off_p, off_y;
        tf2::Quaternion q_offset(
            pose_local.orientation.x, pose_local.orientation.y,
            pose_local.orientation.z, pose_local.orientation.w
        );
        tf2::Matrix3x3(q_offset).getRPY(off_r, off_p, off_y);

        // Soma simples dos ângulos (Roll, Pitch, Yaw)
        double final_r = obj_r + off_r;
        double final_p = obj_p + off_p;
        double final_y = obj_y + off_y;

        tf2::Quaternion q_final;
        q_final.setRPY(final_r, final_p, final_y);
        q_final.normalize();

        target_pose_world.orientation.x = q_final.x();
        target_pose_world.orientation.y = q_final.y();
        target_pose_world.orientation.z = q_final.z();
        target_pose_world.orientation.w = q_final.w();

        // --- Sequência de PEGAR (Pick) ---
        if(pick == true)
        {
            // 1. Move o braço para a posição de pega
            if (positions_for_arm(target_pose_world, 1.0, false)) 
            {
                // Desativa movimentação do obstáculo (trava)
                send_request(received_id, false);

                // 2. Anexa o objeto à garra virtualmente (MoveIt Collision)
                std::vector<std::string> touch_links = move_group_gripper->getLinkNames();
                move_group_arm->attachObject(received_id, "panda_link8", touch_links);
                
                rclcpp::sleep_for(std::chrono::milliseconds(100));

                // 3. Fecha a garra fisicamente
                close_gripper();
                
                rclcpp::sleep_for(std::chrono::milliseconds(100));
                
                // 4. Verifica sensor de força para confirmar se pegou
                bool picked = false;
                int contador = 0;
                {
                    std::lock_guard<std::mutex> lock(contact_sensor_mutex);
                    for(size_t i = 0; i < contact_sensor_data.size(); i++)
                    {
                        if(contact_sensor_data[i] > 0.8) contador++;
                    }
                    if(contador >= 9) picked = true; // Confirmação se muitos leituras forem altas
                }
                
                // Se falhou em pegar, desanexa e aborta
                if(picked == false)
                {
                    move_group_arm->detachObject(received_id);
                    ready();
                    return false;
                }

                // 5. Levanta o objeto (movimento cartesiano para cima)
                target_pose_world.position.z += 0.1;
                attempt_cartesian_move(target_pose_world, 1.0, false);

                // Volta para posição de descanso
                ready();
            }
            else
            {
                return false;
            }
        }
        // --- Sequência de LARGAR (Place) ---
        else if(pick == false)
        {
            // 1. Move para a posição de destino
            if (positions_for_arm(target_pose_world, 1.0, false)) 
            {
                // 2. Abre a garra
                open_gripper();
                
                rclcpp::sleep_for(std::chrono::milliseconds(300));

                // 3. Desanexa o objeto da garra no MoveIt
                move_group_arm->detachObject(received_id);

                rclcpp::sleep_for(std::chrono::milliseconds(200));
                
                // 4. Configura matriz de colisão e libera movimentação do obstáculo
                set_collision_allowance(received_id, "ground_plane", false); 
                send_request(received_id, true);
                
                // Volta para home
                ready();
            }
            else
            {
                return false;
            }
        }
    }
    return true;
}
// DOC-END: calculate_global_pose

// DOC-START: set_collision_allowance
// Modifica a Matriz de Colisões Permitidas (ACM) do MoveIt dinamicamente
// Útil para permitir que o robô coloque um objeto no chão sem o MoveIt achar que é uma batida
void SimpleManipulation::set_collision_allowance(const std::string& id1, const std::string& id2, bool allow_collision)
{
    auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    request->components.components = moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;

    if (!get_planning_scene_client_->wait_for_service(std::chrono::milliseconds(500))) 
    {
        RCLCPP_ERROR(this->get_logger(), "Serviço get_planning_scene indisponível.");
        return;
    }

    // Busca a matriz atual
    auto future = get_planning_scene_client_->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
        RCLCPP_ERROR(this->get_logger(), "Timeout ao buscar Planning Scene.");
        return;
    }

    auto response = future.get();
    auto &acm = response->scene.allowed_collision_matrix;

    // Helper para achar índices na matriz
    auto get_or_add_index = [&](const std::string &name) -> int 
    {
        for (size_t i = 0; i < acm.entry_names.size(); ++i) 
        {
            if (acm.entry_names[i] == name) return i;
        }
        acm.entry_names.push_back(name);
        for (auto &entry : acm.entry_values) entry.enabled.push_back(false); 
        moveit_msgs::msg::AllowedCollisionEntry new_row;
        new_row.enabled.resize(acm.entry_names.size(), false); 
        acm.entry_values.push_back(new_row);
        return acm.entry_names.size() - 1;
    };

    int idx1 = get_or_add_index(id1);
    int idx2 = get_or_add_index(id2);

    // Atualiza a permissão de colisão (simétrica)
    if (acm.entry_values.size() > (size_t)idx1 && acm.entry_values[idx1].enabled.size() > (size_t)idx2)
        acm.entry_values[idx1].enabled[idx2] = allow_collision;
    
    if (acm.entry_values.size() > (size_t)idx2 && acm.entry_values[idx2].enabled.size() > (size_t)idx1)
        acm.entry_values[idx2].enabled[idx1] = allow_collision;

    // Publica a atualização na cena
    moveit_msgs::msg::PlanningScene update_msg;
    update_msg.is_diff = true; 
    update_msg.allowed_collision_matrix = acm;
    planning_scene_publisher_->publish(update_msg);
}
// DOC-END: set_collision_allowance

// DOC-START: send_request
// Envia uma requisição de serviço para controlar colisão ou física de objetos móveis
bool SimpleManipulation::send_request(std::string received_obstacle_id, bool received_activate_movement)
{
    auto request = std::make_shared<mobile_manipulation_interfaces::srv::MobileObjectCollision::Request>();
    request->obstacle_id = received_obstacle_id;
    request->activate_movement = received_activate_movement;

    if (!client_->wait_for_service(std::chrono::seconds(1))) {
        RCLCPP_ERROR(this->get_logger(), "Serviço '/object_collision' não está disponível.");
        return false;
    }

    auto future_result = client_->async_send_request(request);

    if (future_result.wait_for(std::chrono::seconds(5)) == std::future_status::ready)
    {
        try 
        {
            auto response = future_result.get(); 
            if (response->success) 
            {
                RCLCPP_INFO(this->get_logger(), "Service síncrono OK: %s", received_obstacle_id.c_str());
                return true;
            } 
            else 
            {
                RCLCPP_WARN(this->get_logger(), "Service retornou false.");
                return false;
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Erro service: %s", e.what());
            return false;
        }
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Timeout do serviço.");
        return false;
    }
}
// DOC-END: send_request

// --- Callbacks do Action Server (PickObject) ---

// DOC-START: handle_goal
// Aceita ou rejeita o pedido da Action
rclcpp_action::GoalResponse SimpleManipulation::handle_goal(const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const mobile_manipulation_interfaces::action::PickObject::Goal> goal)
{
    RCLCPP_INFO(this->get_logger(), "Action PickObject recebida: pose [x: %.2f, y: %.2f]", 
        goal->pose.position.x, goal->pose.position.y);
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}
// DOC-END: handle_goal

// DOC-START: handle_cancel
// Lida com pedidos de cancelamento
rclcpp_action::CancelResponse SimpleManipulation::handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle)
{
    RCLCPP_INFO(this->get_logger(), "Cancelamento da Action solicitado.");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
}
// DOC-END: handle_cancel

// DOC-START: handle_accepted
// Quando aceito, cria uma thread para executar a lógica pesada sem travar o ROS
void SimpleManipulation::handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle)
{
    using namespace std::placeholders;
    std::thread{std::bind(&SimpleManipulation::execute, this, std::placeholders::_1), goal_handle}.detach();
}
// DOC-END: handle_accepted

// DOC-START: execute
// Lógica principal da Action (roda em thread separada)
void SimpleManipulation::execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle)
{
    RCLCPP_INFO(this->get_logger(), "Executando Action...");

    // Espera o MoveIt estar pronto antes de tentar mover
    while (!moveit_ready_ && rclcpp::ok()) 
    {
        if (goal_handle->is_canceling()) 
        {
            auto result = std::make_shared<mobile_manipulation_interfaces::action::PickObject::Result>();
            result->success = false;
            goal_handle->canceled(result);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<mobile_manipulation_interfaces::action::PickObject::Result>();

    // Se for pegar, garante que a garra começa aberta
    if(goal->pick == true)
    {
        open_gripper();
    }

    // Chama a função principal de cálculo e movimento
    bool action_sucess = calculate_global_pose(goal->obstacle_id, goal->pose, goal->pick);

    // Retorna o resultado para o cliente da Action
    if (rclcpp::ok()) 
    {
        result->success = action_sucess;
        goal_handle->succeed(result);
        RCLCPP_INFO(this->get_logger(), "Action finalizada.");
    }
}
// DOC-END: execute

// DOC-START: topic_callback
// Callback do tópico do sensor de força/contato
// Armazena os últimos 10 valores em um buffer circular (deque)
void SimpleManipulation::topic_callback(const std_msgs::msg::Float32 & msg)
{
    std::lock_guard<std::mutex> lock(contact_sensor_mutex);

    if (contact_sensor_data.size() > 10) 
    {
        contact_sensor_data.pop_front();
    }
    contact_sensor_data.push_back(msg.data);
}
// DOC-END: topic_callback

} // namespace manipulation