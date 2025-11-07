/**
 * @file welding_multiple_objects.cpp
 * @brief Implementação do nó principal de soldagem.
 * 
 * Este arquivo contém a definição da classe `Welding`, responsável por:
 * - Controlar o braço robótico via MoveIt2;
 * - Controlar a esteira (velocidade linear e angular);
 * - Interpretar as detecções 3D recebidas da visão computacional;
 * - Executar as trajetórias de solda conforme o objeto detectado.
 * 
 * @version 1.0
 * @date 07-11-2025
 * @author Lucas Momesso
 */
#include <memory>
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <yaml-cpp/yaml.h>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include "object_manipulation_interfaces/srv/object_collision.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/msg/float32.hpp>

using namespace std::chrono_literals;

namespace std 
{
    template <>
    struct hash<std::tuple<float, float, float>> 
    {
        size_t operator()(const std::tuple<float, float, float>& t) const 
        {
            size_t h1 = hash<float>()(std::get<0>(t));
            size_t h2 = hash<float>()(std::get<1>(t));
            size_t h3 = hash<float>()(std::get<2>(t));
            
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

namespace std {
    template<>
    struct hash<std::tuple<std::pair<int, int>, bool>> {
        size_t operator()(const std::tuple<std::pair<int, int>, bool>& t) const {
            const auto& p = std::get<0>(t);
            bool b = std::get<1>(t);
            size_t h1 = std::hash<int>{}(p.first);
            size_t h2 = std::hash<int>{}(p.second);
            size_t h3 = std::hash<bool>{}(b);
            size_t seed = h1;
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}

template <typename T1, typename T2>
struct pair_hash {
    std::size_t operator ()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);  
    }
};

template<typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, const std::tuple<T1, T2, T3>& t) {
    os << "(" << std::get<0>(t) << ", " 
       << std::get<1>(t) << ", " 
       << std::get<2>(t) << ")";
    return os;
}

struct TupleHash {
    std::size_t operator()(const std::tuple<float, float, float>& t) const {
        auto h1 = std::hash<float>{}(std::get<0>(t));
        auto h2 = std::hash<float>{}(std::get<1>(t));
        auto h3 = std::hash<float>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct TupleEqual {
    bool operator()(const std::tuple<float,float,float>& a,
                    const std::tuple<float,float,float>& b) const noexcept {
        return std::get<0>(a) == std::get<0>(b) &&
               std::get<1>(a) == std::get<1>(b) &&
               std::get<2>(a) == std::get<2>(b);
    }
};

/**
 * @class Welding
 * @brief Classe principal responsável pelo controle automatizado de soldagem.
 * 
 * @details
 * A classe `Welding` implementa o nó ROS2 que coordena todo o processo de soldagem automatizada,
 * incluindo o controle do braço robótico (via MoveIt2), o acionamento da esteira no Isaac Sim e a
 * interpretação de detecções 3D provenientes do sistema de visão computacional.
 * 
 * **Principais responsabilidades:**
 * - Inicializar e configurar toda a comunicação ROS2 necessária (publishers, subscribers e timers);
 * - Carregar as poses locais de solda a partir de um arquivo YAML;
 * - Processar as detecções de objetos recebidas e acionar o robô para realizar a soldagem;
 * - Controlar o movimento da esteira, pausando-a e retomando conforme a posição do objeto detectado;
 * - Planejar e executar trajetórias robóticas para atingir as poses de solda;
 * - Garantir o retorno seguro do robô à posição inicial de soldagem após cada operação.
 * 
 * ### Comunicação ROS2
 * - **Publisher `/conveyor_velocity`** → controla a velocidade linear da esteira.
 * - **Publisher `/conveyor_angular_velocity`** → controla a velocidade angular da esteira.
 * - **Subscriber `/bbox_3d_with_labels`** → recebe detecções 3D de objetos com posição e ID.
 * - **Timer** → tenta inicializar o `MoveGroupInterface` de forma assíncrona.
 * 
 * ### Dependências
 * - **MoveIt2** para planejamento e execução de trajetórias do braço robótico.
 * - **YAML-CPP** para leitura dos arquivos de configuração de poses locais.
 * - **vision_msgs** para interpretar as mensagens de detecção 3D.
 * 
 * @note
 * Esta classe foi projetada para integração com o ambiente de simulação Isaac Sim e
 * assume que o grupo de juntas do robô é denominado `"denso_arm"`.
 * 
 * @see loadLocationsFromYaml(), initMoveGroup(), detectionCallback()
 */
class Welding : public rclcpp::Node {

private:

    //Publishers.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_; ///< Publica velocidade linear para a esteira do Isaac Sim.
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_1;  ///< Publica velocidade angular para a esteira do Isaac Sim.

    //Subscriptions.
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_; ///< Recebe a pose, tamanho e o id de objetos.

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm; ///< Declara o move group do Moveit2.

    std::string yaml_file; ///< Caminho do arquivo yaml que contém as poses locais de solda de cada objeto.

    rclcpp::TimerBase::SharedPtr init_timer_; ///< Inicia o timer para iniciar o move group do Moveit2.

    /**
     * @brief Armazena as poses de solda associadas a cada objeto.
     * @details Este mapa associa um identificador de objeto (string) a uma lista de poses de solda.
     *          Cada entrada contém todas as posições e orientações (`geometry_msgs::msg::Pose`)
     *          que o robô deve posicionar o end-effector para realizar as soldas daquele objeto específico.
     * 
     */
    std::unordered_map<std::string, std::vector<geometry_msgs::msg::Pose>> welding_poses; 


    /**
     * @brief Carrega as localizações de solda a partir de um arquivo YAML.
     * @details 
     * Esta função lê um arquivo YAML que contém um conjunto de objetos e suas respectivas 
     * posições/orientações de soldagem. O formato esperado do arquivo YAML é o seguinte:
     * 
     * @code{.yaml}
     * trashcan:
     *   - location1:
     *       position: [x, y, z]
     *       orientation: [qx, qy, qz, qw]
     *   - location2:
     *       position: [x, y, z]
     *       orientation: [qx, qy, qz, qw]
     * firecabinet:
     *   - location1:
     *       position: [x, y, z]
     *       orientation: [qx, qy, qz, qw]
     * @endcode
     * 
     * Cada chave de nível superior (ex: `"trashcan"`, `"firecabinet"`) representa
     * um objeto, e cada um possui uma lista de localizações com suas respectivas
     * posições (`geometry_msgs::msg::Point`) e orientações (`geometry_msgs::msg::Quaternion`).
     * 
     * As poses extraídas são convertidas em `geometry_msgs::msg::Pose` e armazenadas
     * no mapa `welding_poses`, onde:
     * 
     * - **Chave (`std::string`)** → nome/label do objeto
     * - **Valor (`std::vector<geometry_msgs::msg::Pose>`)** → lista de poses associadas ao objeto
     * 
     * @param[in] yaml_path Caminho completo para o arquivo YAML contendo as localizações.
     * 
     * @note
     * - Entradas inválidas (sem `position` ou `orientation`, ou tamanhos incorretos)
     *   são ignoradas com um aviso no log.
     * - O mapa `welding_poses` é sobrescrito para cada label encontrado.
     * 
     * @throws YAML::Exception Caso o arquivo YAML seja inválido ou não possa ser carregado.
     * 
     * @warning
     * Se o arquivo YAML não seguir o formato esperado, algumas localizações podem
     * não ser carregadas corretamente, ou o mapa pode ficar incompleto.
     * 
     * @see welding_poses
     */
    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);

            // Agora 'config' é um MAPA: { "trashcan": [...], "firecabinet": [...] }
            for (const auto &label_node : config)
            {
                const std::string label = label_node.first.as<std::string>();
                const YAML::Node &locations_node = label_node.second;

                std::vector<geometry_msgs::msg::Pose> locations;

                // locations_node é uma LISTA
                for (const auto &loc_item : locations_node)
                {
                    if (!loc_item.IsMap() || loc_item.size() != 1)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] Ignorando entrada inválida de localização.", label.c_str());
                        continue;
                    }

                    // Cada item é um mapa { "locationX": { position, orientation } }
                    const auto &loc_name = loc_item.begin()->first.as<std::string>();
                    const YAML::Node &loc_data = loc_item.begin()->second;

                    if (!loc_data["position"] || !loc_data["orientation"])
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] '%s' missing position/orientation",
                                    label.c_str(), loc_name.c_str());
                        continue;
                    }

                    const YAML::Node &pos = loc_data["position"];
                    const YAML::Node &ori = loc_data["orientation"];

                    if (pos.size() != 3 || ori.size() != 4)
                    {
                        RCLCPP_WARN(rclcpp::get_logger("yaml_loader"),
                                    "[%s] '%s' invalid position/orientation size",
                                    label.c_str(), loc_name.c_str());
                        continue;
                    }

                    geometry_msgs::msg::Pose pose;
                    pose.position.x = pos[0].as<double>();
                    pose.position.y = pos[1].as<double>();
                    pose.position.z = pos[2].as<double>();
                    pose.orientation.x = ori[0].as<double>();
                    pose.orientation.y = ori[1].as<double>();
                    pose.orientation.z = ori[2].as<double>();
                    pose.orientation.w = ori[3].as<double>();

                    locations.push_back(pose);

                    RCLCPP_INFO(rclcpp::get_logger("yaml_loader"),
                                "Loaded [%s - %s] -> pos:[%.2f, %.2f, %.2f], ori:[%.2f, %.2f, %.2f, %.2f]",
                                label.c_str(), loc_name.c_str(),
                                pose.position.x, pose.position.y, pose.position.z,
                                pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w);
                }

                welding_poses[label] = locations;
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                        "Failed to load YAML file '%s': %s", yaml_path.c_str(), e.what());
        }
    }



    /**
     * @brief Inicializa a interface de controle do grupo de movimento do MoveIt.
     * @details 
     * Esta função cria uma instância de `moveit::planning_interface::MoveGroupInterface` 
     * responsável por controlar o grupo de juntas do manipulador (no caso, `"denso_arm"`).
     * 
     * O método tenta inicializar a interface utilizando o contexto do nó atual (`shared_from_this()`).
     * Caso a inicialização seja bem-sucedida, um log de sucesso é exibido e o timer de 
     * reinicialização (`init_timer_`) é cancelado, evitando chamadas repetidas.
     * 
     * Um pequeno atraso de 5 segundos (`rclcpp::sleep_for`) é incluído para garantir que 
     * o MoveIt e o ROS2 tenham inicializado completamente antes do uso da interface.
     * 
     * Caso ocorra uma exceção durante a criação da interface, um aviso (`RCLCPP_WARN`)
     * é exibido no log, indicando que a inicialização ainda não foi concluída com sucesso.
     * 
     * @note 
     * Esta função normalmente é chamada de forma periódica por um timer (`init_timer_`) 
     * até que o MoveGroupInterface seja criado corretamente. Após a criação bem-sucedida,
     * o timer é cancelado.
     * 
     * @throws std::exception Se ocorrer um erro durante a criação do `MoveGroupInterface`.
     * 
     * @warning 
     * Certifique-se de que o MoveIt esteja em execução e o nome do grupo (`"denso_arm"`)
     * corresponda ao definido no arquivo SRDF/MoveItConfig. Caso contrário, a inicialização falhará.
     * 
     * @see move_group_arm, init_timer_
     */
    void initMoveGroup() {  
        try 
        {

            move_group_arm = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                this->shared_from_this(), "denso_arm");  

                rclcpp::sleep_for(std::chrono::milliseconds(5000));
            
            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface inicializado com sucesso.");

            init_timer_->cancel();  
        } catch (const std::exception &e) 
        {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }

    }

    /**
     * @brief Move o braço robótico de volta para a posição de soldagem padrão.
     * @details 
     * Esta função comanda o grupo de juntas do braço (`move_group_arm`) para retornar
     * a uma configuração pré-definida de soldagem. As posições-alvo das juntas são definidas
     * manualmente através de valores específicos para cada articulação.
     * 
     * O método primeiro verifica se o `MoveGroupInterface` foi inicializado corretamente.
     * Caso contrário, a função retorna imediatamente com uma mensagem de erro.
     * 
     * Em seguida, define as metas de juntas via `setJointValueTarget()`, gera um plano de
     * movimento com `plan()`, e, se o planejamento for bem-sucedido, executa o movimento 
     * usando `execute()`. Após a execução, é feito um pequeno atraso de 100 ms para garantir 
     * estabilidade e sincronização.
     * 
     * @note 
     * Os valores das juntas foram ajustados empiricamente para corresponder à posição de soldagem.
     * É importante garantir que esses ângulos sejam compatíveis com o modelo cinemático do robô
     * e que não causem colisões.
     * 
     * @warning 
     * Se o `MoveGroupInterface` não estiver inicializado, a função não realiza nenhuma ação 
     * e apenas registra um erro no log.
     * 
     * @see initMoveGroup()
     */
    void return_to_welding_position()
    {
         if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do arm não inicializado.");
            return;
        }

        
        move_group_arm->setJointValueTarget({
            {"joint1", 0.0},
            {"joint2", -1.1288},
            {"joint3", 2.057},
            {"joint4", 0.0},
            {"joint5", 0.658},
            {"joint6", 0.0},        
        });

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            auto exec_result = move_group_arm->execute(plan);
            rclcpp::sleep_for(std::chrono::milliseconds(100));
            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) 
            {
                RCLCPP_INFO(this->get_logger(), "Returned to welding position.");
            }
        }
    }

    /**
     * @brief Planeja e executa o movimento do braço robótico até uma pose alvo.
     * @details
     * Esta função define parâmetros de planejamento e executa o movimento do braço
     * utilizando o MoveIt, com base em uma pose 3D alvo (`target_pose`).
     * 
     * O método realiza as seguintes etapas:
     * 1. Verifica se o `MoveGroupInterface` está inicializado.
     * 2. Define o estado inicial do robô como o estado atual.
     * 3. Configura o planejador como `RRTConnect`.
     * 4. Define a pose alvo e os parâmetros de planejamento (tempo e tentativas).
     * 5. Define os fatores de escala para velocidade e aceleração.
     * 6. Ajusta as tolerâncias de posição e orientação.
     * 7. Gera o plano de movimento e o executa caso o planejamento seja bem-sucedido.
     * 
     * @param target_pose Pose alvo do tipo `geometry_msgs::msg::Pose` que contém
     * a posição e orientação desejadas para o efetuador final.
     * 
     * @note
     * - A função usa o planejador `RRTConnect`, que é adequado para ambientes com
     * espaços de busca grandes e sem muitas restrições.
     * - As tolerâncias pequenas (0.01) garantem precisão, mas podem aumentar o tempo de planejamento.
     * 
     * @warning
     * - Se `move_group_arm` não estiver inicializado, a função não realiza nenhuma ação
     * e emite um erro no log.
     * - Se o planejador retornar uma trajetória vazia, um aviso é exibido e nenhuma execução ocorre.
     * 
     * @see initMoveGroup(), return_to_welding_position()
     */
    void positions_for_arm(const geometry_msgs::msg::Pose &target_pose) 
    {
        if (!move_group_arm) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return;
        }

      
        move_group_arm->setStartStateToCurrentState();
        move_group_arm->setPlannerId("RRTConnect");
        move_group_arm->setPoseTarget(target_pose);

        move_group_arm->setPlanningTime(5.0);
        move_group_arm->setNumPlanningAttempts(40);

        move_group_arm->setMaxVelocityScalingFactor(0.5);
        move_group_arm->setMaxAccelerationScalingFactor(0.5);

        move_group_arm->setGoalTolerance(0.01);
        move_group_arm->setGoalJointTolerance(0.01);
        move_group_arm->setGoalPositionTolerance(0.01);
        move_group_arm->setGoalOrientationTolerance(0.01);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_arm->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_arm->execute(plan);
    
            rclcpp::sleep_for(std::chrono::milliseconds(50));
        }

        if (plan.trajectory.joint_trajectory.points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Trajetória vazia retornada pelo planejador");
            return;
        }

    }

    
    
        
    
    /**
     * @brief Publica a velocidade linear da esteira no Isaac Sim.
     * @details
     * Esta função envia uma mensagem `std_msgs::msg::Float32` contendo
     * o valor de velocidade linear, publicada através do `publisher_`.
     * 
     * O valor publicado representa a velocidade de translação da esteira
     * simulada no Isaac Sim, usada para controlar o movimento linear.
     * 
     * @param velocity Valor da velocidade linear (em unidades compatíveis com o simulador).
     * 
     * @note
     * - O publisher usado é `publisher_`.
     * - É necessário garantir que `publisher_` foi corretamente inicializado no construtor do nó.
     * - A mensagem é do tipo `std_msgs::msg::Float32`, portanto, apenas um número de ponto flutuante é transmitido.
     * 
     * @see publish_angular_velocity()
     */
    void publish_velocity(float velocity)
    {
        auto message = std_msgs::msg::Float32();
        message.data = velocity;

        publisher_->publish(message);

    }

    /**
     * @brief Publica a velocidade angular da esteira no Isaac Sim.
     * @details
     * Esta função cria e envia uma mensagem `std_msgs::msg::Float32` contendo
     * o valor da velocidade angular, utilizando o `publisher_1`.
     * 
     * O valor publicado representa a taxa de rotação da esteira ou mecanismo
     * de rotação controlado no ambiente simulado do Isaac Sim.
     * 
     * @param velocity Valor da velocidade angular (em rad/s ou unidade definida pelo sistema).
     * 
     * @note
     * - O publisher usado é `publisher_1`.
     * - Assim como o publisher linear, ele deve ser inicializado no construtor do nó.
     * - Este tópico é independente do de velocidade linear.
     * 
     * @see publish_velocity()
     */
    void publish_angular_velocity(float velocity)
    {
        auto message = std_msgs::msg::Float32();
        message.data = velocity;

        publisher_1->publish(message);

    }


    /*
    
        CALLBACKS.

    */
    
    std::string welding_id;
    bool stopped = false, welding_done = false;

    /**
     * @brief Callback responsável por processar as detecções 3D recebidas e acionar o robô para soldagem.
     * @details
     * Esta função é chamada automaticamente sempre que uma nova mensagem do tipo
     * `vision_msgs::msg::Detection3DArray` é recebida no tópico inscrito.
     * 
     * O callback tem como função principal:
     * - Identificar o ID do objeto detectado;
     * - Verificar se existem poses de solda associadas a esse objeto no mapa `welding_poses`;
     * - Controlar o movimento da esteira (parar e retomar);
     * - Enviar o robô para executar as poses de solda correspondentes;
     * - Retornar o robô à posição inicial de soldagem ao término.
     * 
     * ### Fluxo principal:
     * 1. Para cada detecção recebida:
     *    - Extrai o `class_id` do objeto (antes do primeiro `_`, se houver).
     *    - Verifica se há poses de solda registradas para esse objeto em `welding_poses`.
     * 2. Caso a esteira não esteja parada (`stopped == false`), envia comandos de velocidade linear e angular
     *    para continuar o movimento.
     * 3. Quando um objeto é detectado em uma zona de parada definida pelos limites de `y` e `x`, 
     *    o sistema pausa a esteira e armazena o `welding_id` atual.
     * 4. Se o objeto detectado for o mesmo que o armazenado e o sistema estiver parado (`stopped == true`),
     *    o robô executa as poses de solda do objeto.
     * 5. Após a soldagem (`welding_done = true`), o robô retorna à posição inicial e o movimento da esteira é retomado.
     * 
     * @param msg Ponteiro compartilhado (`SharedPtr`) para a mensagem de entrada 
     * do tipo `vision_msgs::msg::Detection3DArray`, contendo uma lista de detecções 3D.
     * 
     * @note
     * - A função usa variáveis de controle globais:
     *   - `welding_id`: armazena o identificador do objeto atualmente em soldagem.
     *   - `stopped`: indica se a esteira está parada.
     *   - `welding_done`: sinaliza que a operação de solda foi concluída.
     * - O mapa `welding_poses` associa IDs de objetos às suas poses locais de solda, 
     *   usadas para calcular poses globais no ambiente.
     * - O cálculo da pose global usa transformações via `tf2`, aplicando rotação e translação
     *   da caixa delimitadora (`bbox`) sobre as poses locais.
     * 
     * @warning
     * - Se `move_group_arm` não estiver inicializado, a execução das poses de solda falhará silenciosamente.
     * - Caso o ID detectado não exista em `welding_poses`, nenhuma ação será tomada para o objeto.
     * - O controle da esteira depende dos publishers `publisher_` (linear) e `publisher_1` (angular)
     *   estarem corretamente configurados.
     * 
     * @see positions_for_arm(), return_to_welding_position(), publish_velocity(), publish_angular_velocity()
     */
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        std::string id;
        for (const auto &det : msg->detections)
        {
            

            size_t pos = det.results[0].hypothesis.class_id.find('_'); 
            if (pos != std::string::npos) 
            {
                id = det.results[0].hypothesis.class_id.substr(0, pos);  
            } 
            else
            {
                id = det.results[0].hypothesis.class_id;  
            }

            if (det.results.empty() || welding_poses.find(id) == welding_poses.end())
            {
                continue;
            }
            
            if(stopped == false)
            {
                publish_velocity(0.2);
                publish_angular_velocity(0.4);
            }


            if(det.bbox.center.position.y < 0.2 && det.bbox.center.position.y > -0.1  && det.bbox.center.position.x > 0.0 && stopped == true && welding_id == det.results[0].hypothesis.class_id)
            {
                
                if (welding_poses.find(id) != welding_poses.end())  
                {
                    const auto &poses = welding_poses[id];  

                    for (size_t i = 0; i < poses.size(); ++i)
                    {
                        const auto &pose_local = poses[i];

                        tf2::Vector3 local_corner(
                            pose_local.position.x,
                            pose_local.position.y,
                            pose_local.position.z);

                        const auto &bbox_pose = det.bbox.center;

                        tf2::Quaternion q(
                            bbox_pose.orientation.x,
                            bbox_pose.orientation.y,
                            bbox_pose.orientation.z,
                            bbox_pose.orientation.w);

                        tf2::Matrix3x3 rot(q);
                        tf2::Vector3 translation(
                            bbox_pose.position.x,
                            bbox_pose.position.y,
                            bbox_pose.position.z);

                        tf2::Vector3 world_corner = rot * local_corner + translation;

                        geometry_msgs::msg::Pose target_pose;
                        target_pose.position.x = world_corner.x();
                        target_pose.position.y = world_corner.y();
                        target_pose.position.z = world_corner.z();

                        target_pose.orientation = pose_local.orientation;

                        RCLCPP_INFO(this->get_logger(),
                                    "Pose %zu - ponto global: x=%.3f, y=%.3f, z=%.3f",
                                    i, world_corner.x(), world_corner.y(), world_corner.z());

                        positions_for_arm(target_pose);
                    }
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "ID '%s' não encontrado em welding_poses", det.results[0].hypothesis.class_id.c_str());
                }

                welding_done = true;
                return_to_welding_position();
                stopped = false;

                publish_velocity(0.2);
                publish_angular_velocity(0.4);
                rclcpp::sleep_for(std::chrono::milliseconds(50));
            }
            else if(det.bbox.center.position.y < 0.2 && det.bbox.center.position.y > -0.1 && det.bbox.center.position.x > 0.0 && stopped == false && welding_id != det.results[0].hypothesis.class_id)
            {
                publish_velocity(0.0);
                publish_angular_velocity(0.0);
                rclcpp::sleep_for(std::chrono::milliseconds(1000));
                welding_id = det.results[0].hypothesis.class_id;
                welding_done = false;
                stopped = true;
            }

        }
    }


        

public:
    /**
     * @brief Construtor da classe Welding.
     * 
     * @details 
     * Inicializa o nó ROS2 responsável pelo controle do braço robótico e da esteira do Isaac Sim.
     * Este construtor configura toda a comunicação ROS necessária para o funcionamento do sistema de soldagem automatizado.
     * 
     * **Responsabilidades principais:**
     * - Declara e lê o parâmetro `yaml_file` com o caminho das poses locais de solda.
     * - Cria publishers para o controle da velocidade linear e angular da esteira.
     * - Cria um subscriber para receber detecções 3D de objetos com suas poses e identificadores.
     * - Carrega as poses de solda do arquivo YAML.
     * - Inicia um temporizador que tenta inicializar o `MoveGroupInterface` do MoveIt2 de forma assíncrona.
     * 
     * ### Publishers
     * - `publisher_` → Publica mensagens `std_msgs::msg::Float32` no tópico `/conveyor_velocity`, 
     *   controlando a **velocidade linear da esteira** no Isaac Sim.
     * - `publisher_1` → Publica mensagens `std_msgs::msg::Float32` no tópico `/conveyor_angular_velocity`, 
     *   controlando a **velocidade angular da esteira**.
     * 
     * ### Subscriber
     * - `sub_` → Recebe mensagens do tipo `vision_msgs::msg::Detection3DArray` no tópico `/bbox_3d_with_labels`.  
     *   Cada mensagem contém as **detecções 3D de objetos** com posição, tamanho e ID (classe do objeto).  
     *   O callback associado (`detectionCallback`) interpreta essas informações e decide quando iniciar o processo de soldagem.
     * 
     * ### Timer
     * - `init_timer_` → Cria um temporizador que chama periodicamente a função `initMoveGroup()` 
     *   até que o `MoveGroupInterface` seja inicializado com sucesso.  
     *   Isso evita falhas caso o MoveIt2 ainda não esteja pronto no início do nó.
     */
    Welding()
     : Node("welding")
    {
        this->declare_parameter<std::string>("yaml_file", "");
   
        yaml_file = this->get_parameter("yaml_file").as_string();
        
        publisher_ = this->create_publisher<std_msgs::msg::Float32>("/conveyor_velocity", 10);
        publisher_1 = this->create_publisher<std_msgs::msg::Float32>("/conveyor_angular_velocity", 10);
       
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&Welding::detectionCallback, this, std::placeholders::_1));
        
        loadLocationsFromYaml(yaml_file);

        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&Welding::initMoveGroup, this));

    }   
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Welding>());
  rclcpp::shutdown();
  return 0;
}