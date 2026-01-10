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

#include <manipulation/SimpleManipulation.hpp>

#include <yaml-cpp/yaml.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/allowed_collision_matrix.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include <moveit/robot_trajectory/robot_trajectory.h>

#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
using namespace std::chrono_literals;

namespace manipulation {


SimpleManipulation::SimpleManipulation()
 : Node("simple_manipulation")
{
    this->declare_parameter<std::string>("yaml_file", "");
    this->declare_parameter<bool>("use_graspnet", false);

    yaml_file = this->get_parameter("yaml_file").as_string();
    use_graspnet = this->get_parameter("use_graspnet").as_bool();

    moveit_node_ = std::make_shared<rclcpp::Node>("simple_manipulation_moveit_node");

    executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    
    executor_->add_node(moveit_node_);
    executor_thread_ = std::thread([this]() { this->executor_->spin(); });

    subscription_ = this->create_subscription<std_msgs::msg::Float32>(
            "contact_sensor", 10, std::bind(&SimpleManipulation::topic_callback, this, std::placeholders::_1));

    subscription_1 = this->create_subscription<geometry_msgs::msg::Pose>("object_pose", 
      10, 
      std::bind(&SimpleManipulation::object_pose_callback, this, std::placeholders::_1));

    this->action_server_ = rclcpp_action::create_server<mobile_manipulation_interfaces::action::PickObject>(
        this, 
        "pick_object",
        std::bind(&SimpleManipulation::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&SimpleManipulation::handle_cancel, this, std::placeholders::_1),
        std::bind(&SimpleManipulation::handle_accepted, this, std::placeholders::_1));

    client_ = this->create_client<mobile_manipulation_interfaces::srv::MobileObjectCollision>(
        "/object_collision");
    
    planning_scene_publisher_ = this->create_publisher<moveit_msgs::msg::PlanningScene>("planning_scene", 1);        
    get_planning_scene_client_ = this->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    init_timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&SimpleManipulation::initMoveGroup, this));

    loadLocationsFromYaml(yaml_file);
}   

SimpleManipulation::~SimpleManipulation()
{
    executor_->cancel();
    if (executor_thread_.joinable())
    {
        executor_thread_.join();
    }
}


void SimpleManipulation::loadLocationsFromYaml(const std::string &yaml_path)
{
    try
    {
        YAML::Node config = YAML::LoadFile(yaml_path);

        for (const auto &label_node : config)
        {
            const std::string label = label_node.first.as<std::string>();
            const YAML::Node &locations_node = label_node.second;

            std::vector<geometry_msgs::msg::Pose> locations;

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
                                "[%s] '%s' missing position/orientation",
                                label.c_str(), loc_name.c_str());
                    continue;
                }

                const YAML::Node &pos = loc_data["position"];
                const YAML::Node &ori = loc_data["orientation"];

                if (pos.size() != 3 || ori.size() != 3) 
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
                            "Loaded [%s - %s] -> pos:[%.2f, %.2f, %.2f], RPY:[%.2f, %.2f, %.2f], quat:[%.2f, %.2f, %.2f, %.2f]",
                            label.c_str(), loc_name.c_str(),
                            pose.position.x, pose.position.y, pose.position.z,
                            roll, pitch, yaw,
                            pose.orientation.x, pose.orientation.y,
                            pose.orientation.z, pose.orientation.w);
            }

            pick_and_place_poses[label] = locations;
        }
    }
    catch (const YAML::Exception &e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("yaml_loader"),
                    "Failed to load YAML file '%s': %s", yaml_path.c_str(), e.what());
    }
}

void SimpleManipulation::initMoveGroup() 
{
    try 
    {
        move_group_arm = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            moveit_node_, "panda_arm"); 
        
        move_group_gripper = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            moveit_node_, "hand"); 


        move_group_arm->startStateMonitor(); 

        RCLCPP_INFO(this->get_logger(), "Inicializando PlanningSceneMonitor...");
        
        // Cria o monitor usando o nó do moveit e o tópico robot_description
        psm_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
            moveit_node_, "robot_description");

        if (psm_->getPlanningScene())
        {
            // Inicia os monitores para manter a cena atualizada (colisões, estados das juntas)
            psm_->startSceneMonitor();
            psm_->startWorldGeometryMonitor();
            psm_->startStateMonitor();
            RCLCPP_INFO(this->get_logger(), "PlanningSceneMonitor iniciado com sucesso.");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Falha ao inicializar PlanningSceneMonitor.");
        }
        // ============================================================

        RCLCPP_INFO(this->get_logger(), "MoveGroup (arm e gripper) inicializados com sucesso.");

        moveit_ready_ = true; 
        init_timer_->cancel();
    } 
    catch (const std::exception &e) 
    {
        RCLCPP_WARN(this->get_logger(), "Tentando inicializar MoveGroupInterface: %s...", e.what());
    }
}

void SimpleManipulation::ready()
{
    if (!move_group_arm) {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do arm não inicializado.");
        return;
    }

    
    move_group_arm->setJointValueTarget({
        {"panda_joint1", 0.0},
        {"panda_joint2", -0.5934},
        {"panda_joint3", 0.0},
        {"panda_joint4", -1.17},
        {"panda_joint5", 0.0},
        {"panda_joint6", 0.576},
        {"panda_joint7", 0.8552},
    
    });

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto result = move_group_arm->plan(plan);

    if (result == moveit::core::MoveItErrorCode::SUCCESS) 
    {
        auto exec_result = move_group_arm->execute(plan);

        if (exec_result == moveit::core::MoveItErrorCode::SUCCESS) 
        {
            RCLCPP_INFO(this->get_logger(), "Gripper na posição padrão.");
        }
    }
}

void SimpleManipulation::close_gripper() 
{
    if (!move_group_gripper) {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do GRIPPER não inicializado.");
        return;
    }
    move_group_gripper->setStartStateToCurrentState();
    
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

void SimpleManipulation::open_gripper() 
{
    if (!move_group_gripper) {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface do GRIPPER não inicializado.");
        return;
    }

    move_group_gripper->setStartStateToCurrentState();

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

bool SimpleManipulation::attempt_cartesian_move(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool avoid_collisions)
{
    const int MAX_CARTESIAN_ATTEMPTS = 5;
    const double MIN_CARTESIAN_FRACTION = 0.99; 
    const double eef_step = 0.01; 


    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;

    move_group_arm->setStartStateToCurrentState();
    move_group_arm->setMaxVelocityScalingFactor(maxVelocity);
    move_group_arm->setMaxAccelerationScalingFactor(maxVelocity);

    for (int cart_attempt = 1; cart_attempt <= MAX_CARTESIAN_ATTEMPTS; ++cart_attempt)
    {
        double fraction = move_group_arm->computeCartesianPath(waypoints, eef_step, trajectory, avoid_collisions);

        if (fraction >= MIN_CARTESIAN_FRACTION)
        {
            RCLCPP_INFO(this->get_logger(), "Planejamento Cartesiano bem-sucedido (%.1f%%). Executando...", fraction * 100.0);
            
            auto exec_result = move_group_arm->execute(trajectory);

            if (exec_result == moveit::core::MoveItErrorCode::SUCCESS)
            {
                return true; 
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Execução Cartesiana falhou (erro no controlador). Tentando novamente...");
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Planejamento Cartesiano incompleto (fração: %.2f). Tentativa %d/%d", 
                fraction, cart_attempt, MAX_CARTESIAN_ATTEMPTS);
        }
    }

    RCLCPP_ERROR(this->get_logger(), "Falha no planejamento Cartesiano após %d tentativas.", MAX_CARTESIAN_ATTEMPTS);
    return false; 
}

bool SimpleManipulation::positions_for_arm(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool computeCartesian)
{
    if (!move_group_arm)
    {
        RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
        return false;
    }

    const int MAX_PLANNING_CYCLES = 4; 
    bool task_success = false; 

    if (computeCartesian)
    {
        if (attempt_cartesian_move(target_pose, maxVelocity, true))
        {
            return true;
        }
    }

    RCLCPP_INFO(this->get_logger(), "Iniciando planejamento Free-Space para Pose Target...");

    move_group_arm->setStartStateToCurrentState();
    
   
    move_group_arm->setPlannerId("RRTConnectkConfigDefault"); 

    move_group_arm->setPoseTarget(target_pose, "panda_link8");


    move_group_arm->setPlanningTime(5.0); 
    move_group_arm->setNumPlanningAttempts(20); 

    move_group_arm->setMaxVelocityScalingFactor(maxVelocity);
    move_group_arm->setMaxAccelerationScalingFactor(maxVelocity);


    double tolerance = 0.005; 
    move_group_arm->setGoalPositionTolerance(tolerance);
    move_group_arm->setGoalOrientationTolerance(0.01); 

    for (int cycle = 1; cycle <= MAX_PLANNING_CYCLES; ++cycle)
    {
        RCLCPP_INFO(this->get_logger(), "Ciclo de Planejamento %d/%d (Vel: %.2f)", cycle, MAX_PLANNING_CYCLES, maxVelocity);

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
                RCLCPP_WARN(this->get_logger(), "Execução falhou (Erro de controlador ou Trajectory Tolerance).");
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Falha ao encontrar plano válido nesta tentativa.");
        }
        
    }

    if (!task_success)
    {
        RCLCPP_ERROR(this->get_logger(), "FALHA FINAL: Não foi possível atingir a pose na velocidade %.2f.", maxVelocity);
    }

    return task_success;
}

bool SimpleManipulation::calculate_global_pose(std::string received_id, geometry_msgs::msg::Pose pose, bool pick)
{
    std::string id = received_id;
    size_t pos = received_id.find('_');
    if (pos != std::string::npos) 
    {
        id = received_id.substr(0, pos);
    }

    if (pick_and_place_poses.find(id) == pick_and_place_poses.end()) 
    {
        RCLCPP_WARN(this->get_logger(), "ID '%s' não encontrado no YAML.", id.c_str());
        return false;
    }

    
    if(use_graspnet == true)
    {
        if(pick == true)
        {

            tf2::Quaternion q_original(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            );
            double roll_orig, pitch_orig, yaw_orig;
            tf2::Matrix3x3(q_original).getRPY(roll_orig, pitch_orig, yaw_orig);

            // Criar novo quaternion com roll=π, pitch=0, yaw=yaw_original
            tf2::Quaternion q_new;
            q_new.setRPY(M_PI, 0.0, yaw_orig + M_PI/4.0);
            q_new.normalize();

            // Atribuir à pose
            pose.orientation.x = q_new.x();
            pose.orientation.y = q_new.y();
            pose.orientation.z = q_new.z();
            pose.orientation.w = q_new.w();
            pose.position.z = pose.position.z + 0.11;

            if (positions_for_arm(pose, 1.0, false)) 
            {
                send_request(received_id, false);

                std::vector<std::string> touch_links = move_group_gripper->getLinkNames();
                move_group_arm->attachObject(received_id, "panda_link8", touch_links);
                
                rclcpp::sleep_for(std::chrono::milliseconds(100));

                close_gripper();
                
                rclcpp::sleep_for(std::chrono::milliseconds(100));
                
                bool picked = false;
                int contador = 0;

                {
                    std::lock_guard<std::mutex> lock(contact_sensor_mutex);

                    for(size_t i = 0; i < contact_sensor_data.size(); i++)
                    {
                        if(contact_sensor_data[i] > 0.8)
                        {
                            contador++;
                        }

                    }

                    if(contador >= 9)
                    {
                        picked = true;
                    }
                }
                
                if(picked == false)
                {
                    open_gripper();
                    move_group_arm->detachObject(received_id);

                    rclcpp::sleep_for(std::chrono::milliseconds(150));

                    geometry_msgs::msg::Pose updated_object_pose;

                    {
                        std::lock_guard<std::mutex> lock(object_pose_mutex);

                        updated_object_pose = object_pose;
                    }
                    
                    pose.position.z += 0.125;
                    attempt_cartesian_move(pose, 1.0, false);

                    return calculate_global_pose(received_id, updated_object_pose, pick);
                    
                }

                pose.position.z += 0.125;
                attempt_cartesian_move(pose, 1.0, false);

                // set_collision_allowance(received_id, "ground_plane", true);

                // rclcpp::sleep_for(std::chrono::milliseconds(200));

                ready();

                return true;
            }
            else
            {
                return false;
            }
        }
        else if(pick == false)
                {

                    const auto &poses = pick_and_place_poses[id];
                    geometry_msgs::msg::Pose target_pose_world;

                    for (size_t i = 0; i < poses.size(); ++i) 
                    {
                        const auto &pose_local = poses[i];

                        tf2::Vector3 local_point(
                            pose_local.position.x,
                            pose_local.position.y,
                            pose_local.position.z
                        );

                        tf2::Quaternion q_object(
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w
                        );

                        tf2::Vector3 t_object(
                            pose.position.x,
                            pose.position.y,
                            pose.position.z
                        );

                        tf2::Transform object_transform(q_object, t_object);
                        tf2::Vector3 world_point = object_transform * local_point;

                        
                        target_pose_world.position.x = world_point.x();
                        target_pose_world.position.y = world_point.y();
                        target_pose_world.position.z = world_point.z();

                        double obj_r, obj_p, obj_y;
                        tf2::Matrix3x3(q_object).getRPY(obj_r, obj_p, obj_y);

                        
                        double off_r, off_p, off_y;
                        tf2::Quaternion q_offset(
                            pose_local.orientation.x,
                            pose_local.orientation.y,
                            pose_local.orientation.z,
                            pose_local.orientation.w
                        );
                        tf2::Matrix3x3(q_offset).getRPY(off_r, off_p, off_y);

                    
                        bool use_roll  = true;
                        bool use_pitch = true; 
                        bool use_yaw   = true; 

                        double final_r = obj_r + (use_roll  ? off_r : 0.0);
                        double final_p = obj_p + (use_pitch ? off_p : 0.0);
                        double final_y = obj_y + (use_yaw   ? off_y : 0.0);

                        tf2::Quaternion q_final;
                        q_final.setRPY(final_r, final_p, final_y);
                        q_final.normalize();

                        target_pose_world.orientation.x = q_final.x();
                        target_pose_world.orientation.y = q_final.y();
                        target_pose_world.orientation.z = q_final.z();
                        target_pose_world.orientation.w = q_final.w();

                        if (positions_for_arm(target_pose_world, 1.0, false)) 
                        {
                            open_gripper();
                            
                            rclcpp::sleep_for(std::chrono::milliseconds(300));

                            move_group_arm->detachObject(received_id);

                            rclcpp::sleep_for(std::chrono::milliseconds(200));
                            set_collision_allowance(received_id, "ground_plane", false);

                            send_request(received_id, true);
                            ready();

                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }
                }

    }
    else
    {
        const auto &poses = pick_and_place_poses[id];
        geometry_msgs::msg::Pose target_pose_world;

        for (size_t i = 0; i < poses.size(); ++i) 
        {
            const auto &pose_local = poses[i];

            tf2::Vector3 local_point(
                pose_local.position.x,
                pose_local.position.y,
                pose_local.position.z
            );

            tf2::Quaternion q_object(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            );

            tf2::Vector3 t_object(
                pose.position.x,
                pose.position.y,
                pose.position.z
            );

            tf2::Transform object_transform(q_object, t_object);
            tf2::Vector3 world_point = object_transform * local_point;

            
            target_pose_world.position.x = world_point.x();
            target_pose_world.position.y = world_point.y();
            target_pose_world.position.z = world_point.z();

            double obj_r, obj_p, obj_y;
            tf2::Matrix3x3(q_object).getRPY(obj_r, obj_p, obj_y);

            
            double off_r, off_p, off_y;
            tf2::Quaternion q_offset(
                pose_local.orientation.x,
                pose_local.orientation.y,
                pose_local.orientation.z,
                pose_local.orientation.w
            );
            tf2::Matrix3x3(q_offset).getRPY(off_r, off_p, off_y);

        
            bool use_roll  = true;
            bool use_pitch = true; 
            bool use_yaw   = true; 

            double final_r = obj_r + (use_roll  ? off_r : 0.0);
            double final_p = obj_p + (use_pitch ? off_p : 0.0);
            double final_y = obj_y + (use_yaw   ? off_y : 0.0);

            tf2::Quaternion q_final;
            q_final.setRPY(final_r, final_p, final_y);
            q_final.normalize();

            target_pose_world.orientation.x = q_final.x();
            target_pose_world.orientation.y = q_final.y();
            target_pose_world.orientation.z = q_final.z();
            target_pose_world.orientation.w = q_final.w();

            if(pick == true)
            {
                if (positions_for_arm(target_pose_world, 1.0, false)) 
                {
                    send_request(received_id, false);

                    std::vector<std::string> touch_links = move_group_gripper->getLinkNames();
                    move_group_arm->attachObject(received_id, "panda_link8", touch_links);
                    
                    rclcpp::sleep_for(std::chrono::milliseconds(100));

                    close_gripper();
                    
                    rclcpp::sleep_for(std::chrono::milliseconds(100));
                    
                    bool picked = false;
                    int contador = 0;

                    {
                        std::lock_guard<std::mutex> lock(contact_sensor_mutex);

                        for(size_t i = 0; i < contact_sensor_data.size(); i++)
                        {
                            if(contact_sensor_data[i] > 0.8)
                            {
                                contador++;
                            }

                        }

                        if(contador >= 9)
                        {
                            picked = true;
                        }
                    }
                    
                    if(picked == false)
                    {
                        open_gripper();
                        move_group_arm->detachObject(received_id);

                        rclcpp::sleep_for(std::chrono::milliseconds(150));

                        geometry_msgs::msg::Pose updated_object_pose;

                        {
                            std::lock_guard<std::mutex> lock(object_pose_mutex);

                            updated_object_pose = object_pose;
                        }
                        
                        target_pose_world.position.z += 0.125;
                        attempt_cartesian_move(target_pose_world, 1.0, false);

                        return calculate_global_pose(received_id, updated_object_pose, pick);
                        
                    }

                    target_pose_world.position.z += 0.125;
                    attempt_cartesian_move(target_pose_world, 1.0, false);

                    // set_collision_allowance(received_id, "ground_plane", true);

                    // rclcpp::sleep_for(std::chrono::milliseconds(200));

                    ready();

                    return true;
                }
                else
                {
                    return false;
                }
            }
            else if(pick == false)
            {
                if (positions_for_arm(target_pose_world, 1.0, false)) 
                {
                    open_gripper();
                    
                    rclcpp::sleep_for(std::chrono::milliseconds(300));

                    move_group_arm->detachObject(received_id);

                    rclcpp::sleep_for(std::chrono::milliseconds(200));
                    set_collision_allowance(received_id, "ground_plane", false);

                    send_request(received_id, true);
                    ready();

                    return true;
                }
                else
                {
                    return false;
                }
            }

            
        }
    }
    
    


    return true;
}

bool SimpleManipulation::follow_path_with_consistent_ik(
    const std::vector<geometry_msgs::msg::Pose>& path_poses,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>>& goal_handle)
{
    if (path_poses.empty()) return false;
    if (!move_group_arm || !psm_) {
        RCLCPP_ERROR(this->get_logger(), "MoveGroup ou PSM não inicializados.");
        return false;
    }

    // --- 1. Preparação ---
    psm_->requestPlanningSceneState();
    planning_scene_monitor::LockedPlanningSceneRO scene(psm_);
    auto robot_model = move_group_arm->getRobotModel();
    const moveit::core::JointModelGroup* joint_model_group = robot_model->getJointModelGroup("panda_arm");
    
    // Nome do link da ponta para verificação (FK)
    // Geralmente é o último link do grupo. Ajuste se o seu for diferente (ex: "panda_link8")
    const std::string& tip_frame = joint_model_group->getLinkModelNames().back();

    moveit::core::RobotStatePtr current_state = std::make_shared<moveit::core::RobotState>(scene->getCurrentState());
    
    robot_trajectory::RobotTrajectory trajectory(robot_model, joint_model_group);
    trajectory.addSuffixWayPoint(*current_state, 0.0);

    RCLCPP_INFO(this->get_logger(), "Calculando IK RIGIDO para %zu pontos...", path_poses.size());

    collision_detection::CollisionRequest c_req;
    collision_detection::CollisionResult c_res;
    
    // Definição de rigidez: Tolerância máxima de erro angular (radianos)
    // 0.005 rad ~= 0.28 graus (Extremamente rígido)
    const double MAX_ORIENTATION_ERROR = 0.005; 

    int points_added = 0;

    // --- 2. Loop de Cálculo ---
    for (size_t i = 0; i < path_poses.size(); ++i)
    {
        if (goal_handle->is_canceling()) {
            RCLCPP_WARN(this->get_logger(), "Cancelamento no cálculo.");
            return false;
        }

        const auto& target_pose = path_poses[i];

        // Copia estado anterior (Semente para consistência de juntas)
        moveit::core::RobotState candidate_state = *current_state;

        // Tenta IK com timeout curto
        bool found_ik = candidate_state.setFromIK(joint_model_group, target_pose, 0.02);

        if (found_ik)
        {
            candidate_state.update(); // Recalcula FK para verificar o resultado real

            // --- VALIDAÇÃO DE RIGIDEZ DE ORIENTAÇÃO ---
            // O solver pode achar uma solução "aproximada". Vamos verificar se ela é exata.
            
            // 1. Pega a orientação real que o robô ficaria com essa solução
            const Eigen::Isometry3d& actual_transform = candidate_state.getGlobalLinkTransform(tip_frame);
            Eigen::Quaterniond q_actual(actual_transform.rotation());
            
            // 2. Pega a orientação desejada
            Eigen::Quaterniond q_target(
                target_pose.orientation.w,
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z
            );

            // 3. Calcula a distância angular (menor ângulo entre os dois quaternions)
            double angle_diff = q_actual.angularDistance(q_target);

            if (angle_diff > MAX_ORIENTATION_ERROR)
            {
                // REJEITA: A solução existe, mas o ângulo está "torto" além do permitido
                RCLCPP_DEBUG(this->get_logger(), 
                    "Pulo Ponto %zu: IK encontrado, mas erro de orientação (%.4f rad) excede limite rígido.", 
                    i, angle_diff);
                continue; // Pula para o próximo ponto
            }
            // -------------------------------------------

            c_res.clear();
            scene->checkCollision(c_req, c_res, candidate_state);

            if (!c_res.collision)
            {
                *current_state = candidate_state; // Aceita como nova semente
                trajectory.addSuffixWayPoint(*current_state, 0.0);
                points_added++;
            }
            else
            {
                RCLCPP_DEBUG(this->get_logger(), "Pulo Ponto %zu: Colisão.", i);
            }
        }
        else
        {
            RCLCPP_DEBUG(this->get_logger(), "Pulo Ponto %zu: IK Falhou.", i);
        }
    }

    if (points_added == 0) {
        RCLCPP_ERROR(this->get_logger(), "Nenhum ponto atendeu aos critérios de colisão e rigidez.");
        return false;
    }

    // --- 3. Parametrização ---
    trajectory_processing::TimeOptimalTrajectoryGeneration totg;
    if (!totg.computeTimeStamps(trajectory, 0.5, 0.5)) {
        RCLCPP_ERROR(this->get_logger(), "Falha no TOTG.");
        return false;
    }

    // --- 4. Execução ---
    moveit_msgs::msg::RobotTrajectory traj_msg;
    trajectory.getRobotTrajectoryMsg(traj_msg);

    RCLCPP_INFO(this->get_logger(), "Executando %d pontos validados...", points_added);
    move_group_arm->asyncExecute(traj_msg);

    double estimated_duration = trajectory.getDuration();
    auto start_time = std::chrono::steady_clock::now();
    
    while (rclcpp::ok())
    {
        if (goal_handle->is_canceling()) 
        {
            move_group_arm->stop();
            return false;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();

        if (elapsed > (estimated_duration + 0.5)) break; 

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    return true;
}

void SimpleManipulation::set_collision_allowance(const std::string& id1, const std::string& id2, bool allow_collision)
{
    auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    request->components.components = moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;

    if (!get_planning_scene_client_->wait_for_service(std::chrono::milliseconds(500))) 
    {
        RCLCPP_ERROR(this->get_logger(), "Serviço get_planning_scene indisponível.");
        return;
    }

    auto future = get_planning_scene_client_->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
        RCLCPP_ERROR(this->get_logger(), "Timeout ao buscar Planning Scene.");
        return;
    }

    auto response = future.get();
    auto &acm = response->scene.allowed_collision_matrix;

    auto get_or_add_index = [&](const std::string &name) -> int 
    {
        for (size_t i = 0; i < acm.entry_names.size(); ++i) 
        {
            if (acm.entry_names[i] == name) return i;
        }
        
        acm.entry_names.push_back(name);
        
        for (auto &entry : acm.entry_values) 
        {
            entry.enabled.push_back(false); 
        }

        moveit_msgs::msg::AllowedCollisionEntry new_row;
        new_row.enabled.resize(acm.entry_names.size(), false); 
        acm.entry_values.push_back(new_row);

        return acm.entry_names.size() - 1;
    };

    int idx1 = get_or_add_index(id1);
    int idx2 = get_or_add_index(id2);

    
    if (acm.entry_values.size() > (size_t)idx1 && acm.entry_values[idx1].enabled.size() > (size_t)idx2)
        acm.entry_values[idx1].enabled[idx2] = allow_collision;
    
    if (acm.entry_values.size() > (size_t)idx2 && acm.entry_values[idx2].enabled.size() > (size_t)idx1)
        acm.entry_values[idx2].enabled[idx1] = allow_collision;

    moveit_msgs::msg::PlanningScene update_msg;
    update_msg.is_diff = true; 
    update_msg.allowed_collision_matrix = acm;
    
    planning_scene_publisher_->publish(update_msg);
    
}

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
                RCLCPP_WARN(this->get_logger(), "Service retornou false (falha lógica no servidor).");
                return false;
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Erro ao processar resposta do serviço: %s", e.what());
            return false;
        }
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Timeout: O serviço demorou demais para responder.");
        return false;
    }
}

// Subscribers

void SimpleManipulation::object_pose_callback(const geometry_msgs::msg::Pose & msg)
{
    {
        std::lock_guard<std::mutex> lock(object_pose_mutex);

        object_pose = msg;
    }

  
}

// Action server (pick_object).

// Action server (pick_object).

rclcpp_action::GoalResponse SimpleManipulation::handle_goal(const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const mobile_manipulation_interfaces::action::PickObject::Goal> goal)
{
    // Ajuste no log para mostrar a quantidade de poses recebidas em vez de tentar acessar uma única
    RCLCPP_INFO(this->get_logger(), "Recebido pedido de Action PickObject com %zu poses candidatas.", 
        goal->poses.size());
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse SimpleManipulation::handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle)
{
    RCLCPP_INFO(this->get_logger(), "Recebido pedido de cancelamento da Action.");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
}

void SimpleManipulation::handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle)
{
    using namespace std::placeholders;
    
    std::thread{std::bind(&SimpleManipulation::execute, this, std::placeholders::_1), goal_handle}.detach();
}

void SimpleManipulation::execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle)
{
    RCLCPP_INFO(this->get_logger(), "Aguardando inicialização do MoveIt...");

    // 1. Loop de espera inicial com verificação de cancelamento
    while (!moveit_ready_ && rclcpp::ok()) 
    {
        if (goal_handle->is_canceling()) 
        {
            auto result = std::make_shared<mobile_manipulation_interfaces::action::PickObject::Result>();
            result->success = false;
            goal_handle->canceled(result);
            RCLCPP_INFO(this->get_logger(), "Action cancelada durante a inicialização.");
            return;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Ainda aguardando MoveGroup...");
    }

    RCLCPP_INFO(this->get_logger(), "MoveIt pronto. Executando lógica de Pick (Action)...");
    
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<mobile_manipulation_interfaces::action::PickObject::Result>();

    // 2. Lógica inicial de gripper
    

    bool any_pose_succeeded = false;

    if(goal->follow_path == false)
    {
        if(goal->pick == true)
        {
            // Verifica cancelamento antes de abrir a garra
            if (goal_handle->is_canceling()) 
            {
                result->success = false;
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Action cancelada antes de abrir a garra.");
                return;
            }
            open_gripper();
        }
         // 3. Loop sobre as poses candidatas
        for (const auto& target_pose : goal->poses)
        {
            // === VERIFICAÇÃO DE CANCELAMENTO CRÍTICA ===
            if (goal_handle->is_canceling()) 
            {
                // Para qualquer movimento que esteja ocorrendo
                if (move_group_arm) 
                {
                    move_group_arm->stop();
                }

                if (move_group_gripper) 
                {
                    move_group_gripper->stop();
                }

                result->success = false;
                goal_handle->canceled(result);
                RCLCPP_WARN(this->get_logger(), "Action cancelada pelo usuário! Parando o robô.");
                return;
            }
            // ===========================================

            RCLCPP_INFO(this->get_logger(), "Tentando pose [x: %.3f, y: %.3f, z: %.3f]...", 
                target_pose.position.x, target_pose.position.y, target_pose.position.z);

            // Chama a sua função original
            bool current_attempt = calculate_global_pose(goal->obstacle_id, target_pose, goal->pick);

            if (current_attempt)
            {
                any_pose_succeeded = true;
                RCLCPP_INFO(this->get_logger(), "Sucesso na pose atual! Interrompendo tentativas.");
                break; 
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Falha na pose atual. Tentando a próxima (se houver)...");
            }
        }
    }
    else
    {
        bool path_execution_success = follow_path_with_consistent_ik(goal->poses, goal_handle);

        if (goal_handle->is_canceling()) 
        {
            if (move_group_arm) move_group_arm->stop();
            result->success = false;
            goal_handle->canceled(result);
            RCLCPP_WARN(this->get_logger(), "Action cancelada durante Follow Path.");
            return;
        }

        if (path_execution_success)
        {
            any_pose_succeeded = true;
        }
        else
        {
            any_pose_succeeded = false;
        }
    }
   

    // 4. Finalização
    if (rclcpp::ok()) 
    {
        // Verifica cancelamento uma última vez caso tenha ocorrido no último milissegundo
        if (goal_handle->is_canceling()) 
        {
            if (move_group_arm) move_group_arm->stop();
            result->success = false;
            goal_handle->canceled(result);
            return;
        }

        result->success = any_pose_succeeded;
        
        if (any_pose_succeeded) 
        {
            goal_handle->succeed(result);
            RCLCPP_INFO(this->get_logger(), "Action PickObject finalizada com SUCESSO.");
        } 
        else 
        {
            goal_handle->abort(result); 
            RCLCPP_ERROR(this->get_logger(), "Action PickObject FALHOU (Nenhuma pose válida).");
        }
    }
}

void SimpleManipulation::topic_callback(const std_msgs::msg::Float32 & msg)
{
    std::lock_guard<std::mutex> lock(contact_sensor_mutex);

    if (contact_sensor_data.size() > 10) 
    {
        contact_sensor_data.pop_front();
    }


    contact_sensor_data.push_back(msg.data);
}


} // namespace manipulation