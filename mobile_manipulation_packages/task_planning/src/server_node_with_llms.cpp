/**
 * @file server_node.cpp
 * @brief Nó central de controle (Task Planner) - Versão LLM
 * Recebe XML de BehaviorTree via tópico e executa dinamicamente
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
#include "std_msgs/msg/string.hpp"
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

#include <llms/WorldStateNode.hpp>


namespace BT
{
   
    template <>
    inline geometry_msgs::msg::Pose convertFromString(StringView str)
    {
        geometry_msgs::msg::Pose pose;
        
        pose.position.x = 0.0;
        pose.position.y = 0.0;
        pose.position.z = 0.0;
        pose.orientation.x = 0.0;
        pose.orientation.y = 0.0;
        pose.orientation.z = 0.0;
        pose.orientation.w = 1.0;

        if (str.empty()) return pose;

        std::string s(str.data(), str.size());
        std::vector<double> values;
        std::stringstream ss(s);
        std::string token;

        while (std::getline(ss, token, ';'))
        {
            try {
                values.push_back(std::stod(token));
            } catch (...) {
                values.push_back(0.0);
            }
        }

        if (values.size() >= 3)
        {
            pose.position.x = values[0];
            pose.position.y = values[1];
            pose.position.z = values[2];
        }

        if (values.size() >= 7)
        {
            pose.orientation.x = values[3];
            pose.orientation.y = values[4];
            pose.orientation.z = values[5];
            pose.orientation.w = values[6];
        }

        return pose;
    }

    template <>
    inline geometry_msgs::msg::Vector3 convertFromString(StringView str)
    {
        geometry_msgs::msg::Vector3 vec;
        vec.x = 0.0;
        vec.y = 0.0;
        vec.z = 0.0;

        if (str.empty()) return vec;

        std::string s(str.data(), str.size());
        std::vector<double> values;
        std::stringstream ss(s);
        std::string token;

        while (std::getline(ss, token, ';'))
        {
            try {
                values.push_back(std::stod(token));
            } catch (...) {
                values.push_back(0.0);
            }
        }

        if (values.size() >= 3)
        {
            vec.x = values[0];
            vec.y = values[1];
            vec.z = values[2];
        }

        return vec;
    }
}

enum class TaskState
{
    IDLE,
    RUNNING,
    SUCCESS,
    FAILURE
};

// Nó de Controle Personalizado: "Parallel Any"
class ParallelAny : public BT::ControlNode
{
public:
    ParallelAny(const std::string& name, const BT::NodeConfig& config)
        : BT::ControlNode(name, config) {}

    static BT::PortsList providedPorts() { return {}; }

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

    void halt() override
    {
        haltChildren();
        BT::ControlNode::halt();
    }
};

// Wrapper para Actions Assíncronas
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
        if (halt_fun_) halt_fun_(*this);
    }

private:
    std::function<BT::NodeStatus(BT::TreeNode&)> tick_fun_;
    std::function<void(BT::TreeNode&)> halt_fun_; 
};


class ServerNode : public rclcpp::Node
{
public:
    ServerNode(
        std::shared_ptr<manipulation::IsGripperHolding> gripper_node,
        std::shared_ptr<storage_manager::StorageNode> storage_node,
        std::shared_ptr<storage_manager::OrganizeNode> organize_node,
        std::shared_ptr<manipulation::ProjectedReachabilityAnalysis> reachability_node,
        std::shared_ptr<navigation::SharedObstacleGraph> obstacle_graph_node,
        std::shared_ptr<manipulation::IKValidator> ik_validator_node,
        std::shared_ptr<manipulation::CloudBoxFilter> cloud_box_filter_node,
        std::shared_ptr<drl_to_pick_cpp::BridgeToInference> bridge_to_inference_node,
        std::shared_ptr<llms::WorldStateNode> world_state_node
    )
    : Node("server_node"),
    gripper_monitor_node_(gripper_node),
    storage_node_(storage_node),
    organize_node_(organize_node),
    reachability_node_(reachability_node),
    obstacle_graph_node_(obstacle_graph_node),
    ik_validator_node_(ik_validator_node),
    cloud_box_filter_node_(cloud_box_filter_node),
    bridge_to_inference_node_(bridge_to_inference_node),
    world_state_node_(world_state_node)
    {
        // Parâmetros
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("subtrees_path", "/home/momesso/pibic/src/mobile_manipulation_packages/task_planning/bt/LLM_subtrees");

        yaml_file = this->get_parameter("yaml_file").as_string();
        subtrees_path_ = this->get_parameter("subtrees_path").as_string();

        // Subscriber para odometria
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&ServerNode::odom_callback, this, std::placeholders::_1));

       
        bt_xml_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/behavior_tree_xml", 10,
            std::bind(&ServerNode::on_bt_xml_received, this, std::placeholders::_1));

        publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("object_pose", 10);
        
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(this, "pick_object");
        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(this, "path");
        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(this, "controller");

        // Estados
        path_state_ = TaskState::IDLE;
        nav_state_ = TaskState::IDLE;
        manipulation_state_ = TaskState::IDLE;

        setup_behavior_tree_factory();

        bt_thread_ = std::thread(&ServerNode::bt_loop, this);

        RCLCPP_INFO(this->get_logger(), "ServerNode iniciado. Aguardando XML em /behavior_tree_xml");

        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&ServerNode::publish_pose, this));
    }

    ~ServerNode()
    {
        running_ = false;
        if (bt_thread_.joinable()) bt_thread_.join();
    }

private:
    // --- Injeção de Dependências ---
    std::shared_ptr<llms::WorldStateNode> world_state_node_;
    std::shared_ptr<drl_to_pick_cpp::BridgeToInference> bridge_to_inference_node_;
    std::shared_ptr<manipulation::CloudBoxFilter> cloud_box_filter_node_;
    std::shared_ptr<manipulation::IKValidator> ik_validator_node_;
    std::shared_ptr<navigation::SharedObstacleGraph> obstacle_graph_node_;
    std::shared_ptr<manipulation::ProjectedReachabilityAnalysis> reachability_node_;
    std::shared_ptr<manipulation::IsGripperHolding> gripper_monitor_node_;
    std::shared_ptr<storage_manager::StorageNode> storage_node_;
    std::shared_ptr<storage_manager::OrganizeNode> organize_node_;
    std::unique_ptr<BT::Groot2Publisher> groot_publisher_;

    // --- ROS Communication ---
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bt_xml_sub_;  

    // --- Action Clients ---
    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;

    // --- Action Handles ---
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::SharedPtr active_controller_goal_handle_;
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::SharedPtr active_path_goal_handle_;

    // --- Configuração ---
    std::string yaml_file;
    std::string subtrees_path_;
    std::unordered_set<std::string> authorized_labels;
    std::unordered_set<std::string> picked;

    struct ObjectInfo
    {
        std::string id;
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size;
    };

    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;
    ObjectInfo cached_object_;
    std::string current_target_id_ = "";
    geometry_msgs::msg::Pose current_target_pose_;

    rclcpp::TimerBase::SharedPtr timer_;

    // --- Behavior Tree ---
    std::thread bt_thread_;
    std::mutex bt_mutex_;
    std::mutex odom_mutex;
    BT::BehaviorTreeFactory factory_;  
    std::unique_ptr<BT::Tree> bt_tree_;  
    std::atomic<bool> has_new_tree_{false};  
    std::atomic<bool> running_{true};
    std::string pending_xml_; 
    std::mutex xml_mutex_;
    std::atomic<int> tree_counter_{0};  

    TaskState path_state_;
    TaskState nav_state_;
    TaskState manipulation_state_;

    std::mutex state_mutex_;
    std::mutex path_mutex_;

    nav_msgs::msg::Path last_calculated_path_;
    nav_msgs::msg::Path last_no_filter_calculated_path_;

    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0;
    bool has_new_object_ = false;

  
    void on_bt_xml_received(const std_msgs::msg::String::SharedPtr msg)
    {
        if (msg->data.empty())
        {
            RCLCPP_WARN(this->get_logger(), "XML vazio recebido, ignorando.");
            return;
        }

        if (bt_tree_ && bt_tree_->rootNode() && 
            bt_tree_->rootNode()->status() == BT::NodeStatus::RUNNING)
        {
            RCLCPP_WARN(this->get_logger(), "Árvore em execução! Ignorando novo XML.");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "XML recebido via tópico:\n%s", msg->data.c_str());

        {
            std::lock_guard<std::mutex> lock(xml_mutex_);
            pending_xml_ = msg->data;
            has_new_tree_ = true;
        }
    }

    
    void setup_behavior_tree_factory()
    {
        factory_.registerNodeType<ParallelAny>("ParallelAny");

    
        // --- IsReachable ---
        factory_.registerSimpleCondition("IsReachable", [&](BT::TreeNode &self)
        {
            auto target_pose_opt = self.getInput<geometry_msgs::msg::Pose>("target");
            auto authorized_id_opt = self.getInput<std::string>("object_id");
            auto robot_base_z_opt = self.getInput<double>("robot_base_z");
            auto max_reach_3d_opt = self.getInput<double>("max_reach_3d");

            if (!target_pose_opt || !authorized_id_opt || !robot_base_z_opt || !max_reach_3d_opt)
            {
                RCLCPP_ERROR(this->get_logger(), "IsReachable: Parâmetros faltando");
                return BT::NodeStatus::FAILURE;
            }

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

            std::sort(viable_points_3d.begin(), viable_points_3d.end(), 
            [&actual_robot_position](const auto& a, const auto& b) 
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

            auto best_base_opt = this->ik_validator_node_->find_best_base_position(
                viable_points_3d, target, true, this->obstacle_graph_node_, authorized_id);

            if (best_base_opt.has_value())
            {
                auto p = best_base_opt.value(); 
                float px = std::get<0>(p); 
                float py = std::get<1>(p); 

                float dx_curr = px - this->pose_x;
                float dy_curr = py - this->pose_y;
                float dist_sq = std::sqrt((dx_curr * dx_curr) + (dy_curr * dy_curr));

                const float threshold_sq = 0.07f; 

                if (dist_sq <= threshold_sq)
                {
                    RCLCPP_INFO(this->get_logger(), "Robô JÁ ESTÁ na posição ideal.");
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
                    RCLCPP_INFO(this->get_logger(), "Ajuste necessário: (%.2f, %.2f)", px, py);
                    return BT::NodeStatus::FAILURE;
                }
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Robô não conseguirá alcançar o objeto.");
                return BT::NodeStatus::FAILURE;
            }
        },
        {
            BT::InputPort<geometry_msgs::msg::Pose>("target"),
            BT::InputPort<std::string>("object_id"),
            BT::InputPort<double>("robot_base_z"),
            BT::InputPort<double>("max_reach_3d"),
            BT::OutputPort<geometry_msgs::msg::Pose>("adjustment_pose")
        });

        // --- IsStillReachable ---
        factory_.registerSimpleCondition("IsStillReachable", [&](BT::TreeNode &self)
        {
            bool reachable = this->ik_validator_node_->is_still_reachable(this->obstacle_graph_node_);
            return reachable ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
        });

        // --- ClearTarget ---
        factory_.registerSimpleAction("ClearTarget", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' liberado.", current_target_id_.c_str());
            current_target_id_ = ""; 
            return BT::NodeStatus::SUCCESS;
        });

        // --- IsPathClear ---
        factory_.registerSimpleCondition("IsPathClear", [&](BT::TreeNode& self)
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

        // --- IsGripperHoldingObject ---
        factory_.registerSimpleCondition("IsGripperHoldingObject",
            [this](BT::TreeNode& self) -> BT::NodeStatus
            {
                std::lock_guard<std::mutex> lock(bt_mutex_); 
                if (this->gripper_monitor_node_ && this->gripper_monitor_node_->checkIsHolding()) 
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

        // --- DecrementStorageCount ---
        factory_.registerSimpleAction("DecrementStorageCount", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("storage_id");
            if (!id_opt) return BT::NodeStatus::FAILURE;

            if (storage_node_)
            {
                storage_node_->incrementStorageCount(id_opt.value(), -1);
                RCLCPP_WARN(this->get_logger(), "ROLLBACK: Espaço liberado em '%s'.", id_opt.value().c_str());
            }
            return BT::NodeStatus::SUCCESS;
        },
        { BT::InputPort<std::string>("storage_id") });

        // --- ComputePath (Assíncrono) ---
        BT::NodeBuilder builder_compute = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    if (path_state_ == TaskState::SUCCESS) { 
                        path_state_ = TaskState::IDLE; 
                        return BT::NodeStatus::SUCCESS; 
                    }
                    if (path_state_ == TaskState::FAILURE) { 
                        path_state_ = TaskState::IDLE; 
                        return BT::NodeStatus::FAILURE; 
                    }
                    if (path_state_ == TaskState::RUNNING) return BT::NodeStatus::RUNNING;
                }

                auto target = self.getInput<geometry_msgs::msg::Pose>("target");
                if (!target) 
                {
                    RCLCPP_ERROR(this->get_logger(), "ComputePath: Target inválido");
                    return BT::NodeStatus::FAILURE;
                }

                this->send_path_goal(target.value());

                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    path_state_ = TaskState::RUNNING;
                }
                
                return BT::NodeStatus::RUNNING;
            });
        };
        factory_.registerBuilder(BT::TreeNodeManifest{
            BT::NodeType::ACTION, "ComputePath", 
            { BT::InputPort<geometry_msgs::msg::Pose>("target"), BT::InputPort<std::string>("planner") }, 
            {} 
        }, builder_compute);

        // --- FollowPath (Assíncrono) ---
        factory_.registerBuilder<AsyncAction>("FollowPath", [&](const std::string& name, const BT::NodeConfig& config)
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
                    RCLCPP_WARN(this->get_logger(), "FollowPath: HALT!");
                    this->cancel_controller_goal();
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    nav_state_ = TaskState::IDLE;
                }
            );
        });

        // --- PickObject (Assíncrono) ---
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

                    // Armazena em cache
                    cached_object_.id = id.value();
                    cached_object_.pose = target;
                    cached_object_.size = target_size;

                    target_size.x += 0.005;
                    target_size.y += 0.005;
                    target_size.z += 0.005;

                    this->cloud_box_filter_node_->set_bounding_box(target, target_size);

                    rclcpp::sleep_for(std::chrono::milliseconds(2000));
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
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory_.registerBuilder(BT::TreeNodeManifest{
            BT::NodeType::ACTION, "PickObject", 
            { 
                BT::InputPort<geometry_msgs::msg::Pose>("object_pose"), 
                BT::InputPort<geometry_msgs::msg::Vector3>("object_size"), 
                BT::InputPort<std::string>("id") 
            }, 
            {} 
        }, builder_pick);

        // --- PlaceObject (Assíncrono) ---
        BT::NodeBuilder builder_place = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE)
                {
                    auto pose = self.getInput<geometry_msgs::msg::Pose>("pose");
                    if (!pose) return BT::NodeStatus::FAILURE;

                    std::string id_dummy = cached_object_.id;
                    this->send_goal(id_dummy, pose.value(), false);
                    manipulation_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory_.registerBuilder(BT::TreeNodeManifest{
            BT::NodeType::ACTION, "PlaceObject", 
            { BT::InputPort<geometry_msgs::msg::Pose>("pose") }, 
            {} 
        }, builder_place);

        RCLCPP_INFO(this->get_logger(), "Todos os nós registrados na factory.");

    
        if (!subtrees_path_.empty())
        {
            try
            {
                factory_.registerBehaviorTreeFromFile(subtrees_path_ + "/pick.xml");
                RCLCPP_INFO(this->get_logger(), "Subtree 'Pick' registrada.");
                
                factory_.registerBehaviorTreeFromFile(subtrees_path_ + "/place.xml");
                RCLCPP_INFO(this->get_logger(), "Subtree 'Place' registrada.");
                
                factory_.registerBehaviorTreeFromFile(subtrees_path_ + "/goto_location.xml");
                RCLCPP_INFO(this->get_logger(), "Subtree 'GoToLocation' registrada.");
            }
            catch (const std::exception& e)
            {
                RCLCPP_ERROR(this->get_logger(), "Erro ao carregar subtrees: %s", e.what());
            }
        }

        RCLCPP_INFO(this->get_logger(), "Factory completamente configurada.");
    }

  
    void bt_loop()
    {
        rclcpp::Rate rate(50);

        while (running_ && rclcpp::ok())
        {
            if (has_new_tree_)
            {
                std::string xml_to_process;
                {
                    std::lock_guard<std::mutex> lock(xml_mutex_);
                    xml_to_process = pending_xml_;
                    pending_xml_.clear();
                    has_new_tree_ = false;
                }

                try
                {
                    int tree_id = tree_counter_++;
                    std::string unique_tree_name = "LLMPlan_" + std::to_string(tree_id);
                    
                    std::string modified_xml = xml_to_process;
                    
                    size_t pos = modified_xml.find("main_tree_to_execute=\"MainPlan\"");
                    if (pos != std::string::npos) {
                        modified_xml.replace(pos, 31, "main_tree_to_execute=\"" + unique_tree_name + "\"");
                    }
                    
                    pos = modified_xml.find("ID=\"MainPlan\"");
                    if (pos != std::string::npos) {
                        modified_xml.replace(pos, 13, "ID=\"" + unique_tree_name + "\"");
                    }
                    
                    RCLCPP_INFO(this->get_logger(), "Registrando árvore: %s", unique_tree_name.c_str());
                    
                    factory_.registerBehaviorTreeFromText(modified_xml);
                    
                    bt_tree_ = std::make_unique<BT::Tree>(
                        factory_.createTree(unique_tree_name)
                    );

                    groot_publisher_ = std::make_unique<BT::Groot2Publisher>(*bt_tree_, 1666);

                    RCLCPP_INFO(this->get_logger(), "Nova árvore '%s' criada com sucesso!", unique_tree_name.c_str());

                    RCLCPP_INFO(this->get_logger(), "\n%s", BT::WriteTreeToXML(*bt_tree_, false, false).c_str());

                }
                catch (const std::exception& e)
                {
                    RCLCPP_ERROR(this->get_logger(), "Erro ao criar árvore: %s", e.what());
                    bt_tree_.reset();
                    continue;
                }
            }

            // Executa tick se houver árvore
            if (bt_tree_ && bt_tree_->rootNode())
            {
                BT::NodeStatus status = bt_tree_->rootNode()->status();

                // Se não está em IDLE, continua executando
                if (status == BT::NodeStatus::RUNNING || status == BT::NodeStatus::IDLE)
                {
                    BT::NodeStatus result = bt_tree_->tickOnce();

                    if (result == BT::NodeStatus::SUCCESS)
                    {
                        RCLCPP_INFO(this->get_logger(), "========== ÁRVORE: SUCESSO ==========");
                        reset_states();
                        bt_tree_.reset();  // Libera a árvore para receber nova
                    }
                    else if (result == BT::NodeStatus::FAILURE)
                    {
                        RCLCPP_ERROR(this->get_logger(), "========== ÁRVORE: FALHOU ==========");
                        reset_states();
                        bt_tree_.reset();
                    }
                }
            }

            rate.sleep();
        }
    }

    void reset_states()
    {
        std::lock_guard<std::mutex> slock(state_mutex_);
        path_state_ = TaskState::IDLE;
        nav_state_ = TaskState::IDLE;
        manipulation_state_ = TaskState::IDLE;
    }

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


    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(odom_mutex);
        pose_x = msg->pose.pose.position.x;
        pose_y = msg->pose.pose.position.y;
        pose_z = 0.0;
    }

    void cancel_controller_goal()
    {
        if (this->active_controller_goal_handle_)
        {
            RCLCPP_WARN(this->get_logger(), "Cancelando navegação...");
            this->controller_client->async_cancel_goal(this->active_controller_goal_handle_);
        }
    }

    void send_path_goal(const geometry_msgs::msg::Pose & target_pose)
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            this->active_path_goal_handle_.reset();
        }

        if (!this->path_client->wait_for_action_server(std::chrono::seconds(2)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server 'path' indisponível!");
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

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SendGoalOptions();
        send_goal_options.goal_response_callback = 
            std::bind(&ServerNode::path_goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = 
            std::bind(&ServerNode::path_result_callback, this, std::placeholders::_1);

        this->path_client->async_send_goal(goal_msg, send_goal_options);
    }

    void path_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>> & goal_handle)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!goal_handle) 
        {
            RCLCPP_ERROR(this->get_logger(), "Path Planning REJEITADO.");
        } 
        else 
        {
            this->active_path_goal_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Path Planning aceito.");
        }
    }

    void path_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::WrappedResult & result)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);

        if (!this->active_path_goal_handle_ || result.goal_id != this->active_path_goal_handle_->get_goal_id()) {
            return;
        }
        this->active_path_goal_handle_.reset();

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            if (result.result->success && !result.result->path.poses.empty())
            {
                std::lock_guard<std::mutex> p_lock(path_mutex_);
                this->last_calculated_path_ = result.result->path;
                this->last_no_filter_calculated_path_ = result.result->path_without_filter;
                RCLCPP_INFO(this->get_logger(), "Path: SUCCESS (%zu poses)", this->last_calculated_path_.poses.size());
                path_state_ = TaskState::SUCCESS;
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Path: Caminho VAZIO.");
                path_state_ = TaskState::FAILURE;
            }
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Path: ABORTED/CANCELED");
            path_state_ = TaskState::FAILURE;
        }
    }

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

    void controller_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>> & goal_handle)
    {
        if (!goal_handle) 
        {
            RCLCPP_ERROR(this->get_logger(), "Controller REJEITADO");
            nav_state_ = TaskState::FAILURE;
        } 
        else 
        {
            this->active_controller_goal_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Controller aceito.");
        }
    }

    void controller_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::WrappedResult & result)
    {
        std::lock_guard<std::mutex> s_lock(state_mutex_);

        if (this->active_controller_goal_handle_ && result.goal_id != this->active_controller_goal_handle_->get_goal_id()) 
        {
            return;
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
            nav_state_ = TaskState::IDLE;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Controller: FALHOU.");
            nav_state_ = TaskState::FAILURE;
        }
    }

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

        RCLCPP_INFO(this->get_logger(), "Enviando Goal MANIPULATION (%s)...", pick ? "PICK" : "PLACE");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SendGoalOptions();
        send_goal_options.goal_response_callback = std::bind(&ServerNode::goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::result_callback, this, std::placeholders::_1);

        this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
    }

    void goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>> & goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Manipulation REJEITADO");
            manipulation_state_ = TaskState::FAILURE;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Manipulation aceito.");
        }
    }

    void result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success)
        {
            manipulation_state_ = TaskState::SUCCESS;
            RCLCPP_INFO(this->get_logger(), "MANIPULATION SUCCESS");
        }
        else
        {
            manipulation_state_ = TaskState::FAILURE;
            RCLCPP_ERROR(this->get_logger(), "MANIPULATION FAILED");
        }
    }

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

// ============================================================================
// MAIN
// ============================================================================

bool has_flag(const std::vector<std::string>& args, const std::string& flag) 
{
    return std::find(args.begin(), args.end(), flag) != args.end();
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    std::vector<std::string> args(argv, argv + argc);

    bool enable_organize = !has_flag(args, "--no-organize");
    bool enable_storage  = !has_flag(args, "--no-storage");
    bool enable_gripper  = !has_flag(args, "--no-gripper");

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

    rclcpp::NodeOptions world_state_node_opts;
    world_state_node_opts.arguments({"--ros-args", "-r", "__node:=world_state_node"});

    std::shared_ptr<storage_manager::OrganizeNode> organize_node = nullptr;
    std::shared_ptr<storage_manager::StorageNode> storage_node = nullptr;
    std::shared_ptr<manipulation::IsGripperHolding> gripper_node = nullptr;
    std::shared_ptr<manipulation::ProjectedReachabilityAnalysis> reachability_node = nullptr; 
    std::shared_ptr<manipulation::IKValidator> ik_validator_node = nullptr; 
    std::shared_ptr<manipulation::CloudBoxFilter> cloud_box_filter_node = nullptr; 
    std::shared_ptr<navigation::SharedObstacleGraph> obstacle_graph_node = nullptr; 
    std::shared_ptr<drl_to_pick_cpp::BridgeToInference> bridge_to_inference_node = nullptr; 
    std::shared_ptr<llms::WorldStateNode> world_state_node = nullptr; 

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

    world_state_node = std::make_shared<llms::WorldStateNode>(world_state_node_opts);
    executor.add_node(world_state_node);

    auto server_node = std::make_shared<ServerNode>(
        gripper_node, storage_node, organize_node, reachability_node, 
        obstacle_graph_node, ik_validator_node, cloud_box_filter_node, 
        bridge_to_inference_node, world_state_node
    );

    executor.add_node(server_node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}