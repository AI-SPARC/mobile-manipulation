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

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/xml_parsing.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"
#include <nav_msgs/msg/odometry.hpp>

#include <yaml-cpp/yaml.h>

#include "mobile_manipulation_interfaces/srv/stop_pose.hpp"
#include "mobile_manipulation_interfaces/action/pick_object.hpp"
#include "mobile_manipulation_interfaces/action/path.hpp"
#include "mobile_manipulation_interfaces/action/controller.hpp"

using namespace std::chrono_literals;

namespace BT
{
    template <>
    inline geometry_msgs::msg::Pose convertFromString(StringView key)
    {
        return geometry_msgs::msg::Pose();
    }
}

enum class TaskState
{
    IDLE,
    RUNNING,
    SUCCESS,
    FAILURE
};


class AsyncAction : public BT::StatefulActionNode
{
public:
    AsyncAction(const std::string& name, const BT::NodeConfig& config, 
                std::function<BT::NodeStatus(BT::TreeNode&)> tick_fun)
        : BT::StatefulActionNode(name, config), tick_fun_(tick_fun) {}

    BT::NodeStatus onStart() override { return tick_fun_(*this); }
    BT::NodeStatus onRunning() override { return tick_fun_(*this); }
    void onHalted() override {}

private:
    std::function<BT::NodeStatus(BT::TreeNode&)> tick_fun_;
};

class ServerNode : public rclcpp::Node 
{
public:
    ServerNode()
     : Node("pick_and_organize")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("label_to_storage_yaml_file", "");
        this->declare_parameter<std::string>("storage_poses_yaml_file", "");
        this->declare_parameter<std::string>("bt_xml_path", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        label_to_storage_yaml_file = this->get_parameter("label_to_storage_yaml_file").as_string();
        storage_poses_yaml_file = this->get_parameter("storage_poses_yaml_file").as_string();
        std::string bt_xml_path = this->get_parameter("bt_xml_path").as_string();

        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&ServerNode::detection_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&ServerNode::odom_callback, this, std::placeholders::_1));

        client_ = this->create_client<mobile_manipulation_interfaces::srv::StopPose>("stop_pose");
        
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(this, "pick_object");
        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(this, "path");
        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(this, "controller");

        if(!yaml_file.empty()) 
        {
            loadLocationsFromYaml(yaml_file);
        }

        if(!label_to_storage_yaml_file.empty())
        {
            loadLabelToStorage(label_to_storage_yaml_file);
        }

        if(!storage_poses_yaml_file.empty())
        {
            loadStoragePoses(storage_poses_yaml_file);
        }

        setup_behavior_tree(bt_xml_path);

        bt_timer_ = this->create_wall_timer(100ms, std::bind(&ServerNode::bt_tick_callback, this));
            
        RCLCPP_INFO(this->get_logger(), "ServerNode iniciado com Behavior Tree.");
    } 

private:
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Client<mobile_manipulation_interfaces::srv::StopPose>::SharedPtr client_;

    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;

    std::string yaml_file, label_to_storage_yaml_file, storage_poses_yaml_file;

    std::unordered_set<std::string> authorized_labels;
    std::unordered_set<std::string> picked;
    std::unordered_map<std::string, std::vector<geometry_msgs::msg::Pose>> storage;
    std::unordered_map<std::string, std::vector<std::string>> labels_to_storage;

    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;

    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0;

    // Flags de estado
    bool storing = false;
    bool action_busy = false;
    std::pair<std::string, geometry_msgs::msg::Pose> cached_object_;
    bool has_new_object_ = false;

    BT::Tree bt_tree_;
    rclcpp::TimerBase::SharedPtr bt_timer_;
    TaskState current_task_state_ = TaskState::IDLE;
    nav_msgs::msg::Path last_calculated_path_; 

    BT::NodeStatus check_task_status()
    {
        if (current_task_state_ == TaskState::SUCCESS)
        {
            current_task_state_ = TaskState::IDLE;
            return BT::NodeStatus::SUCCESS;
        }
        else if (current_task_state_ == TaskState::FAILURE)
        {
            current_task_state_ = TaskState::IDLE;
            return BT::NodeStatus::FAILURE;
        }
        return BT::NodeStatus::RUNNING;
    }

    // --------------------------------------------------------------------------------
    // CONFIGURAÇÃO DA BEHAVIOR TRfEE
    // --------------------------------------------------------------------------------

    void setup_behavior_tree(const std::string &xml_path)
    {
        BT::BehaviorTreeFactory factory;

        // ==============================================================================
        // 1. CONDIÇÕES
        // ==============================================================================

        factory.registerSimpleCondition("IsObjectDetected", [&](BT::TreeNode &)
        {
            return has_new_object_ ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
        });

        factory.registerSimpleCondition("IsRobotNear", [&](BT::TreeNode &self)
        {
            auto target = self.getInput<geometry_msgs::msg::Pose>("target");
            auto threshold = self.getInput<double>("threshold");
            if (!target || !threshold) return BT::NodeStatus::FAILURE;

            double dx = this->pose_x - target.value().position.x;
            double dy = this->pose_y - target.value().position.y;
            double dist = std::sqrt(dx*dx + dy*dy);
            return (dist <= threshold.value()) ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
        }, 
        { BT::InputPort<geometry_msgs::msg::Pose>("target"), BT::InputPort<double>("threshold") });

        factory.registerSimpleCondition("CheckIfPickable", [&](BT::TreeNode &self)
        {
            return self.getInput<geometry_msgs::msg::Pose>("target") ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
        }, { BT::InputPort<geometry_msgs::msg::Pose>("target") });

        factory.registerSimpleCondition("IsDestinationFree", [&](BT::TreeNode &)
        {
            return BT::NodeStatus::SUCCESS;
        }, { BT::InputPort<geometry_msgs::msg::Pose>("dest") });


        // ==============================================================================
        // 2. AÇÕES SÍNCRONAS
        // ==============================================================================

        factory.registerSimpleAction("DetectObject", [&](BT::TreeNode &self)
        {
            if (!has_new_object_) return BT::NodeStatus::FAILURE;
            std::string raw_id = cached_object_.first;
            geometry_msgs::msg::Pose obj_pose = cached_object_.second;

            picked.insert(raw_id);
            pick_pose = cached_object_; 
            self.setOutput("output_pose", obj_pose);
            self.setOutput("output_id", raw_id);
            action_busy = true;      
            
            RCLCPP_INFO(this->get_logger(), "BT: Processando objeto salvo: %s", raw_id.c_str());
            return BT::NodeStatus::SUCCESS;
        }, 
        { BT::OutputPort<geometry_msgs::msg::Pose>("output_pose"), BT::OutputPort<std::string>("output_id") });

        factory.registerSimpleAction("GetStorageLocation", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("object_id");
            if (!id_opt) {
                RCLCPP_ERROR(this->get_logger(), "GetStorageLocation: ID não fornecido!");
                return BT::NodeStatus::FAILURE;
            }
            std::string full_id = id_opt.value();
            std::string clean_id = full_id;
            size_t pos = full_id.find('_'); 
            if (pos != std::string::npos) clean_id = full_id.substr(0, pos);

            try {
                auto [storage_name, storage_pose] = getClosestStorage(clean_id, this->pose_x, this->pose_y, 0.0);
                self.setOutput("output_pose", storage_pose);
                RCLCPP_INFO(this->get_logger(), "BT: Storage encontrado: %s", storage_name.c_str());
                return BT::NodeStatus::SUCCESS;
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "BT Error: %s", e.what());
                return BT::NodeStatus::FAILURE;
            }
        }, 
        { BT::InputPort<std::string>("object_id"), BT::OutputPort<geometry_msgs::msg::Pose>("output_pose") });


        // ==============================================================================
        // 3. AÇÕES ASSÍNCRONAS (CORRIGIDO REGISTRO DE BUILDER)
        // ==============================================================================

        // --- ComputePath ---
        BT::NodeBuilder builder_compute = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (current_task_state_ == TaskState::IDLE) {
                    auto target_opt = self.getInput<geometry_msgs::msg::Pose>("target");
                    if (!target_opt) {
                        RCLCPP_ERROR(this->get_logger(), "ComputePath: Sem target!");
                        return BT::NodeStatus::FAILURE;
                    }
                    this->send_path_goal(target_opt.value());
                    current_task_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status();
            });
        };
        // Criação manual do manifesto para incluir as portas
        BT::TreeNodeManifest manifest_compute = { BT::NodeType::ACTION, "ComputePath", { BT::InputPort<geometry_msgs::msg::Pose>("target") } };
        factory.registerBuilder(manifest_compute, builder_compute);


        // --- NavigateTo (Sem portas, usa registro simplificado) ---
        factory.registerBuilder<AsyncAction>("NavigateTo", [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &)
            {
                if (current_task_state_ == TaskState::IDLE) {
                    this->send_controller_goal(this->last_calculated_path_);
                    current_task_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status();
            });
        });


        // --- PickObject / PlaceObject ---
        auto pick_place_lambda = [&](BT::TreeNode &self)
        {
            if (current_task_state_ == TaskState::IDLE) {
                auto pose_opt = self.getInput<geometry_msgs::msg::Pose>("pose");
                auto id_opt = self.getInput<std::string>("id");

                if (!pose_opt || !id_opt) {
                    RCLCPP_ERROR(this->get_logger(), "Pick/Place: Faltando pose ou ID!");
                    return BT::NodeStatus::FAILURE;
                }
                bool is_storing = (self.name() == "PlaceObject");
                this->storing = is_storing;
                this->send_goal(id_opt.value(), pose_opt.value());
                current_task_state_ = TaskState::RUNNING;
                return BT::NodeStatus::RUNNING;
            }
            return check_task_status();
        };

        // Builder genérico para Pick/Place
        BT::NodeBuilder builder_pick_place = [&](const std::string& name, const BT::NodeConfig& config) {
            return std::make_unique<AsyncAction>(name, config, pick_place_lambda);
        };

        // Definição das portas
        BT::PortsList pick_ports = { 
            BT::InputPort<geometry_msgs::msg::Pose>("pose"), 
            BT::InputPort<std::string>("id") 
        };

        // Registro manual dos manifestos
        BT::TreeNodeManifest manifest_pick = { BT::NodeType::ACTION, "PickObject", pick_ports };
        factory.registerBuilder(manifest_pick, builder_pick_place);

        BT::TreeNodeManifest manifest_place = { BT::NodeType::ACTION, "PlaceObject", pick_ports };
        factory.registerBuilder(manifest_place, builder_pick_place);


        // ==============================================================================
        // 4. DUMMIES & UTILS
        // ==============================================================================

        factory.registerSimpleAction("ClearDestination", [&](BT::TreeNode &)
        {
             std::this_thread::sleep_for(500ms);
             return BT::NodeStatus::SUCCESS;
        }, { BT::InputPort<geometry_msgs::msg::Pose>("dest") });

        factory.registerSimpleAction("Log", [&](BT::TreeNode &self)
        {
            auto msg = self.getInput<std::string>("msg");
            if(msg) RCLCPP_INFO(this->get_logger(), "%s", msg.value().c_str());
            return BT::NodeStatus::SUCCESS;
        }, { BT::InputPort<std::string>("msg") });

        factory.registerSimpleAction("Wait", [&](BT::TreeNode &self)
        { 
             auto msec = self.getInput<int>("msec");
             if(msec) std::this_thread::sleep_for(std::chrono::milliseconds(msec.value()));
             return BT::NodeStatus::SUCCESS; 
        }, { BT::InputPort<int>("msec") });

        factory.registerSimpleAction("ResetHardware", [&](BT::TreeNode &){ return BT::NodeStatus::SUCCESS; });
        factory.registerSimpleAction("StopMovement", [&](BT::TreeNode &){ return BT::NodeStatus::SUCCESS; });
        factory.registerSimpleAction("RecalibrateArm", [&](BT::TreeNode &){ return BT::NodeStatus::SUCCESS; });
        factory.registerSimpleAction("AbortMission", [&](BT::TreeNode &){ return BT::NodeStatus::FAILURE; });

        // ==============================================================================
        // 5. CARREGAMENTO DO ARQUIVO
        // ==============================================================================
        try
        {
            bt_tree_ = factory.createTreeFromFile(xml_path);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Erro BT XML: %s", e.what());
        }
    }


    void bt_tick_callback()
    {
        if(bt_tree_.rootNode())
        {
            bt_tree_.tickOnce();
        }
    }

    // --------------------------------------------------------------------------------
    // FUNÇÕES DE CARREGAMENTO (YAML)
    // --------------------------------------------------------------------------------

    void loadLocationsFromYaml(const std::string &yaml_path)
    {
        try
        {
            YAML::Node config = YAML::LoadFile(yaml_path);
            for (const auto &label_node : config)
            {
                authorized_labels.insert(label_node.first.as<std::string>());
            }
        }
        catch (const YAML::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load YAML: %s", e.what());
        }
    }

    void loadLabelToStorage(const std::string &yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);
        for (auto it = config.begin(); it != config.end(); ++it)
        {
            std::string group_name = it->first.as<std::string>();  
            const YAML::Node &entries = it->second;                
            std::vector<std::string> storages;
            for (const auto &entry : entries)
            {
                const YAML::Node &value = entry["storage"];
                if (value)
                {
                    storages.push_back(value.as<std::string>());
                }
            }
            labels_to_storage[group_name] = storages;
        }
    }

    void loadStoragePoses(const std::string &yaml_file)
    {
        YAML::Node config = YAML::LoadFile(yaml_file);
        for (auto it = config.begin(); it != config.end(); ++it)
        {
            std::string storage_name = it->first.as<std::string>();
            const YAML::Node &locations = it->second;
            std::vector<geometry_msgs::msg::Pose> poses;
            for (const auto &loc : locations)
            {
                for (auto loc_it = loc.begin(); loc_it != loc.end(); ++loc_it)
                {
                    const YAML::Node &loc_data = loc_it->second;
                    geometry_msgs::msg::Pose pose;
                    const YAML::Node &pos = loc_data["position"];
                    pose.position.x = pos[0].as<double>();
                    pose.position.y = pos[1].as<double>();
                    pose.position.z = pos[2].as<double>();
                    if (loc_data["orientation"])
                    {
                        const YAML::Node &ori = loc_data["orientation"];
                        pose.orientation.x = ori[0].as<double>();
                        pose.orientation.y = ori[1].as<double>();
                        pose.orientation.z = ori[2].as<double>();
                        pose.orientation.w = ori[3].as<double>();
                    }
                    else
                    {
                        pose.orientation.x = 0.0; pose.orientation.y = 0.0; pose.orientation.z = 0.0; pose.orientation.w = 1.0;
                    }
                    poses.push_back(pose);
                }
            }
            storage[storage_name] = poses;
        }
    }

    std::pair<std::string, geometry_msgs::msg::Pose> getClosestStorage(const std::string& label, double px, double py, double pz)
    {
        double best_dist = std::numeric_limits<double>::max();
        std::string best_storage_name;
        geometry_msgs::msg::Pose best_pose;

        if (!labels_to_storage.count(label))
        {
            throw std::runtime_error("Label não encontrada: " + label);
        }

        const auto& storage_list = labels_to_storage[label];
        for (const auto& storage_name : storage_list)
        {
            if (!storage.count(storage_name)) continue;
            const auto& poses = storage.at(storage_name);
            for (const auto& pose : poses)
            {
                double dx = pose.position.x - px;
                double dy = pose.position.y - py;
                double dz = pose.position.z - pz;
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < best_dist)
                {
                    best_dist = dist; best_storage_name = storage_name; best_pose = pose;
                }
            }
        }
        if (best_storage_name.empty())
        {
            throw std::runtime_error("Nenhum storage encontrado para a label: " + label);
        }
        return { best_storage_name, best_pose };
    }

    // ---------
    // CALLBACKS
    // ---------

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) 
    {
        pose_x = msg->pose.pose.position.x;
        pose_y = msg->pose.pose.position.y;
        pose_z = 0.0;
    }

    void detection_callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        if(action_busy == true || has_new_object_ == true) 
        {
            return;
        }

        for (const auto &det : msg->detections)
        {
            if (det.results.empty()) continue;
            std::string raw_id = det.results[0].hypothesis.class_id;
            
            // Lógica de ID limpo
            std::string id = raw_id;
            size_t pos = raw_id.find('_'); 
            if (pos != std::string::npos)
            {
                id = raw_id.substr(0, pos);
            }

            if (authorized_labels.find(id) == authorized_labels.end())
            {
                continue;
            }
            
            if(picked.find(raw_id) == picked.end())
            {
                
                geometry_msgs::msg::Pose pose;
                pose.position = det.bbox.center.position;
                pose.orientation = det.bbox.center.orientation;

                cached_object_ = std::make_pair(raw_id, pose);
                has_new_object_ = true; 

                break;
            }
        }
    }
    

    // --------------------------------------------------------------------------------
    // CLIENT FUNCTIONS (Adaptadas para BT)
    // --------------------------------------------------------------------------------

    void send_request(geometry_msgs::msg::Pose pose)
    {
        auto request = std::make_shared<mobile_manipulation_interfaces::srv::StopPose::Request>();
        request->stop_pose = pose;
        client_->async_send_request(request,
            [this](rclcpp::Client<mobile_manipulation_interfaces::srv::StopPose>::SharedFuture future_response) 
            {
                auto response = future_response.get();  
                if (response->success)
                {
                    RCLCPP_INFO(this->get_logger(), "Service executado com sucesso!");
                }
                else
                {
                    RCLCPP_WARN(this->get_logger(), "Falha ao executar service");
                }
            }
        );
    }

    // --- Action: Path ---

    void send_path_goal(const geometry_msgs::msg::Pose & target_pose)
    {
        if (!this->path_client->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            action_busy = false;
            current_task_state_ = TaskState::FAILURE; 
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::Path::Goal();
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "BT: Enviando Goal (Pose) para A*...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SendGoalOptions();
        send_goal_options.goal_response_callback = std::bind(&ServerNode::path_goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::path_result_callback, this, std::placeholders::_1);
        send_goal_options.feedback_callback = std::bind(&ServerNode::path_feedback_callback, this, std::placeholders::_1, std::placeholders::_2);

        this->path_client->async_send_goal(goal_msg, send_goal_options);
    }

    void path_feedback_callback(
        rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::SharedPtr,
        const std::shared_ptr<const mobile_manipulation_interfaces::action::Path::Feedback> feedback)
    {
        if (feedback->recalculating_path == true)
        {
            send_request(feedback->stop_pose);
        }
    }

    void path_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>> & goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal PATH rejeitado");
            current_task_state_ = TaskState::FAILURE;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Goal PATH aceito, executando...");
        }
    }

    void path_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            this->last_calculated_path_ = result.result->path;
            current_task_state_ = TaskState::SUCCESS;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Goal PATH falhou/abortou");
            current_task_state_ = TaskState::FAILURE;
        }
    }

    // --- Action: Controller ---

    void send_controller_goal(const nav_msgs::msg::Path &target_path)
    {
        if (!this->controller_client->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            current_task_state_ = TaskState::FAILURE;
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::Controller::Goal();
        goal_msg.path = target_path;

        RCLCPP_INFO(this->get_logger(), "BT: Enviando Goal (Pose) para CONTROLLER...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SendGoalOptions();
        send_goal_options.goal_response_callback = std::bind(&ServerNode::controller_goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::controller_result_callback, this, std::placeholders::_1);

        this->controller_client->async_send_goal(goal_msg, send_goal_options);
    }

    void controller_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>> & goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal CONTROLLER rejeitado");
            current_task_state_ = TaskState::FAILURE;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Goal CONTROLLER aceito, executando...");
        }
    }

    void controller_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            current_task_state_ = TaskState::SUCCESS;
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Goal CONTROLLER falhou/abortou");
            current_task_state_ = TaskState::FAILURE;
        }
    }

    // --- Action: PickObject ---

    void send_goal(const std::string id, const geometry_msgs::msg::Pose & target_pose)
    {
        if (!this->client_ptr_->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server not available");
            current_task_state_ = TaskState::FAILURE;
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::PickObject::Goal();
        goal_msg.obstacle_id = id;
        goal_msg.pick = storing;
        goal_msg.pose = target_pose;

        RCLCPP_INFO(this->get_logger(), "BT: Enviando Goal (Pose) para MANIPULATION...");

        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SendGoalOptions();
        send_goal_options.goal_response_callback = std::bind(&ServerNode::goal_response_callback, this, std::placeholders::_1);
        send_goal_options.result_callback = std::bind(&ServerNode::result_callback, this, std::placeholders::_1);

        this->client_ptr_->async_send_goal(goal_msg, send_goal_options);
    }

    void goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>> & goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal PICK rejeitado");
            current_task_state_ = TaskState::FAILURE;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Goal PICK aceito, executando...");
        }
    }

    void result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success)
        {
            // Lógica de sucesso (Toggle Storing mode)
            if(storing == false)
            {
                storing = true;
            }
            else
            {
                storing = false;
            }

            current_task_state_ = TaskState::SUCCESS;
            RCLCPP_INFO(this->get_logger(), "PICK SUCCESS");
        }
        else
        {
            // Falhou ao pegar, remove do set de picked para tentar de novo ou ignorar
            picked.erase(std::get<0>(pick_pose));
            current_task_state_ = TaskState::FAILURE;
            RCLCPP_ERROR(this->get_logger(), "PICK FAILED or ABORTED");
        }
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ServerNode>());
  rclcpp::shutdown();
  return 0;
}