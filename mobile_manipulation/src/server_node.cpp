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

#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/xml_parsing.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>

#include <yaml-cpp/yaml.h>

#include "mobile_manipulation_interfaces/srv/get_storage_info.hpp"

#include "mobile_manipulation_interfaces/action/pick_object.hpp"
#include "mobile_manipulation_interfaces/action/path.hpp"
#include "mobile_manipulation_interfaces/action/controller.hpp"

using namespace std::chrono_literals;

namespace BT
{
    template <>
    inline geometry_msgs::msg::Pose convertFromString(StringView)
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

class ParallelAny : public BT::ControlNode
{
public:
    ParallelAny(const std::string& name, const BT::NodeConfig& config)
        : BT::ControlNode(name, config) {}

    static BT::PortsList providedPorts() { return {}; } 

    BT::NodeStatus tick() override
    {
        size_t failure_count = 0;

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
                failure_count++;
            }
        }

        if (failure_count == children_nodes_.size())
        {
            haltChildren();
            return BT::NodeStatus::FAILURE;
        }

        return BT::NodeStatus::RUNNING;
    }

    void halt() override
    {
        haltChildren();
        BT::ControlNode::halt();
    }
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
     : Node("server_node")
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("bt_xml_path", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        std::string bt_xml_path = this->get_parameter("bt_xml_path").as_string();

        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&ServerNode::detection_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&ServerNode::odom_callback, this, std::placeholders::_1));

        storage_client_ = this->create_client<mobile_manipulation_interfaces::srv::GetStorageInfo>("get_storage_info");
        
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(this, "pick_object");
        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(this, "path");
        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(this, "controller");

        if(!yaml_file.empty()) 
        {
            loadLocationsFromYaml(yaml_file);
        }


        path_state_ = TaskState::IDLE;
        nav_state_ = TaskState::IDLE;
        manipulation_state_ = TaskState::IDLE;

        setup_behavior_tree(bt_xml_path);

        bt_thread_ = std::thread(&ServerNode::bt_loop, this);
            
        RCLCPP_INFO(this->get_logger(), "ServerNode iniciado (Com correção de Preempção).");
    } 

    ~ServerNode()
    {
        if (bt_thread_.joinable()) 
        {
            bt_thread_.join();
        }
    }

private:

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Client<mobile_manipulation_interfaces::srv::GetStorageInfo>::SharedPtr storage_client_;

    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;

    // --- VARIÁVEL NOVA PARA CORREÇÃO ---
    // Armazena o handle do objetivo de controle ATUALMENTE VÁLIDO
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::SharedPtr active_controller_goal_handle_;

    std::string yaml_file;

    std::unordered_set<std::string> authorized_labels;
    std::unordered_set<std::string> picked;
    
    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;
    std::pair<std::string, geometry_msgs::msg::Pose> cached_object_;
    
    std::thread bt_thread_;
    std::mutex bt_mutex_;

    BT::Tree bt_tree_;
    
    TaskState path_state_;
    TaskState nav_state_;
    TaskState manipulation_state_;
    
    nav_msgs::msg::Path last_calculated_path_; 
    std::mutex path_mutex_;

    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0;

    bool action_busy = false;
    bool has_new_object_ = false;


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

    void setup_behavior_tree(const std::string &xml_path)
    {
        BT::BehaviorTreeFactory factory;

        factory.registerNodeType<ParallelAny>("ParallelAny");

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


        factory.registerSimpleAction("DetectObject", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            if (!has_new_object_) return BT::NodeStatus::FAILURE;
            self.setOutput("output_pose", cached_object_.second);
            self.setOutput("output_id", cached_object_.first);
            picked.insert(cached_object_.first);
            pick_pose = cached_object_; 
            action_busy = true;      
            RCLCPP_INFO(this->get_logger(), "BT: Objeto '%s' detectado.", cached_object_.first.c_str());
            return BT::NodeStatus::SUCCESS;
        }, 
        { BT::OutputPort<geometry_msgs::msg::Pose>("output_pose"), BT::OutputPort<std::string>("output_id") });


        factory.registerSimpleAction("GetStorageInfo", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("object_id");
            if (!id_opt) return BT::NodeStatus::FAILURE;

            std::string id;
            size_t pos = id_opt.value().find('_');
            if (pos != std::string::npos) id = id_opt.value().substr(0, pos);

            geometry_msgs::msg::Pose current_obj_pose;
            {
                std::lock_guard<std::mutex> lock(bt_mutex_);
                current_obj_pose = pick_pose.second;
            }

            geometry_msgs::msg::Pose storage_pose;
            std::vector<double> storage_limits;

            bool success = this->get_storage_info_sync(id, current_obj_pose, storage_pose, storage_limits);

            if (success) 
            {
                self.setOutput("storage_pose", storage_pose);
                self.setOutput("storage_limits", storage_limits);
                return BT::NodeStatus::SUCCESS;
            } 
            return BT::NodeStatus::FAILURE;
        }, 
        { 
            BT::InputPort<std::string>("object_id"),
            BT::OutputPort<geometry_msgs::msg::Pose>("storage_pose"),
            BT::OutputPort<std::vector<double>>("storage_limits")
        });


        BT::NodeBuilder builder_compute = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (self.status() == BT::NodeStatus::IDLE && path_state_ != TaskState::IDLE) {
                    path_state_ = TaskState::IDLE;
                }

                if (path_state_ == TaskState::IDLE) 
                {
                    auto target = self.getInput<geometry_msgs::msg::Pose>("target");
                    if (!target) return BT::NodeStatus::FAILURE;
                    this->send_path_goal(target.value());
                    path_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(path_state_);
            });
        };
        factory.registerBuilder(
            BT::TreeNodeManifest{BT::NodeType::ACTION, "ComputePath", { BT::InputPort<geometry_msgs::msg::Pose>("target") }}, 
            builder_compute
        );


        factory.registerBuilder<AsyncAction>("NavigateTo", [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (self.status() == BT::NodeStatus::IDLE && nav_state_ != TaskState::IDLE) {
                    nav_state_ = TaskState::IDLE;
                }

                if (nav_state_ == TaskState::IDLE) 
                {
                    nav_msgs::msg::Path path_to_send;
                    bool has_path = false;

                    {
                        std::lock_guard<std::mutex> lock(path_mutex_);
                        if (!last_calculated_path_.poses.empty()) {
                            path_to_send = last_calculated_path_;
                            has_path = true;
                        }
                    }

                    if (has_path) 
                    {
                        RCLCPP_INFO(this->get_logger(), "BT: [NavigateTo] Caminho com %zu poses. Executando...", path_to_send.poses.size());
                        bool sent = this->send_controller_goal(path_to_send);
                        if(sent) {
                            nav_state_ = TaskState::RUNNING;
                            return BT::NodeStatus::RUNNING;
                        } else {
                            return BT::NodeStatus::FAILURE;
                        }
                    }
                    else 
                    {
                        return BT::NodeStatus::RUNNING;
                    }
                }
                return check_task_status(nav_state_);
            });
        });


        BT::NodeBuilder builder_pick = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE) 
                {
                    auto pose = self.getInput<geometry_msgs::msg::Pose>("pose");
                    auto id = self.getInput<std::string>("id");
                    if (!pose || !id) return BT::NodeStatus::FAILURE;
                    this->send_goal(id.value(), pose.value(), true);
                    manipulation_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory.registerBuilder(
            BT::TreeNodeManifest{BT::NodeType::ACTION, "PickObject", { BT::InputPort<geometry_msgs::msg::Pose>("pose"), BT::InputPort<std::string>("id") }}, 
            builder_pick
        );


        BT::NodeBuilder builder_place = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE) {
                    auto pose = self.getInput<geometry_msgs::msg::Pose>("pose");
                    auto limits = self.getInput<std::vector<double>>("limits"); 
                    if (!pose || !limits) return BT::NodeStatus::FAILURE;
                    std::string id_dummy = cached_object_.first; 
                    this->send_goal(id_dummy, pose.value(), false); 
                    manipulation_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory.registerBuilder(
            BT::TreeNodeManifest{BT::NodeType::ACTION, "PlaceObject", { BT::InputPort<geometry_msgs::msg::Pose>("pose"), BT::InputPort<std::vector<double>>("limits") }}, 
            builder_place
        );


        try 
        {
            bt_tree_ = factory.createTreeFromFile(xml_path);
        } 
        catch (const std::exception &e) 
        {
            RCLCPP_ERROR(this->get_logger(), "Erro Fatal ao criar Tree: %s", e.what());
        }
    }

    void bt_loop()
    {
        rclcpp::Rate rate(10); 

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

            if (status == BT::NodeStatus::RUNNING || new_obj)
            {
                BT::NodeStatus result = bt_tree_.tickOnce();

                if (result == BT::NodeStatus::SUCCESS || result == BT::NodeStatus::FAILURE)
                {
                    std::lock_guard<std::mutex> lock(bt_mutex_);
                    has_new_object_ = false;
                    action_busy = false; 
                    
                    if (result == BT::NodeStatus::FAILURE) 
                    {
                        picked.erase(cached_object_.first);
                    }
                    
                    path_state_ = TaskState::IDLE;
                    nav_state_ = TaskState::IDLE;
                    manipulation_state_ = TaskState::IDLE;

                    RCLCPP_INFO(this->get_logger(), "--- Missão BT Finalizada: %s ---", toStr(result).c_str());
                }
            }
            rate.sleep();
        }
    }

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

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) 
    {
        pose_x = msg->pose.pose.position.x;
        pose_y = msg->pose.pose.position.y;
        pose_z = 0.0;
    }

    void detection_callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(bt_mutex_);

        if(action_busy == true || has_new_object_ == true) return;

        for (const auto &det : msg->detections)
        {
            if (det.results.empty()) continue;
            std::string raw_id = det.results[0].hypothesis.class_id;
            
            std::string id = raw_id;
            size_t pos = raw_id.find('_'); 
            if (pos != std::string::npos) id = raw_id.substr(0, pos);

            if (authorized_labels.find(id) == authorized_labels.end()) continue;
            
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
    

    bool get_storage_info_sync(const std::string &id, const geometry_msgs::msg::Pose &current_pose, 
        geometry_msgs::msg::Pose &pose_out, std::vector<double> &limits_out) 
    {
        if (!storage_client_->wait_for_service(std::chrono::seconds(1))) 
        {
            RCLCPP_ERROR(this->get_logger(), "Serviço de storage indisponível.");
            return false;
        }

        auto request = std::make_shared<mobile_manipulation_interfaces::srv::GetStorageInfo::Request>();
        request->object_id = id;            
        request->pose = current_pose;       

        auto result_future = storage_client_->async_send_request(request);

        if (result_future.wait_for(std::chrono::seconds(3)) == std::future_status::ready)
        {
            auto response = result_future.get();
            if (response->success) 
            {
                pose_out = response->pose;
                limits_out.clear();
                for(auto val : response->limits) limits_out.push_back(static_cast<double>(val));
                return true;
            } 
            else 
            {
                RCLCPP_WARN(this->get_logger(), "Storage service retornou false.");
                return false;
            }
        }
        RCLCPP_ERROR(this->get_logger(), "Timeout esperando resposta do Storage Service.");
        return false;
    }

    // --- Path Callbacks ---

    void send_path_goal(const geometry_msgs::msg::Pose & target_pose)
    {
        if (!this->path_client->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server 'path' not available");
            action_busy = false;
            path_state_ = TaskState::FAILURE; 
            return;
        }

       
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

    void path_feedback_callback(
        rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::SharedPtr,
        const std::shared_ptr<const mobile_manipulation_interfaces::action::Path::Feedback> feedback)
    {
        {
            std::lock_guard<std::mutex> lock(path_mutex_);
            this->last_calculated_path_ = feedback->path;
        }

        if (feedback->recalculating_path)
        {
            send_controller_goal(feedback->path);        
        }
    }

    void path_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>> & goal_handle)
    {
        if (!goal_handle)
        {
            RCLCPP_ERROR(this->get_logger(), "Goal PATH rejeitado");
            path_state_ = TaskState::FAILURE;
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Goal PATH aceito, calculando...");
        }
    }

    void path_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            if (result.result->success) path_state_ = TaskState::SUCCESS;
            else path_state_ = TaskState::FAILURE;
        }
        
    }

    // --- Controller Callbacks ---

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
            RCLCPP_ERROR(this->get_logger(), "Goal CONTROLLER rejeitado");
            nav_state_ = TaskState::FAILURE;
        }
        else
        {
            // CORREÇÃO: Salva o handle atual como o "válido"
            this->active_controller_goal_handle_ = goal_handle;
            
            RCLCPP_INFO(this->get_logger(), "Goal CONTROLLER aceito, executando...");
        }
    }

    void controller_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::WrappedResult & result)
    {
        // CORREÇÃO CRÍTICA:
        // Verifica se o resultado recebido é do objetivo ATUAL.
        // Se o ID for diferente, significa que é o fantasma de um objetivo antigo abortado.
        if (this->active_controller_goal_handle_ && result.goal_id != this->active_controller_goal_handle_->get_goal_id())
        {
            RCLCPP_WARN(this->get_logger(), "Ignorando resultado de goal antigo (Preempção ocorrida).");
            return; 
        }

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
            {
                std::lock_guard<std::mutex> lock(path_mutex_);
                last_calculated_path_.poses.clear();
            }
            
            path_state_ = TaskState::IDLE;
            nav_state_ = TaskState::SUCCESS;
            RCLCPP_INFO(this->get_logger(), "Navegação concluída! Dados de caminho limpos.");
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Goal CONTROLLER falhou/abortou (Falha Real)");
            nav_state_ = TaskState::FAILURE;
        }
        
        // Limpa o ponteiro
        this->active_controller_goal_handle_.reset();
    }

    // --- Pick/Place Callbacks ---

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

    void result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success)
        {
            manipulation_state_ = TaskState::SUCCESS;
            RCLCPP_INFO(this->get_logger(), "PICK SUCCESS");
        }
        else
        {
            picked.erase(std::get<0>(pick_pose));
            manipulation_state_ = TaskState::FAILURE;
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