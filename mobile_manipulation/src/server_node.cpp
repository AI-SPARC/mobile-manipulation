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

#include "mobile_manipulation_interfaces/srv/get_storage_info.hpp"
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

        client_ = this->create_client<mobile_manipulation_interfaces::srv::StopPose>("stop_pose");
        storage_client_ = this->create_client<mobile_manipulation_interfaces::srv::GetStorageInfo>("get_storage_info");
        
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(this, "pick_object");
        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(this, "path");
        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(this, "controller");

        if(!yaml_file.empty()) 
        {
            loadLocationsFromYaml(yaml_file);
        }

        setup_behavior_tree(bt_xml_path);

        bt_thread_ = std::thread(&ServerNode::bt_loop, this);
            
        RCLCPP_INFO(this->get_logger(), "ServerNode iniciado com Behavior Tree.");
    } 

    ~ServerNode()
    {
        if (bt_thread_.joinable()) 
        {
            bt_thread_.join();
        }
    }

private:

    struct StorageResult 
    {
        bool ready = false;
        bool success = false;
        geometry_msgs::msg::Pose pose;
        std::vector<float> limits;
    }last_storage_result_;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp::Client<mobile_manipulation_interfaces::srv::GetStorageInfo>::SharedPtr storage_client_;
    rclcpp::Client<mobile_manipulation_interfaces::srv::StopPose>::SharedPtr client_;

    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;
    
    std::string yaml_file;

    std::unordered_set<std::string> authorized_labels;
    std::unordered_set<std::string> picked;
    std::unordered_map<std::string, std::vector<geometry_msgs::msg::Pose>> storage;
    std::unordered_map<std::string, std::vector<std::string>> labels_to_storage;

    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;
    std::pair<std::string, geometry_msgs::msg::Pose> cached_object_;
    
    std::thread bt_thread_;
    std::mutex bt_mutex_;

    BT::Tree bt_tree_;
    TaskState current_task_state_ = TaskState::IDLE;
    nav_msgs::msg::Path last_calculated_path_; 
    
    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0;

    bool storing = false;
    bool action_busy = false;
    
    bool has_new_object_ = false;



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


    void setup_behavior_tree(const std::string &xml_path)
    {
        BT::BehaviorTreeFactory factory;

        // ==============================================================================
        // 1. CONDIÇÕES (Leitura de Estado)
        // ==============================================================================

        // [Condition] IsObjectDetected
        // Protegido por Mutex pois lê flag compartilhada
        factory.registerSimpleCondition("IsObjectDetected", [&](BT::TreeNode &)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            return has_new_object_ ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
        });

        // [Condition] IsRobotNear
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
        // 2. AÇÕES SÍNCRONAS (Rápidas ou Bloqueantes na Thread da BT)
        // ==============================================================================

        // [Action] DetectObject (Consome dados do Cache)
        factory.registerSimpleAction("DetectObject", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_); // TRAVA CRÍTICA

            if (!has_new_object_) return BT::NodeStatus::FAILURE;
            
            std::string raw_id = cached_object_.first;
            geometry_msgs::msg::Pose obj_pose = cached_object_.second;

            picked.insert(raw_id);
            pick_pose = cached_object_; 
            
            self.setOutput("output_pose", obj_pose);
            self.setOutput("output_id", raw_id);

            // Nota: action_busy continua true para impedir novas detecções durante a missão
            action_busy = true;      
            
            RCLCPP_INFO(this->get_logger(), "BT: Objeto '%s' confirmado. Iniciando sequência.", raw_id.c_str());
            return BT::NodeStatus::SUCCESS;
        }, 
        { BT::OutputPort<geometry_msgs::msg::Pose>("output_pose"), BT::OutputPort<std::string>("output_id") });


        // // [Action] GetStorageLocation (Síncrono/Bloqueante via Serviço)
        // factory.registerSimpleAction("GetStorageLocation", [&](BT::TreeNode &self)
        // {
        //     auto id_opt = self.getInput<std::string>("object_id");
        //     if (!id_opt) {
        //         RCLCPP_ERROR(this->get_logger(), "GetStorageLocation: ID ausente.");
        //         return BT::NodeStatus::FAILURE;
        //     }

        //     geometry_msgs::msg::Pose object_current_pose;
        //     {
        //         std::lock_guard<std::mutex> lock(bt_mutex_);
        //         object_current_pose = pick_pose.second;
        //     }

        //     geometry_msgs::msg::Pose storage_pose_result;

        //     bool success = this->get_storage_info_sync(id_opt.value(), object_current_pose, storage_pose_result);

        //     if (success) 
        //     {
        //         self.setOutput("output_pose", storage_pose_result);
        //         return BT::NodeStatus::SUCCESS;
        //     } 
        //     else 
        //     {
        //         return BT::NodeStatus::FAILURE;
        //     }
        // }, 
        // { BT::InputPort<std::string>("object_id"), BT::OutputPort<geometry_msgs::msg::Pose>("output_pose") });


        // ==============================================================================
        // 3. AÇÕES ASSÍNCRONAS (Actions ROS Longas -> Usam AsyncAction Wrapper)
        // ==============================================================================

        // --- ComputePath ---
        BT::NodeBuilder builder_compute = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (current_task_state_ == TaskState::IDLE) {
                    auto target_opt = self.getInput<geometry_msgs::msg::Pose>("target");
                    if (!target_opt) return BT::NodeStatus::FAILURE;
                    
                    this->send_path_goal(target_opt.value());
                    current_task_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status();
            });
        };
        BT::TreeNodeManifest manifest_compute = { BT::NodeType::ACTION, "ComputePath", { BT::InputPort<geometry_msgs::msg::Pose>("target") } };
        factory.registerBuilder(manifest_compute, builder_compute);


        // --- NavigateTo (Sem portas) ---
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
                    RCLCPP_ERROR(this->get_logger(), "Pick/Place: Dados insuficientes.");
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

        BT::NodeBuilder builder_pick_place = [&](const std::string& name, const BT::NodeConfig& config) {
            return std::make_unique<AsyncAction>(name, config, pick_place_lambda);
        };

        BT::PortsList pick_ports = { 
            BT::InputPort<geometry_msgs::msg::Pose>("pose"), 
            BT::InputPort<std::string>("id") 
        };

        BT::TreeNodeManifest manifest_pick = { BT::NodeType::ACTION, "PickObject", pick_ports };
        factory.registerBuilder(manifest_pick, builder_pick_place);

        BT::TreeNodeManifest manifest_place = { BT::NodeType::ACTION, "PlaceObject", pick_ports };
        factory.registerBuilder(manifest_place, builder_pick_place);


        // ==============================================================================
        // 4. UTILS
        // ==============================================================================

        factory.registerSimpleAction("ClearDestination", [&](BT::TreeNode &) {
             std::this_thread::sleep_for(500ms);
             return BT::NodeStatus::SUCCESS;
        }, { BT::InputPort<geometry_msgs::msg::Pose>("dest") });

        factory.registerSimpleAction("Log", [&](BT::TreeNode &self) {
            auto msg = self.getInput<std::string>("msg");
            if(msg) RCLCPP_INFO(this->get_logger(), "%s", msg.value().c_str());
            return BT::NodeStatus::SUCCESS;
        }, { BT::InputPort<std::string>("msg") });

        factory.registerSimpleAction("Wait", [&](BT::TreeNode &self) { 
             auto msec = self.getInput<int>("msec");
             if(msec) std::this_thread::sleep_for(std::chrono::milliseconds(msec.value()));
             return BT::NodeStatus::SUCCESS; 
        }, { BT::InputPort<int>("msec") });

        factory.registerSimpleAction("ResetHardware", [&](BT::TreeNode &){ return BT::NodeStatus::SUCCESS; });
        factory.registerSimpleAction("StopMovement", [&](BT::TreeNode &){ return BT::NodeStatus::SUCCESS; });
        factory.registerSimpleAction("RecalibrateArm", [&](BT::TreeNode &){ return BT::NodeStatus::SUCCESS; });
        factory.registerSimpleAction("AbortMission", [&](BT::TreeNode &){ return BT::NodeStatus::FAILURE; });

        try 
        {
            bt_tree_ = factory.createTreeFromFile(xml_path);
        } 
        catch (const std::exception &e) 
        {
            RCLCPP_ERROR(this->get_logger(), "Erro Fatal BT XML: %s", e.what());
        }
    }


    void bt_loop()
    {
        rclcpp::Rate rate(10); // 10Hz

        while (rclcpp::ok())
        {
            if (!bt_tree_.rootNode()) 
            {
                rate.sleep();
                continue;
            }

            BT::NodeStatus status = bt_tree_.rootNode()->status();

            // Leitura segura da flag
            bool new_obj = false;
            {
                std::lock_guard<std::mutex> lock(bt_mutex_);
                new_obj = has_new_object_;
            }

            // Só executa se já estiver rodando OU se tiver novidade
            if (status == BT::NodeStatus::RUNNING || new_obj)
            {
                BT::NodeStatus result = bt_tree_.tickOnce();

                // Se a missão acabou (sucesso ou falha)
                if (result == BT::NodeStatus::SUCCESS || result == BT::NodeStatus::FAILURE)
                {
                    std::lock_guard<std::mutex> lock(bt_mutex_);
                    has_new_object_ = false;
                    action_busy = false; // Libera o callback
                    
                    if (result == BT::NodeStatus::FAILURE) 
                    {
                        // Se falhou, remove do picked para poder tentar de novo
                        picked.erase(cached_object_.first);
                    }
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
        std::lock_guard<std::mutex> lock(bt_mutex_);

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
    

    // -----------------
    // CLIENT FUNCTIONS 
    // -----------------

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

    bool get_storage_info_sync(const std::string& object_id, const geometry_msgs::msg::Pose& obj_pose, geometry_msgs::msg::Pose& pose_out)
    {
        if (!storage_client_->wait_for_service(std::chrono::seconds(1))) {
            RCLCPP_ERROR(this->get_logger(), "Serviço de storage indisponível.");
            return false;
        }

        auto request = std::make_shared<mobile_manipulation_interfaces::srv::GetStorageInfo::Request>();
        request->object_id = object_id;
        request->pose = obj_pose;

        // Envia request e pega o Future
        auto result_future = storage_client_->async_send_request(request);

        // BLOQUEIA AQUI até ter resposta (Seguro pois estamos na thread separada)
        // Espera até 3 segundos
        if (result_future.wait_for(std::chrono::seconds(3)) == std::future_status::ready)
        {
            auto response = result_future.get();
            if (response->success) 
            {
                pose_out = response->pose;
                // Se quiser usar limites: response->limits...
                return true;
            } else {
                RCLCPP_WARN(this->get_logger(), "Storage service retornou false.");
                return false;
            }
        }
        
        RCLCPP_ERROR(this->get_logger(), "Timeout esperando resposta do Storage Service.");
        return false;
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