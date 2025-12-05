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
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <yaml-cpp/yaml.h>

#include "mobile_manipulation_interfaces/action/pick_object.hpp"
#include "mobile_manipulation_interfaces/action/path.hpp"
#include "mobile_manipulation_interfaces/action/controller.hpp"

#include <manipulation/IsGripperHolding.hpp> 
#include <storage_manager/GetStorageInfo.hpp> 
#include <storage_manager/Organize.hpp> 


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
        // Parâmetros
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("bt_xml_path", "");

        yaml_file = this->get_parameter("yaml_file").as_string();
        std::string bt_xml_path = this->get_parameter("bt_xml_path").as_string();

        // Subscribers
        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/bbox_3d_with_labels", 10,
            std::bind(&ServerNode::detection_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&ServerNode::odom_callback, this, std::placeholders::_1));

        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(this, "pick_object");
        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(this, "path");
        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(this, "controller");

        path_state_ = TaskState::IDLE;
        nav_state_ = TaskState::IDLE;
        manipulation_state_ = TaskState::IDLE;

        setup_behavior_tree(bt_xml_path);

        bt_thread_ = std::thread(&ServerNode::bt_loop, this);
            
        RCLCPP_INFO(this->get_logger(), "ServerNode iniciado (Modo High-Performance).");

        if(!yaml_file.empty()) 
        {
            loadLocationsFromYaml(yaml_file);
        }
    } 

    ~ServerNode()
    {
        if (bt_thread_.joinable()) bt_thread_.join();
    }

private:
    struct ObjectInfo 
    {
        std::string id;
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size; 
    };

    std::shared_ptr<manipulation::IsGripperHolding> gripper_monitor_node_;
    std::shared_ptr<storage_manager::StorageNode> storage_node_;
    std::shared_ptr<storage_manager::OrganizeNode> organize_node_;

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::SharedPtr active_controller_goal_handle_;

    std::string yaml_file;
    std::unordered_set<std::string> authorized_labels;
    std::unordered_set<std::string> picked;
    
    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;
    ObjectInfo cached_object_;
    
    std::string current_target_id_ = "";
    geometry_msgs::msg::Pose current_target_pose_;

    std::thread bt_thread_;
    std::mutex bt_mutex_;
    BT::Tree bt_tree_;
    
    TaskState path_state_;
    TaskState nav_state_;
    TaskState manipulation_state_;
    
    nav_msgs::msg::Path last_calculated_path_; 
    std::mutex path_mutex_;

    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0;
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
            self.setOutput("adjustment_pose", target);
            return BT::NodeStatus::FAILURE; 
        }, 
        { 
            BT::InputPort<geometry_msgs::msg::Pose>("target"), 
            BT::InputPort<double>("max_dist"),
            BT::InputPort<double>("min_dist"),
            BT::OutputPort<geometry_msgs::msg::Pose>("adjustment_pose")
        });

        factory.registerSimpleAction("DetectObject", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);

            if (!current_target_id_.empty())
            {
                
                self.setOutput("output_pose", current_target_pose_);
                self.setOutput("output_id", current_target_id_);
                self.setOutput("object_size", cached_object_.size);
                
                return BT::NodeStatus::SUCCESS;
            }

            if (!has_new_object_) 
            {
                return BT::NodeStatus::RUNNING;
            }

            current_target_id_ = cached_object_.id;
            current_target_pose_ = cached_object_.pose;
            
            self.setOutput("output_pose", current_target_pose_);
            self.setOutput("output_id", current_target_id_);
            self.setOutput("object_size", cached_object_.size); 
            
            picked.insert(current_target_id_);
            has_new_object_ = false;
            
            RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' (Size: %.2fx%.2fx%.2f) processado.", 
                current_target_id_.c_str(), cached_object_.size.x, cached_object_.size.y, cached_object_.size.z);

            return BT::NodeStatus::SUCCESS;
        }, 
        { 
            BT::OutputPort<geometry_msgs::msg::Pose>("output_pose"), 
            BT::OutputPort<std::string>("output_id"),
            BT::OutputPort<geometry_msgs::msg::Vector3>("object_size") 
        });


        factory.registerSimpleAction("ClearTarget", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' finalizado.", current_target_id_.c_str());
            
  
            current_target_id_ = "";
            return BT::NodeStatus::SUCCESS;
        });

        factory.registerSimpleAction("GetStorageInfo", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("object_id");
            if (!id_opt) 
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: O Input 'object_id' não foi fornecido.");
                return BT::NodeStatus::FAILURE;
            }

            std::string full_id = id_opt.value();
            std::string label = full_id;
            size_t pos = full_id.find('_');

            if (pos != std::string::npos) 
            {
                label = full_id.substr(0, pos);
            }

            geometry_msgs::msg::Pose current_obj_pose;

            {
                std::lock_guard<std::mutex> lock(bt_mutex_);
                current_obj_pose = current_target_pose_; 
            }

            auto result = storage_node_->getBestStorage(label, current_obj_pose);


            if (result.success) 
            {
                self.setOutput("storage_pose", result.pose);
                self.setOutput("storage_limits", result.limits);
                self.setOutput("storage_id", result.storage_name);
                self.setOutput("indexes", result.indexes);
                self.setOutput("storage_size", result.size); 

                RCLCPP_INFO(this->get_logger(), "Storage: %s (Ocupação: %d | Índices: [%d, %d, %d])", 
                    result.storage_name.c_str(), 
                    result.current_count,
                    result.indexes[0], result.indexes[1], result.indexes[2]);

                RCLCPP_INFO(this->get_logger(), "Storage: %.2f, %.2f, %.2f", 
                result.size.x, result.size.y, result.size.z);

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

        factory.registerSimpleAction("ComputePoseToOrganize", [&](BT::TreeNode &self)
        {
            auto storagePose = self.getInput<geometry_msgs::msg::Pose>("storage_pose");
            auto storageSize = self.getInput<geometry_msgs::msg::Vector3>("storage_size");
            auto objectSize = self.getInput<geometry_msgs::msg::Vector3>("object_size");
            auto indexes = self.getInput<std::vector<int>>("indexes");
            auto objectPadding = self.getInput<float>("object_padding");
            auto zLiftOffset = self.getInput<float>("z_lift_offset");

 
            if (!storagePose )
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: StoragePOse.");
                return BT::NodeStatus::FAILURE;
            }

            if (!storageSize)
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: storageSize.");
                return BT::NodeStatus::FAILURE;
            }

            if (!objectSize )
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: objectSize.");
                return BT::NodeStatus::FAILURE;
            }

            if (!indexes)
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: indexes.");
                return BT::NodeStatus::FAILURE;
            }

            if (!objectPadding)
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: objectPadding.");
                return BT::NodeStatus::FAILURE;
            }

            if (!zLiftOffset)
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO:zLiftOffset.");
                return BT::NodeStatus::FAILURE;
            }
            
            geometry_msgs::msg::Pose storage_pose = storagePose.value();
            geometry_msgs::msg::Vector3 storage_size = storageSize.value();
            geometry_msgs::msg::Vector3 object_size = objectSize.value();

            std::vector<int> idx_vec = indexes.value();
            float object_padding = objectPadding.value();
            float z_lift_offset = zLiftOffset.value();

            if (idx_vec.size() != 3) 
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: Indexes vector size must be 3 (x, y, z).");
                return BT::NodeStatus::FAILURE;
            }
            
            int idx_x = idx_vec[0];
            int idx_y = idx_vec[1];
            int idx_z = idx_vec[2];
            
            std::pair<geometry_msgs::msg::Pose, std::vector<int>> result = organize_node_->placeObjectInBox(
                storage_pose, 
                storage_size, 
                object_size, 
                object_padding, 
                z_lift_offset, 
                idx_x, 
                idx_y, 
                idx_z 
            );


          
            self.setOutput("output_final_pose", std::get<0>(result));
            self.setOutput("new_indexes", std::get<1>(result));
            
            RCLCPP_INFO(this->get_logger(), 
                "Pose Calculada | XYZ: [%.3f, %.3f, %.3f] | Quat: [%.3f, %.3f, %.3f, %.3f]",
                std::get<0>(result).position.x, 
                std::get<0>(result).position.y, 
                std::get<0>(result).position.z,
                std::get<0>(result).orientation.x, 
                std::get<0>(result).orientation.y, 
                std::get<0>(result).orientation.z, 
                std::get<0>(result).orientation.w);

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

        factory.registerSimpleAction("ComputePoseToStore", [&](BT::TreeNode &self)
        {
            auto storagePose = self.getInput<geometry_msgs::msg::Pose>("storage_pose");
            auto storageSize = self.getInput<geometry_msgs::msg::Vector3>("storage_size");
            auto zLiftOffset = self.getInput<float>("z_lift_offset");

 
            if (!storagePose )
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: storagePose.");
                return BT::NodeStatus::FAILURE;
            }

            if (!storageSize)
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: storageSize.");
                return BT::NodeStatus::FAILURE;
            }

            if (!zLiftOffset)
            {
                RCLCPP_ERROR(this->get_logger(), "ERRO: zLiftOffset.");
                return BT::NodeStatus::FAILURE;
            }
            
            geometry_msgs::msg::Pose storage_pose = storagePose.value();
            geometry_msgs::msg::Vector3 storage_size = storageSize.value();
            float z_lift_offset = zLiftOffset.value();


            geometry_msgs::msg::Pose output_final_pose = storage_pose;
            std::cout << output_final_pose.position.z << std::endl;
            output_final_pose.position.z = output_final_pose.position.z + storage_size.z + z_lift_offset;
            std::cout << output_final_pose.position.z << std::endl;
          
            self.setOutput("output_final_pose", output_final_pose);
            
            RCLCPP_INFO(this->get_logger(), 
                "Pose Calculada | XYZ: [%.3f, %.3f, %.3f] | Quat: [%.3f, %.3f, %.3f, %.3f]",
                output_final_pose.position.x, 
                output_final_pose.position.y, 
                output_final_pose.position.z,
                output_final_pose.orientation.x, 
                output_final_pose.orientation.y, 
                output_final_pose.orientation.z, 
                output_final_pose.orientation.w);

            return BT::NodeStatus::SUCCESS;
        
        }, 
        { 
            BT::InputPort<geometry_msgs::msg::Pose>("storage_pose"),
            BT::InputPort<geometry_msgs::msg::Vector3>("storage_size"),
            BT::InputPort<float>("z_lift_offset"),
            BT::OutputPort<geometry_msgs::msg::Pose>("output_final_pose")
        });


        factory.registerSimpleAction("IncrementOrganizedStorageIndexes", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("storage_id");
            auto newIndexes = self.getInput<std::vector<int>>("new_indexes");
            if (!id_opt || !newIndexes) 
            {
                return BT::NodeStatus::FAILURE;
            }

            std::string storage_name = id_opt.value();
            std::vector<int> new_indexes = newIndexes.value();
            
            storage_node_->addNewIndexes(storage_name, new_indexes);
            
            RCLCPP_WARN(this->get_logger(), "Adicionando novos índices ocupados no armazém '%s'.", storage_name.c_str());

            return BT::NodeStatus::SUCCESS;
        }, 
        { 
            BT::InputPort<std::string>("storage_id"),
            BT::InputPort<std::vector<int>>("new_indexes") 
        });

        factory.registerSimpleAction("DecrementStorageCount", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("storage_id");
            if (!id_opt) 
            {
                return BT::NodeStatus::FAILURE;
            }

            std::string storage_name = id_opt.value();
            
            storage_node_->incrementStorageCount(storage_name, -1);
            
            RCLCPP_WARN(this->get_logger(), "ROLLBACK: Liberando vaga no storage '%s' devido a falha.", storage_name.c_str());

            return BT::NodeStatus::SUCCESS;
        }, 
        { BT::InputPort<std::string>("storage_id") });


        factory.registerSimpleCondition("IsGripperHoldingObject", 
            [this](BT::TreeNode& self) -> BT::NodeStatus 
            {
                bool is_holding = this->gripper_monitor_node_->checkIsHolding();
                
                if (is_holding) 
                {
                    return BT::NodeStatus::SUCCESS; 
                }
                else 
                {
                    return BT::NodeStatus::FAILURE; 
                }
            }
        );


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
        factory.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "ComputePath", { BT::InputPort<geometry_msgs::msg::Pose>("target"), BT::InputPort<std::string>("planner") }, {} }, builder_compute);


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

                    if (!pose || !id) 
                    {
                        return BT::NodeStatus::FAILURE;
                    }
                    
                    this->send_goal(id.value(), pose.value(), true);
                    manipulation_state_ = TaskState::RUNNING;
                    
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "PickObject", { BT::InputPort<geometry_msgs::msg::Pose>("pose"), BT::InputPort<std::string>("id") }, {} }, builder_pick);

        
     
        BT::NodeBuilder builder_place = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE) 
                {
                    auto pose = self.getInput<geometry_msgs::msg::Pose>("pose");
                    if (!pose) 
                    {
                        return BT::NodeStatus::FAILURE;
                    }
                    
                    std::string id_dummy = cached_object_.id; 
                    
                    this->send_goal(id_dummy, pose.value(), false); 
                    manipulation_state_ = TaskState::RUNNING;
                    
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "PlaceObject", { BT::InputPort<geometry_msgs::msg::Pose>("pose"), BT::InputPort<std::vector<double>>("limits") }, {} }, builder_place);

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

            if (status == BT::NodeStatus::RUNNING || new_obj || !current_target_id_.empty())
            {
                BT::NodeStatus result = bt_tree_.tickOnce();

                if (result == BT::NodeStatus::SUCCESS || result == BT::NodeStatus::FAILURE)
                {
                    std::lock_guard<std::mutex> lock(bt_mutex_);
                    has_new_object_ = false;
                    
                    if (result == BT::NodeStatus::FAILURE) 
                    {
                         picked.erase(cached_object_.id);
                         current_target_id_ = ""; 
                    }
                    
                    path_state_ = TaskState::IDLE;
                    nav_state_ = TaskState::IDLE;
                    manipulation_state_ = TaskState::IDLE;
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


        for (const auto &det : msg->detections)
        {
            if (det.results.empty()) 
            {
                continue;
            }

            std::string raw_id = det.results[0].hypothesis.class_id;
            
            std::string label = raw_id;
            size_t pos = raw_id.find('_'); 
            if (pos != std::string::npos) 
            {
                label = raw_id.substr(0, pos);
            }

            if (authorized_labels.find(label) == authorized_labels.end()) 
            {
                continue;
            }

            if (picked.find(raw_id) != picked.end()) 
            {
                continue;
            }

            
            geometry_msgs::msg::Pose pose;
            pose.position = det.bbox.center.position;
            pose.orientation = det.bbox.center.orientation;
            
            cached_object_.id = raw_id;
            cached_object_.pose = pose;
            cached_object_.size = det.bbox.size;
            
            has_new_object_ = true; 
            
            RCLCPP_INFO(this->get_logger(), "Nova detecção salva: '%s' (Size: %.2fx%.2fx%.2f)", 
                raw_id.c_str(), cached_object_.size.x, cached_object_.size.y, cached_object_.size.z);

            break;
        }
    }



    // --- Path Callbacks ---


    void send_path_goal(const geometry_msgs::msg::Pose & target_pose)
    {
        if (!this->path_client->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server 'path' not available");
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

    void cancel_controller_goal()
    {
        if (this->active_controller_goal_handle_)
        {
            RCLCPP_WARN(this->get_logger(), "Solicitando PARADA IMEDIATA (Cancelando Action Controller)...");
            this->controller_client->async_cancel_goal(this->active_controller_goal_handle_);
        }
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
            {
                std::lock_guard<std::mutex> lock(path_mutex_);
                last_calculated_path_.poses.clear(); 
            }
            
            RCLCPP_INFO(this->get_logger(), "Planner recalculando: cancelando controller atual...");
            cancel_controller_goal();
        }
        else 
        {
            if (!feedback->path.poses.empty())
            {
                RCLCPP_INFO(this->get_logger(), "Novo caminho recebido e armazenado. Aguardando BT executar.");
            }
        }
    }

    void path_goal_response_callback(const std::shared_ptr<rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>> & goal_handle)
    {
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Goal PATH rejeitado");
            path_state_ = TaskState::FAILURE;
        } else {
            RCLCPP_INFO(this->get_logger(), "Goal PATH aceito, calculando...");
        }
    }

    void path_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) 
        {
            
            if (result.result->success) 
            {
                path_state_ = TaskState::SUCCESS;
                RCLCPP_INFO(this->get_logger(), "PATH RESULT SUCESS");
            }
            else 
            {
                path_state_ = TaskState::FAILURE;
                RCLCPP_INFO(this->get_logger(), "PATH RESULT FAILURE");
            }
        }
    }

    // --- Controller Callbacks ---

    bool send_controller_goal(const nav_msgs::msg::Path &target_path)
    {
        if (!this->controller_client->wait_for_action_server(std::chrono::seconds(2))) {
            RCLCPP_ERROR(this->get_logger(), "Action server 'controller' not available");
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Mandando para o controller");

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
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Goal CONTROLLER rejeitado");
            nav_state_ = TaskState::FAILURE;
        } else {
            this->active_controller_goal_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Goal CONTROLLER aceito, executando...");
        }
    }

    void controller_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::WrappedResult & result)
    {
        if (this->active_controller_goal_handle_ && result.goal_id != this->active_controller_goal_handle_->get_goal_id()) {
            RCLCPP_WARN(this->get_logger(), "Ignorando resultado de controller antigo.");
            return; 
        }

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) 
        {
            {
                std::lock_guard<std::mutex> lock(path_mutex_);
                last_calculated_path_.poses.clear();
            }
            
            nav_state_ = TaskState::SUCCESS;
            RCLCPP_INFO(this->get_logger(), "Navegação concluída!");
        } 
        else if (result.code == rclcpp_action::ResultCode::CANCELED) 
        {
            RCLCPP_INFO(this->get_logger(), "Navegação cancelada.");
            nav_state_ = TaskState::IDLE;
        }
        else 
        {
            RCLCPP_ERROR(this->get_logger(), "Goal CONTROLLER falhou/abortou");
            nav_state_ = TaskState::FAILURE;
        }
        
        if (this->active_controller_goal_handle_ && result.goal_id == this->active_controller_goal_handle_->get_goal_id()) {
            this->active_controller_goal_handle_.reset();
        }
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
            manipulation_state_ = TaskState::FAILURE;
            RCLCPP_ERROR(this->get_logger(), "PICK FAILED or ABORTED");
        }
    }

};

bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
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
  

  std::shared_ptr<storage_manager::OrganizeNode> organize_node = nullptr;
  std::shared_ptr<storage_manager::StorageNode> storage_node   = nullptr;
  std::shared_ptr<manipulation::IsGripperHolding> gripper_node = nullptr;

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

  auto server_node = std::make_shared<ServerNode>(gripper_node, storage_node, organize_node);
  executor.add_node(server_node);

  executor.spin();

  rclcpp::shutdown();
  return 0;
}