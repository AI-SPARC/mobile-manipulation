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
#include <optional>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/xml_parsing.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
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
#include <sqlite3.h> 
#include "geometry_msgs/msg/pose_array.hpp"

#include "mobile_manipulation_interfaces/action/pick_object.hpp"
#include "mobile_manipulation_interfaces/action/path.hpp"
#include "mobile_manipulation_interfaces/action/controller.hpp"

#include <manipulation/IsGripperHolding.hpp>
#include <manipulation/ProjectedReachabilityAnalysis.hpp>
#include <manipulation/IKValidator.hpp>

#include <vision/GenerateScanPoses.hpp>
#include <vision/ObjectMapping.hpp>

#include <storage_manager/GetStorageInfo.hpp>
#include <storage_manager/Organize.hpp>

#include <navigation/SharedObstacleGraph.hpp>

#include <drl_to_pick_cpp/BridgeToInference.hpp>

#include <llms/DatabaseHandler.hpp>
#include <llms/WorldStateNode.hpp>

static DatabaseHandler* g_db_handler = nullptr;

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

enum class GraspPhase
{
    IDLE,           
    GRASPNET_SCAN,  
    SEND_GOAL,
    WAITING,
    SUCCESS,
    FAILURE
};


class ForEach : public BT::DecoratorNode
{
public:
    ForEach(const std::string& name, const BT::NodeConfig& config)
        : BT::DecoratorNode(name, config), current_index_(0) {}

    static BT::PortsList providedPorts()
    {
        return {
            BT::InputPort<std::string>("items"),
            BT::InputPort<std::string>("dests"),
            BT::OutputPort<std::string>("item"),
            BT::OutputPort<std::string>("pose"),
            BT::OutputPort<std::string>("size"),
            BT::OutputPort<std::string>("dest"),
            BT::OutputPort<std::string>("dest_pose"),
            BT::OutputPort<std::string>("dest_size")  
        };
    }

    BT::NodeStatus tick() override
    {
        if (current_index_ == 0 && items_.empty())
        {
            auto items_str = getInput<std::string>("items");
            
            if (!items_str) 
            {
                std::cerr << "[ForEach] Erro: 'items' não fornecido!" << std::endl;
                return BT::NodeStatus::FAILURE;
            }
            
            items_ = split(items_str.value(), '|');
            
            auto dests_str = getInput<std::string>("dests");
            if (dests_str) 
            {
                dests_ = split(dests_str.value(), '|');
            }
        }
        
        if (current_index_ >= items_.size())
        {
            reset();
            return BT::NodeStatus::SUCCESS;
        }
        
        std::string current_item = items_[current_index_];
        setOutput("item", current_item);
        
        if (g_db_handler) 
        {
            auto props = g_db_handler->get_object_data(current_item);
            if (props) 
            {
                setOutput("pose", props->pose_str);
                setOutput("size", props->size_str);
            } 
            else 
            {
                std::cerr << "[ForEach] DB FALHOU - item '" << current_item << "' não encontrado!" << std::endl;
                setOutput("pose", "");
                setOutput("size", "");
            }
        }
        
        if (!dests_.empty())
        {
            size_t dest_idx;
            if (dests_.size() == 1) 
            {
                dest_idx = 0;  
            } 
            else 
            {
                dest_idx = current_index_;  
            }
            
            if (dest_idx < dests_.size())
            {
                std::string current_dest = dests_[dest_idx];
                setOutput("dest", current_dest);
                
                if (g_db_handler) 
                {
                    auto props = g_db_handler->get_object_data(current_dest);
                    if (props) 
                    {
                        setOutput("dest_pose", props->pose_str);
                        setOutput("dest_size", props->size_str);
                    } 
                    else 
                    {
                        setOutput("dest_pose", "");
                        setOutput("dest_size", "");
                    }
                }
            }
        }
        
        BT::NodeStatus child_status = child_node_->executeTick();
        
        if (child_status == BT::NodeStatus::SUCCESS)
        {
            current_index_++;
            haltChild();
            return BT::NodeStatus::RUNNING;  
        }
        else if (child_status == BT::NodeStatus::FAILURE)
        {
            std::cerr << "[ForEach] Filho falhou no item: " << current_item << std::endl;
            reset();
            return BT::NodeStatus::FAILURE;
        }
        
        return BT::NodeStatus::RUNNING;
    }

    void halt() override
    {
        reset();
        BT::DecoratorNode::halt();
    }

private:
    size_t current_index_;
    std::vector<std::string> items_, dests_;
    
    void reset()
    {
        current_index_ = 0;
        items_.clear();
        dests_.clear();
    }
    
    std::vector<std::string> split(const std::string& str, char delim)
    {
        std::vector<std::string> result;
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, delim)) 
        {
            if (!item.empty()) 
            {
                result.push_back(item);
            }
        }
        return result;
    }
};

// NÓ ORIGINAL: ParallelAny
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

// NÓ ORIGINAL: AsyncAction
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


class ServerNode : public rclcpp::Node
{
public:
    ServerNode(
        std::shared_ptr<manipulation::IsGripperHolding> gripper_node,
        std::shared_ptr<manipulation::ProjectedReachabilityAnalysis> reachability_node,
        std::shared_ptr<manipulation::IKValidator> ik_validator_node,
        std::shared_ptr<navigation::SharedObstacleGraph> obstacle_graph_node,
        std::shared_ptr<storage_manager::StorageNode> storage_node,
        std::shared_ptr<storage_manager::OrganizeNode> organize_node,
        std::shared_ptr<drl_to_pick_cpp::BridgeToInference> bridge_to_inference_node,
        std::shared_ptr<vision::GenerateScanPoses> scan_object_node,
        std::shared_ptr<vision::ObjectMapping> object_mapping_node,
        std::shared_ptr<llms::WorldStateNode> world_state_node
    )
    : Node("server_node"),
    gripper_monitor_node_(gripper_node),
    reachability_node_(reachability_node),
    ik_validator_node_(ik_validator_node),
    obstacle_graph_node_(obstacle_graph_node),
    storage_node_(storage_node),
    organize_node_(organize_node),
    bridge_to_inference_node_(bridge_to_inference_node),
    scan_object_node_(scan_object_node),
    object_mapping_node_(object_mapping_node),
    world_state_node_(world_state_node)
    {
        this->declare_parameter<std::string>("yaml_file", "");
        this->declare_parameter<std::string>("bt_xml_path", "");
        this->declare_parameter<std::string>("database_path", "/home/momesso/pibic/src/mobile_manipulation_packages/llms/db/robot_world_data.db");
        this->declare_parameter<bool>("use_llm", false); 
        this->declare_parameter<bool>("use_graspnet", true);
        this->declare_parameter<bool>("use_pca", false);
        this->declare_parameter<int>("max_graspnet_attempts", 3);

        yaml_file = this->get_parameter("yaml_file").as_string();
        std::string bt_xml_path = this->get_parameter("bt_xml_path").as_string();
        db_path_ = this->get_parameter("database_path").as_string();
        use_llm = this->get_parameter("use_llm").as_bool();
        use_graspnet = this->get_parameter("use_graspnet").as_bool();
        use_pca = this->get_parameter("use_pca").as_bool();
        this->grasp_context_.graspnet_maximum_attempts = this->get_parameter("max_graspnet_attempts").as_int();

        db_handler_ = std::make_unique<DatabaseHandler>(db_path_);
        g_db_handler = db_handler_.get();  

        RCLCPP_INFO(this->get_logger(), "DatabaseHandler conectado: %s", db_path_.c_str());

      
        if (sqlite3_open_v2(db_path_.c_str(), &db_read_conn_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) 
        {
            RCLCPP_ERROR(this->get_logger(), "Falha ao abrir DB para leitura direta: %s", sqlite3_errmsg(db_read_conn_));
        }

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&ServerNode::odom_callback, this, std::placeholders::_1));

        
        if (use_llm)
        {
            RCLCPP_INFO(this->get_logger(), "Modo LLM ATIVADO. Aguardando XML em /behavior_tree_xml");
            bt_xml_sub_ = this->create_subscription<std_msgs::msg::String>(
                "/behavior_tree_xml", 10,
                std::bind(&ServerNode::on_bt_xml_received, this, std::placeholders::_1));
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Modo AUTO (Random DB) ATIVADO.");
        }

        publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("object_pose", 10);
        
        
        client_ptr_ = rclcpp_action::create_client<mobile_manipulation_interfaces::action::PickObject>(this, "pick_object");
        path_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Path>(this, "path");
        controller_client = rclcpp_action::create_client<mobile_manipulation_interfaces::action::Controller>(this, "controller");
        

        pose_array_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/debug_grasps", 10);

        
        path_state_ = TaskState::IDLE;
        nav_state_ = TaskState::IDLE;
        manipulation_state_ = TaskState::IDLE;
        current_pick_phase_ = GraspPhase::IDLE;

       
        setup_behavior_tree(bt_xml_path);

        bt_thread_ = std::thread(&ServerNode::bt_loop, this);

        RCLCPP_INFO(this->get_logger(), "ServerNode iniciado.");

        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&ServerNode::publish_pose, this));

        if(!yaml_file.empty())
        {
            loadLocationsFromYaml(yaml_file);
        }
    }

    ~ServerNode()
    {
        if (bt_thread_.joinable()) bt_thread_.join();
        if (db_read_conn_) sqlite3_close(db_read_conn_);
    }

private:
    struct ObjectInfo
    {
        std::string id;
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size;
    };

    struct GraspContext
    {
        std::vector<geometry_msgs::msg::Pose> grasp_poses = {};
        int graspnet_attempts = 0;
        int graspnet_maximum_attempts = 3; 
    };

  
    std::shared_ptr<manipulation::IsGripperHolding> gripper_monitor_node_;
    std::shared_ptr<manipulation::ProjectedReachabilityAnalysis> reachability_node_;
    std::shared_ptr<manipulation::IKValidator> ik_validator_node_;
    std::shared_ptr<navigation::SharedObstacleGraph> obstacle_graph_node_;
    std::shared_ptr<storage_manager::StorageNode> storage_node_;
    std::shared_ptr<storage_manager::OrganizeNode> organize_node_;
    std::shared_ptr<drl_to_pick_cpp::BridgeToInference> bridge_to_inference_node_;
    std::shared_ptr<vision::GenerateScanPoses> scan_object_node_;
    std::shared_ptr<vision::ObjectMapping> object_mapping_node_;
    std::unique_ptr<BT::Groot2Publisher> groot_publisher_;
    std::shared_ptr<llms::WorldStateNode> world_state_node_;

    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr bt_xml_sub_; 

    rclcpp_action::Client<mobile_manipulation_interfaces::action::PickObject>::SharedPtr client_ptr_;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Path>::SharedPtr path_client;
    rclcpp_action::Client<mobile_manipulation_interfaces::action::Controller>::SharedPtr controller_client;

    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Controller>::SharedPtr active_controller_goal_handle_;
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::SharedPtr active_path_goal_handle_;
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>::SharedPtr active_arm_handle_;

    
    std::string yaml_file;
    std::string db_path_;
    std::unordered_set<std::string> authorized_labels;
    std::unordered_set<std::string> picked;

    std::pair<std::string, geometry_msgs::msg::Pose> pick_pose;
    ObjectInfo cached_object_;
    GraspContext grasp_context_;

    std::unique_ptr<DatabaseHandler> db_handler_;
    sqlite3* db_read_conn_ = nullptr;

    std::string current_target_id_ = "";
    geometry_msgs::msg::Pose current_target_pose_;

    rclcpp::TimerBase::SharedPtr timer_;

    
    std::thread bt_thread_;
    std::mutex bt_mutex_;
    std::mutex odom_mutex;
    
    
    BT::BehaviorTreeFactory factory_;
    std::unique_ptr<BT::Tree> bt_tree_;

   
    std::mutex xml_mutex_;
    std::string pending_xml_;
    std::atomic<bool> has_new_tree_{false};
    std::atomic<int> tree_counter_{0};

    
    TaskState path_state_;
    TaskState nav_state_;
    TaskState manipulation_state_;
    GraspPhase current_pick_phase_;

    std::mutex state_mutex_;
    std::mutex path_mutex_;

    nav_msgs::msg::Path last_calculated_path_;
    nav_msgs::msg::Path last_no_filter_calculated_path_;

    float pose_x = 0.0, pose_y = 0.0, pose_z = 0.0;
    bool has_new_object_ = false;
    bool use_llm = false;
    bool use_graspnet = false;
    bool use_pca = true;


    void on_bt_xml_received(const std_msgs::msg::String::SharedPtr msg)
    {
        if (msg->data.empty())
        {
            return;
        }

       
        if (bt_tree_ && bt_tree_->rootNode() && bt_tree_->rootNode()->status() == BT::NodeStatus::RUNNING)
        {
            RCLCPP_WARN(this->get_logger(), "Árvore em execução! Ignorando novo XML por segurança.");
            // return; 
        }

        RCLCPP_INFO(this->get_logger(), "XML recebido via tópico.");
        
        std::string expanded_xml = expand_xml_with_db(msg->data);

        {
            std::lock_guard<std::mutex> lock(xml_mutex_);
            pending_xml_ = expanded_xml;
            has_new_tree_ = true;
        }
    }

    std::string expand_xml_with_db(const std::string& xml)
    {
        std::string result = xml;
        result = expand_pick_subtrees(result);
        result = expand_place_subtrees(result);
        result = expand_goto_subtrees(result);
        return result;
    }

    std::string expand_pick_subtrees(const std::string& xml)
    {
        std::string result = xml;
        std::string search = "<SubTree ID=\"Pick\" target_id=\"";
        size_t pos = 0;
        
        while ((pos = result.find(search, pos)) != std::string::npos)
        {
            size_t id_start = pos + search.length();
            size_t id_end = result.find("\"", id_start);
            
            if (id_end == std::string::npos) break;
            
            std::string target_id = result.substr(id_start, id_end - id_start);
            
            if (!target_id.empty() && target_id[0] == '{') 
            {
                pos = id_end; 
                continue; 
            }
            
            size_t line_end = result.find("/>", id_end);
            std::string line = result.substr(pos, line_end - pos);
            
            if (line.find("target_pose=") != std::string::npos) 
            {
                pos = line_end; 
                continue; 
            }
            
            std::string pose_str = "", size_str = "";
            
            if (db_handler_) 
            {
                auto props = db_handler_->get_object_data(target_id);
                if (props) 
                {
                    pose_str = props->pose_str; 
                    size_str = props->size_str;
                }
            }
            
            std::string insert = " target_pose=\"" + pose_str + "\" target_size=\"" + size_str + "\"";
            result.insert(line_end, insert);
            pos = line_end + insert.length();
        }
        return result;
    }

    std::string expand_place_subtrees(const std::string& xml)
    {
        std::string result = xml;
        std::string search = "<SubTree ID=\"Place\" storage_id=\"";
        size_t pos = 0;
        
        while ((pos = result.find(search, pos)) != std::string::npos)
        {
            size_t id_start = pos + search.length();
            size_t id_end = result.find("\"", id_start);
            if (id_end == std::string::npos) break;
            
            std::string storage_id = result.substr(id_start, id_end - id_start);
            
            if (!storage_id.empty() && storage_id[0] == '{') 
            {
                pos = id_end; 
                continue; 
            }
            
            size_t line_end = result.find("/>", id_end);
            std::string line = result.substr(pos, line_end - pos);
            
            if (line.find("final_placement_pose=") != std::string::npos) 
            {
                pos = line_end; 
                continue; 
            }
            
            std::string pose_str = "";
            
            if (db_handler_) 
            {
                auto props = db_handler_->get_object_data(storage_id);
                if (props) 
                {
                    pose_str = props->pose_str;
                }
            }
            
            std::string insert = " final_placement_pose=\"" + pose_str + "\"";
            result.insert(line_end, insert);
            pos = line_end + insert.length();
        }
        
        
        search = "<SubTree ID=\"Place\" target=\"";
        pos = 0;
        
        while ((pos = result.find(search, pos)) != std::string::npos)
        {
            size_t target_start = pos + search.length();
            size_t target_end = result.find("\"", target_start);
            
            if (target_end == std::string::npos) break;
            
            std::string target_coords = result.substr(target_start, target_end - target_start);
            
            if (!target_coords.empty() && target_coords[0] == '{') 
            {
                pos = target_end; 
                continue; 
            }
            
            size_t line_end = result.find("/>", target_end);
            std::string line = result.substr(pos, line_end - pos);
            
            if (line.find("final_placement_pose=") != std::string::npos) 
            {
                pos = line_end; 
                continue; 
            }
            
            std::string insert = " storage_id=\"direct\" final_placement_pose=\"" + target_coords + "\"";
            result.insert(line_end, insert);
            pos = line_end + insert.length();
        }
        return result;
    }

    std::string expand_goto_subtrees(const std::string& xml)
    {
        std::string result = xml;
        std::string search = "<SubTree ID=\"GoToLocation\" target_id=\"";
        size_t pos = 0;
        
        while ((pos = result.find(search, pos)) != std::string::npos)
        {
            size_t id_start = pos + search.length();
            size_t id_end = result.find("\"", id_start);
            if (id_end == std::string::npos) break;
            
            std::string target_id = result.substr(id_start, id_end - id_start);
            
            if (!target_id.empty() && target_id[0] == '{') 
            {
                pos = id_end; 
                continue; 
            }
            
            size_t line_end = result.find("/>", id_end);
            std::string line = result.substr(pos, line_end - pos);
            
            if (line.find("nav_target=") != std::string::npos) 
            {
                pos = line_end; 
                continue; 
            }
            
            std::string pose_str = "";
            if (db_handler_) 
            {
                auto props = db_handler_->get_object_data(target_id);
                if (props) 
                {
                    pose_str = props->pose_str;
                }
            }
            
            std::string insert = " nav_target=\"" + pose_str + "\"";
            result.insert(line_end, insert);
            pos = line_end + insert.length();
        }
        
        search = "<SubTree ID=\"GoToLocation\" target=\"";
        pos = 0;
        
        while ((pos = result.find(search, pos)) != std::string::npos)
        {
            size_t target_start = pos + search.length();
            size_t target_end = result.find("\"", target_start);
            
            if (target_end == std::string::npos) break;
            
            std::string target_coords = result.substr(target_start, target_end - target_start);
            
            if (!target_coords.empty() && target_coords[0] == '{') 
            {
                pos = target_end; 
                continue; 
            }
            
            size_t line_end = result.find("/>", target_end);
            std::string line = result.substr(pos, line_end - pos);
            
            if (line.find("nav_target=") != std::string::npos) 
            {
                pos = line_end; 
                continue; 
            }
            
            std::string insert = " nav_target=\"" + target_coords + "\"";
            result.insert(line_end, insert);
            pos = line_end + insert.length();
        }
        return result;
    }

    // --- UTILS ---

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

    BT::NodeStatus check_pick_phase_status(GraspPhase &state)
    {
        if (state == GraspPhase::SUCCESS)
        {
            state = GraspPhase::IDLE; 
            return BT::NodeStatus::SUCCESS;
        }
        else if (state == GraspPhase::FAILURE)
        {
            state = GraspPhase::IDLE; 
            return BT::NodeStatus::FAILURE;
        }
        return BT::NodeStatus::RUNNING; 
    }

    // --- SETUP BT ---

    void setup_behavior_tree(const std::string &xml_path)
    {
        
        factory_.registerNodeType<ParallelAny>("ParallelAny");
        
        
        factory_.registerNodeType<ForEach>("ForEach");

        
        factory_.registerSimpleCondition("IsReachable", [&](BT::TreeNode &self)
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

                float dx_curr = px - this->pose_x;
                float dy_curr = py - this->pose_y;
                float dist_sq = std::sqrt((dx_curr * dx_curr) + (dy_curr * dy_curr));

                const float threshold_sq = 0.2f; 

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

        // --- Condition: IsStillReachable ---
        factory_.registerSimpleCondition("IsStillReachable", [&](BT::TreeNode &self)
        {
            bool reachable = this->ik_validator_node_->is_still_reachable(this->obstacle_graph_node_);

            if(reachable == true)
            {
                return BT::NodeStatus::SUCCESS;
            }
            else
            {
                return BT::NodeStatus::FAILURE;
            }
        });

        // --- Action: DetectObject ---
        BT::NodeBuilder builder_detect = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                std::lock_guard<std::mutex> lock(bt_mutex_); 

                // Se já temos um alvo selecionado anteriormente
                if (!current_target_id_.empty())
                {
                    self.setOutput("output_pose", current_target_pose_);
                    self.setOutput("output_id", current_target_id_);
                    self.setOutput("output_size", cached_object_.size);
                    return BT::NodeStatus::SUCCESS;
                }

                // Se não há objeto novo detectado/sorteado, aguarda (RUNNING)
                if (!has_new_object_)
                {
                    return BT::NodeStatus::RUNNING;
                }

                // Objeto encontrado! Trava o alvo.
                current_target_id_ = cached_object_.id;
                current_target_pose_ = cached_object_.pose;

                self.setOutput("output_pose", current_target_pose_);
                self.setOutput("output_id", current_target_id_);
                self.setOutput("output_size", cached_object_.size);

                picked.insert(current_target_id_);
                has_new_object_ = false;

                RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' travado.", current_target_id_.c_str());
                return BT::NodeStatus::SUCCESS;
            });
        };

        // REGISTRO OBRIGATÓRIO DAS PORTAS AQUI:
        factory_.registerBuilder(BT::TreeNodeManifest{
            BT::NodeType::ACTION, 
            "DetectObject", 
            { 
                BT::OutputPort<geometry_msgs::msg::Pose>("output_pose"), 
                BT::OutputPort<std::string>("output_id"),
                BT::OutputPort<geometry_msgs::msg::Vector3>("output_size")
            }, 
            {} 
        }, builder_detect);

        // --- Action: ClearTarget ---
        factory_.registerSimpleAction("ClearTarget", [&](BT::TreeNode &self)
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            RCLCPP_INFO(this->get_logger(), "BT: Alvo '%s' liberado.", current_target_id_.c_str());
            current_target_id_ = ""; 
            return BT::NodeStatus::SUCCESS;
        });

        // --- Condition: IsPathClear ---
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

        // --- Action: GetStorageInfo ---
        factory_.registerSimpleAction("GetStorageInfo", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("object_id");
            if (!id_opt) return BT::NodeStatus::FAILURE;

            std::string full_id = id_opt.value();
            std::string label = full_id;
            size_t pos = full_id.find('_');
            if (pos != std::string::npos) label = full_id.substr(0, pos);

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

        // --- Action: ComputePoseToOrganize ---
        factory_.registerSimpleAction("ComputePoseToOrganize", [&](BT::TreeNode &self)
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

        // --- Action: ComputePoseToStore ---
        factory_.registerSimpleAction("ComputePoseToStore", [&](BT::TreeNode &self)
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

        // --- Action: IncrementOrganizedStorageIndexes ---
        factory_.registerSimpleAction("IncrementOrganizedStorageIndexes", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("storage_id");
            auto newIndexes = self.getInput<std::vector<int>>("new_indexes");
            if (!id_opt || !newIndexes) return BT::NodeStatus::FAILURE;

            storage_node_->addNewIndexes(id_opt.value(), newIndexes.value());
            RCLCPP_WARN(this->get_logger(), "Storage '%s' atualizado.", id_opt.value().c_str());
            return BT::NodeStatus::SUCCESS;
        },
        { BT::InputPort<std::string>("storage_id"), BT::InputPort<std::vector<int>>("new_indexes") });

        // --- Action: DecrementStorageCount ---
        factory_.registerSimpleAction("DecrementStorageCount", [&](BT::TreeNode &self)
        {
            auto id_opt = self.getInput<std::string>("storage_id");
            if (!id_opt) return BT::NodeStatus::FAILURE;

            storage_node_->incrementStorageCount(id_opt.value(), -1);
            RCLCPP_WARN(this->get_logger(), "ROLLBACK: Espaço liberado em '%s'.", id_opt.value().c_str());
            return BT::NodeStatus::SUCCESS;
        },
        { BT::InputPort<std::string>("storage_id") });

        // --- Condition: IsGripperHoldingObject ---
        factory_.registerSimpleCondition("IsGripperHoldingObject",
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

        // --- Action: ComputePath (Assíncrona) ---
        BT::NodeBuilder builder_compute = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    
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
                    if (path_state_ == TaskState::RUNNING) return BT::NodeStatus::RUNNING;
                }

                auto target = self.getInput<geometry_msgs::msg::Pose>("target");
                if (!target) 
                {
                    RCLCPP_ERROR(this->get_logger(), "ComputePath: Target inválido na Blackboard.");
                    rclcpp::sleep_for(std::chrono::milliseconds(2000)); 
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
        factory_.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "ComputePath", { BT::InputPort<geometry_msgs::msg::Pose>("target"), BT::InputPort<std::string>("planner") }, {} }, builder_compute);

        // --- Action: FollowPath (Assíncrona) ---
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
                    RCLCPP_WARN(this->get_logger(), "FollowPath: HALT recebido! Cancelando Action...");
                    this->cancel_controller_goal();
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    nav_state_ = TaskState::IDLE;
                }
            );
        });

        // --- Action: PickObject ---
        BT::NodeBuilder builder_pick = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
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
                    if (current_pick_phase_ == GraspPhase::IDLE)
                    {
                        std::vector<geometry_msgs::msg::Pose> poses = {};
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                            "PICK TARGET -> Pose: [x: %.3f, y: %.3f, z: %.3f] | Size: [x: %.3f, y: %.3f, z: %.3f]",
                            target.position.x, target.position.y, target.position.z,
                            target_size.x, target_size.y, target_size.z);

                                                
                        
                        auto scan_data_opt = this->scan_object_node_->getSortedScanPoses(cached_object_.id);

                        if (scan_data_opt.has_value())
                        {
                            auto [raw_poses, robot_pos_tf] = scan_data_opt.value();
                            std::tuple<float, float, float> robot_pos_tuple = std::make_tuple(
                                (float)robot_pos_tf.x(),
                                (float)robot_pos_tf.y(),
                                (float)robot_pos_tf.z()
                            );

                            std::vector<geometry_msgs::msg::Pose> valid_poses = this->ik_validator_node_->find_valid_targets_from_base(
                                robot_pos_tuple, raw_poses, true, " "); 

                            if (!valid_poses.empty())
                            {
                                RCLCPP_INFO(this->get_logger(), "Encontradas %zu poses válidas.", valid_poses.size());
                                poses = this->scan_object_node_->getOptimizedScanPoses(valid_poses, cached_object_.id);
                                RCLCPP_INFO(this->get_logger(), "Poses: %zu.", poses.size());
                            }
                            else
                            {
                                RCLCPP_WARN(this->get_logger(), "Nenhuma pose de scan é alcançável (IK falhou ou colisão).");
                                return BT::NodeStatus::RUNNING;
                            }
                        }
                        else
                        {
                            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                                "Falha ao obter poses: Objeto não encontrado ou posição do robô desconhecida.");
                            return BT::NodeStatus::RUNNING;
                        }

                        this->object_mapping_node_->ObjectToMap(cached_object_.id);
                        

                        if(!poses.empty())
                        {
                            for (auto & pose : poses)
                            {
                                tf2::Quaternion q_scan_result;
                                tf2::fromMsg(pose.orientation, q_scan_result);
                                tf2::Quaternion q_gripper_offset;
                                q_gripper_offset.setRPY(M_PI, 0.0, -M_PI / 4.0);
                                tf2::Quaternion q_final = q_scan_result * q_gripper_offset;
                                q_final.normalize();
                                pose.orientation = tf2::toMsg(q_final);
                            }
                            
                            for(size_t i = 0; i < 5; i++) 
                            { 
                                if (i < poses.size())
                                    RCLCPP_INFO(this->get_logger(), "Pose %ld ajustada enviada.", i);
                            }

                            this->send_goal(id.value(), poses, true, true); 
                            current_pick_phase_ = GraspPhase::GRASPNET_SCAN;
                            return BT::NodeStatus::RUNNING;
                        }                      
                    }
                    
                    if (current_pick_phase_ == GraspPhase::GRASPNET_SCAN)
                    {
                        auto duration = rclcpp::Duration(2, 0); 
                        auto start_time = this->get_clock()->now();

                        while ((this->get_clock()->now() - start_time) < duration) 
                        {
                            if (!rclcpp::ok()) break;

                            std::vector<geometry_msgs::msg::Pose> tcp_poses = this->bridge_to_inference_node_->get_latest_grasps();
                            std::vector<geometry_msgs::msg::Pose> wrist_poses; 

                           
                            double retreat_dist = 0.1034; 

                            
                            tf2::Transform tf_correction;
                            tf_correction.setIdentity();

                            
                            tf_correction.setOrigin(tf2::Vector3(-retreat_dist, 0.0, 0.0));

                        
                            tf2::Quaternion q_rot;
                            q_rot.setRPY(0.0, 1.57079632679, 0.0); 
                            tf_correction.setRotation(q_rot);

                            for (const auto& tcp_pose_msg : tcp_poses) {
                                
                                
                                tf2::Transform tf_world_to_grasp;
                                tf2::fromMsg(tcp_pose_msg, tf_world_to_grasp);

                                
                                tf2::Transform tf_world_to_wrist = tf_world_to_grasp * tf_correction;

                                
                                geometry_msgs::msg::Pose wrist_pose_msg;
                                tf2::toMsg(tf_world_to_wrist, wrist_pose_msg);
                                
                               
                                tf2::Quaternion q_final = tf_world_to_wrist.getRotation();
                                q_final.normalize();
                                wrist_pose_msg.orientation = tf2::toMsg(q_final);

                                wrist_poses.push_back(wrist_pose_msg);
                            }

                            // Publique wrist_poses e veja no Rviz
                                                        
                            if(wrist_poses.empty())
                            {
                                this->grasp_context_.grasp_poses.clear();
                                return BT::NodeStatus::RUNNING;
                            }
                            else
                            {

                                geometry_msgs::msg::PoseArray msg;

                                
                                msg.header.stamp = this->now();
                                msg.header.frame_id = "world"; 

                                
                                msg.poses = wrist_poses;

                                
                                if (pose_array_publisher_) {
                                    pose_array_publisher_->publish(msg);
                                }
                                    
                                

                                std::vector<geometry_msgs::msg::Pose> validated_poses = this->ik_validator_node_->find_valid_targets_from_base(
                                    std::make_tuple(pose_x, pose_y, pose_z), wrist_poses, true, cached_object_.id);

                                    
                                if(validated_poses.empty())
                                {
                                    return BT::NodeStatus::RUNNING;
                                }

                                                                
                                this->grasp_context_.grasp_poses = validated_poses;

                                RCLCPP_INFO(this->get_logger(), 
                                "VAI SE FERRAR -> Pose: [x: %.3f, y: %.3f, z: %.3f] | Orient: [x: %.3f, y: %.3f, z: %.3f, w: %.3f]",
                                this->grasp_context_.grasp_poses[0].position.x, 
                                this->grasp_context_.grasp_poses[0].position.y, 
                                this->grasp_context_.grasp_poses[0].position.z,
                                this->grasp_context_.grasp_poses[0].orientation.x,
                                this->grasp_context_.grasp_poses[0].orientation.y,
                                this->grasp_context_.grasp_poses[0].orientation.z,
                                this->grasp_context_.grasp_poses[0].orientation.w);
 
                                break;
                            }
                        }

                        if (this->active_arm_handle_ != nullptr)
                        {
                            RCLCPP_INFO(this->get_logger(), "Enviando solicitação de CANCELAMENTO do goal...");
                            current_pick_phase_ = GraspPhase::WAITING;
                            auto future_cancel = this->client_ptr_->async_cancel_goal(this->active_arm_handle_);
                            std::future_status cancel_status = future_cancel.wait_for(std::chrono::seconds(10));

                            if (cancel_status == std::future_status::ready)
                            {
                                try
                                {
                                    auto cancel_response = future_cancel.get();
                                    
                                    if (cancel_response->return_code == action_msgs::srv::CancelGoal::Response::ERROR_NONE)
                                    {
                                        RCLCPP_INFO(this->get_logger(), "Pedido de cancelamento ACEITO. Aguardando confirmação...");
                                        
                                        auto future_result = this->client_ptr_->async_get_result(this->active_arm_handle_);
                                        std::future_status result_status = future_result.wait_for(std::chrono::seconds(10));
                                        
                                        if (result_status == std::future_status::ready)
                                        {
                                            auto wrapped_result = future_result.get();
                                            
                                            // Verifica o status final do goal
                                            switch (wrapped_result.code)
                                            {
                                                case rclcpp_action::ResultCode::CANCELED:
                                                    RCLCPP_INFO(this->get_logger(), "Action CANCELADA com sucesso!");
                                                    current_pick_phase_ = GraspPhase::SEND_GOAL;
                                                    this->active_arm_handle_ = nullptr;
                                                    break;
                                                    
                                                case rclcpp_action::ResultCode::SUCCEEDED:
                                                    RCLCPP_WARN(this->get_logger(), "Action terminou com SUCESSO antes do cancelamento.");
                                                    current_pick_phase_ = GraspPhase::SEND_GOAL;
                                                    this->active_arm_handle_ = nullptr;
                                                    break;
                                                    
                                                case rclcpp_action::ResultCode::ABORTED:
                                                    RCLCPP_WARN(this->get_logger(), "Action foi ABORTADA (não cancelada).");
                                                    current_pick_phase_ = GraspPhase::SEND_GOAL;
                                                    this->active_arm_handle_ = nullptr;
                                                    break;
                                                    
                                                case rclcpp_action::ResultCode::UNKNOWN:
                                                    RCLCPP_ERROR(this->get_logger(), "Status DESCONHECIDO da action.");
                                                    break;
                                                    
                                                default:
                                                    RCLCPP_ERROR(this->get_logger(), "ResultCode inesperado: %d", (int)wrapped_result.code);
                                                    break;
                                            }
                                        }
                                        else
                                        {
                                            RCLCPP_ERROR(this->get_logger(), "Timeout aguardando resultado final da action.");
                                        }
                                    }
                                    
                                }
                                catch (const std::exception &e)
                                {
                                    RCLCPP_ERROR(this->get_logger(), "Exceção: %s", e.what());
                                }
                            }
                            else
                            {
                                RCLCPP_ERROR(this->get_logger(), "Timeout no pedido de cancelamento.");
                            }
                        }
                        else
                        {
                            current_pick_phase_ = GraspPhase::SEND_GOAL;
                        }
                        return BT::NodeStatus::RUNNING;
                    }

                    if (current_pick_phase_ == GraspPhase::SEND_GOAL)
                    {
                        this->grasp_context_.graspnet_attempts += 1;

                        // std::cout << "ALOOOOOO" << std::endl;
                        if(!this->grasp_context_.grasp_poses.empty() && this->active_arm_handle_ == nullptr)
                        {
                            std::cout << cached_object_.id << std::endl;
                             RCLCPP_INFO(this->get_logger(), 
                            "MERDA DO CARAMBA -> Pose: [x: %.3f, y: %.3f, z: %.3f] | Orient: [x: %.3f, y: %.3f, z: %.3f, w: %.3f]",
                            this->grasp_context_.grasp_poses[0].position.x, 
                            this->grasp_context_.grasp_poses[0].position.y, 
                            this->grasp_context_.grasp_poses[0].position.z,
                            this->grasp_context_.grasp_poses[0].orientation.x,
                            this->grasp_context_.grasp_poses[0].orientation.y,
                            this->grasp_context_.grasp_poses[0].orientation.z,
                            this->grasp_context_.grasp_poses[0].orientation.w);

                            if (std::isnan(this->grasp_context_.grasp_poses[0].position.x) || std::isnan(this->grasp_context_.grasp_poses[0].orientation.w)) {
                                RCLCPP_ERROR(this->get_logger(), "ERRO CRÍTICO: Tentativa de enviar Pose com NaN!");
                                return BT::NodeStatus::RUNNING;
                            }
                            this->send_goal(id.value(),{this->grasp_context_.grasp_poses[0]}, true, false);
                            current_pick_phase_ = GraspPhase::WAITING;
                        }
                        return BT::NodeStatus::RUNNING;
                    }
                    
                   return BT::NodeStatus::RUNNING;
                }
                // else if(use_pca == true)
                // {

                // }
                else 
                {
                    // Lógica original sem GraspNet (se necessário)
                    if (manipulation_state_ == TaskState::IDLE)
                    {
                        std::vector<geometry_msgs::msg::Pose> poses = {object_pose.value()};
                        this->send_goal(id.value(), poses, true, false);
                        manipulation_state_ = TaskState::RUNNING;
                        return BT::NodeStatus::RUNNING;
                    }
                    return check_task_status(manipulation_state_);
                }
                
                return BT::NodeStatus::RUNNING;
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

        // --- Action: PlaceObject ---
        BT::NodeBuilder builder_place = [&](const std::string& name, const BT::NodeConfig& config)
        {
            return std::make_unique<AsyncAction>(name, config, [&](BT::TreeNode &self)
            {
                if (manipulation_state_ == TaskState::IDLE)
                {
                    auto pose = self.getInput<geometry_msgs::msg::Pose>("pose");
                    if (!pose) return BT::NodeStatus::FAILURE;

                    std::string id_dummy = cached_object_.id;
                    std::vector<geometry_msgs::msg::Pose> poses;
                    poses.push_back(pose.value());
                    this->send_goal(id_dummy, poses, false, false); // false = Place
                    manipulation_state_ = TaskState::RUNNING;
                    return BT::NodeStatus::RUNNING;
                }
                return check_task_status(manipulation_state_);
            });
        };
        factory_.registerBuilder(BT::TreeNodeManifest{BT::NodeType::ACTION, "PlaceObject", { BT::InputPort<geometry_msgs::msg::Pose>("pose"), BT::InputPort<std::vector<double>>("limits") }, {} }, builder_place);
        
        if (!use_llm)
        {
            try
            {
                bt_tree_ = std::make_unique<BT::Tree>(factory_.createTreeFromFile(xml_path));
                groot_publisher_ = std::make_unique<BT::Groot2Publisher>(*bt_tree_, 1666);
                RCLCPP_INFO(this->get_logger(), "Groot 2 Publisher iniciado na porta 1666");
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Erro Fatal ao criar Tree: %s", e.what());
            }
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Factory configurada. Aguardando árvore dinâmica...");
        }
    }

    void update_current_target_from_db()
    {
        
        if (current_target_id_.empty() || !db_handler_) return;

        std::lock_guard<std::mutex> lock(bt_mutex_);

        auto props = db_handler_->get_object_data(current_target_id_);
        if (props)
        {
            std::vector<double> p = parse_string_to_vector(props->pose_str);
            std::vector<double> s = parse_string_to_vector(props->size_str);

            if (p.size() >= 3)
            {
                // Atualiza Pose
                cached_object_.pose.position.x = p[0];
                cached_object_.pose.position.y = p[1];
                cached_object_.pose.position.z = p[2];

                if (p.size() >= 7) 
                {
                    cached_object_.pose.orientation.x = p[3];
                    cached_object_.pose.orientation.y = p[4];
                    cached_object_.pose.orientation.z = p[5];
                    cached_object_.pose.orientation.w = p[6];
                } 
                else 
                {
                    cached_object_.pose.orientation.w = 1.0;
                }
            }

            if (s.size() >= 3)
            {
                cached_object_.size.x = s[0];
                cached_object_.size.y = s[1];
                cached_object_.size.z = s[2];
            }
            
            current_target_pose_ = cached_object_.pose;
            
            
            // RCLCPP_DEBUG(this->get_logger(), "Dados do objeto '%s' atualizados do DB.", current_target_id_.c_str());
        }
    }

    // --- LOOP BT ---

    void bt_loop()
    {
        rclcpp::sleep_for(std::chrono::milliseconds(4000)); 
        rclcpp::Rate rate(50);
        while (rclcpp::ok())
        {
            // Lógica de Troca de Árvore (Feature LLM)
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
                    
                    if (pos != std::string::npos) 
                    {
                        modified_xml.replace(pos, 31, "main_tree_to_execute=\"" + unique_tree_name + "\"");
                    }
                    
                    pos = modified_xml.find("ID=\"MainPlan\"");
                    if (pos != std::string::npos) 
                    {
                        modified_xml.replace(pos, 13, "ID=\"" + unique_tree_name + "\"");
                    }
                    
                    RCLCPP_INFO(this->get_logger(), "Registrando árvore dinâmica: %s", unique_tree_name.c_str());
                    
                    groot_publisher_.reset(); 
                    factory_.registerBehaviorTreeFromText(modified_xml); 
                    bt_tree_ = std::make_unique<BT::Tree>(factory_.createTree(unique_tree_name)); 
                    
                    try 
                    {
                        groot_publisher_ = std::make_unique<BT::Groot2Publisher>(*bt_tree_, 1666);
                    } 
                    catch (...) {}
                    
                    RCLCPP_INFO(this->get_logger(), "Nova árvore carregada!");

                    { 
                        std::lock_guard<std::mutex> slock(state_mutex_); 
                        path_state_ = TaskState::IDLE; 
                    }
                    nav_state_ = TaskState::IDLE; 
                    manipulation_state_ = TaskState::IDLE;
                }
                catch (const std::exception& e)
                {
                    RCLCPP_ERROR(this->get_logger(), "Erro ao criar árvore dinâmica: %s", e.what());
                    bt_tree_.reset();
                    continue;
                }
            }

            // Execução padrão da BT
            if (bt_tree_ && bt_tree_->rootNode())
            {
                BT::NodeStatus status = bt_tree_->rootNode()->status();

                // LÓGICA ALEATÓRIA DO DB QUANDO LLM DESATIVADA
                if (!use_llm && status == BT::NodeStatus::IDLE && current_target_id_.empty() && !has_new_object_)
                {
                    fetch_random_object_from_db();
                }

                bool new_obj = false;
                {
                    std::lock_guard<std::mutex> lock(bt_mutex_);
                    new_obj = has_new_object_;
                }

                update_current_target_from_db();

                if (status == BT::NodeStatus::RUNNING || new_obj || !current_target_id_.empty() || status == BT::NodeStatus::IDLE)
                {
                    BT::NodeStatus result = bt_tree_->tickOnce();

                    if (result == BT::NodeStatus::SUCCESS || result == BT::NodeStatus::FAILURE)
                    {
                        std::lock_guard<std::mutex> lock(bt_mutex_);
                        has_new_object_ = false;

                        if (result == BT::NodeStatus::FAILURE)
                        {
                            picked.erase(cached_object_.id); 
                            current_target_id_ = "";
                        }

                        {
                            std::lock_guard<std::mutex> slock(state_mutex_);
                            path_state_ = TaskState::IDLE;
                        }
                        nav_state_ = TaskState::IDLE;
                        manipulation_state_ = TaskState::IDLE;
                        current_pick_phase_ = GraspPhase::IDLE;
                        
                        if (use_llm) 
                        {
                            RCLCPP_INFO(this->get_logger(), "Execução da árvore LLM finalizada.");
                            groot_publisher_.reset();
                            bt_tree_.reset();
                        }
                    }
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
        std::lock_guard<std::mutex>lock(odom_mutex);

        pose_x = msg->pose.pose.position.x;
        pose_y = msg->pose.pose.position.y;
        pose_z = 0.11; 
    }
    // std::unordered_set<std::string> blocked_ids;

    // Função de leitura do DB para o modo aleatório
    void fetch_random_object_from_db()
    {
        if (!db_read_conn_) return;

        
        {
            std::lock_guard<std::mutex> lock(bt_mutex_);
            if (has_new_object_) return;
        }

        std::string sql = "SELECT id, pose, size FROM objects ORDER BY RANDOM();";
        sqlite3_stmt* stmt;

       
        if (sqlite3_prepare_v2(db_read_conn_, sql.c_str(), -1, &stmt, 0) == SQLITE_OK) 
        {
            while (sqlite3_step(stmt) == SQLITE_ROW) 
            {
                std::string id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                
                
                std::string label = id;
                size_t pos = id.find('_');
                if (pos != std::string::npos) label = id.substr(0, pos);

                if (authorized_labels.find(label) == authorized_labels.end()) continue;

               
                bool is_invalid = false;
                {
                    std::lock_guard<std::mutex> lock(bt_mutex_);
                    if (picked.find(id) != picked.end()) is_invalid = true;
                    // if (blocked_ids.find(id) != blocked_ids.end()) is_invalid = true;
                }

                if (is_invalid) continue;
               
                std::string pose_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                std::string size_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

                std::vector<double> p_vec = parse_string_to_vector(pose_str);
                std::vector<double> s_vec = parse_string_to_vector(size_str);

                if (p_vec.size() >= 3 && s_vec.size() >= 3)
                {
                    std::lock_guard<std::mutex> lock(bt_mutex_);
                    
                 
                    if (picked.find(id) != picked.end()) continue; 

                    geometry_msgs::msg::Pose pose;
                    pose.position.x = p_vec[0];
                    pose.position.y = p_vec[1];
                    pose.position.z = p_vec[2];
                    pose.orientation.w = 1.0; 

                    cached_object_.id = id;
                    cached_object_.pose = pose;
                    cached_object_.size.x = s_vec[0];
                    cached_object_.size.y = s_vec[1];
                    cached_object_.size.z = s_vec[2];
                    
                    has_new_object_ = true;
                    
                    RCLCPP_INFO(this->get_logger(), "DB Random Pick: Selecionado '%s'", id.c_str());
                    break; 
                }
            }
            sqlite3_finalize(stmt);
        }
        else
        {
            // Log do erro real do SQLite (ajuda a descobrir se é erro de SQL ou Database Locked)
            RCLCPP_ERROR(this->get_logger(), "SQL Prepare Error: %s", sqlite3_errmsg(db_read_conn_));
        }
    }

    std::vector<double> parse_string_to_vector(const std::string& s)
    {
        std::vector<double> v;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ';')) 
        {
            try { v.push_back(std::stod(item)); } catch (...) { v.push_back(0.0); }
        }
        return v;
    }

    void cancel_controller_goal()
    {
        if (this->active_controller_goal_handle_)
        {
            RCLCPP_WARN(this->get_logger(), "Solicitando PARADA IMEDIATA...");
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
            RCLCPP_ERROR(this->get_logger(), "O servidor REJEITOU o pedido de Path Planning.");
        } 
        else 
        {
            this->active_path_goal_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Pedido aceito pelo servidor. Calculando...");
        }
    }

    void path_result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::Path>::WrappedResult & result)
    {
        std::lock_guard<std::mutex> lock(state_mutex_); 

        if (!this->active_path_goal_handle_ || result.goal_id != this->active_path_goal_handle_->get_goal_id()) 
        {
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
                RCLCPP_INFO(this->get_logger(), "Path Calculation: SUCCESS (%zu poses)", this->last_calculated_path_.poses.size());
                path_state_ = TaskState::SUCCESS; 
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

    bool send_controller_goal(const nav_msgs::msg::Path &target_path)
    {
        if (!this->controller_client->wait_for_action_server(std::chrono::seconds(5))) 
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
            this->active_controller_goal_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Goal CONTROLLER aceito.");
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
            RCLCPP_ERROR(this->get_logger(), "Controller: FALHOU (Aborted).");
            nav_state_ = TaskState::FAILURE;
        }
    }

    void send_goal(const std::string id, const std::vector<geometry_msgs::msg::Pose> & target_poses, bool pick, bool follow_path)
    {
        if (!this->client_ptr_->wait_for_action_server(std::chrono::seconds(10)))
        {
            RCLCPP_ERROR(this->get_logger(), "Action server manipulação not available");

            if(use_graspnet == true)
            {
                if(current_pick_phase_ == GraspPhase::SEND_GOAL && this->grasp_context_.graspnet_attempts <= this->grasp_context_.graspnet_maximum_attempts)
                {
                    current_pick_phase_ = GraspPhase::GRASPNET_SCAN;
                }
                else if(current_pick_phase_ == GraspPhase::SEND_GOAL)
                {
                    current_pick_phase_ = GraspPhase::FAILURE; 
                }
            }
            else
            {
                manipulation_state_ = TaskState::FAILURE;
            }
            
            return;
        }

        auto goal_msg = mobile_manipulation_interfaces::action::PickObject::Goal();
        goal_msg.obstacle_id = id;
        goal_msg.pick = pick;
        goal_msg.follow_path = follow_path;
        
        goal_msg.poses = target_poses; 

        RCLCPP_INFO(this->get_logger(), "BT: Enviando Goal para MANIPULATION com %zu poses...", target_poses.size());

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

            if(use_graspnet == true)
            {
                if(current_pick_phase_ == GraspPhase::SEND_GOAL && this->grasp_context_.graspnet_attempts <= this->grasp_context_.graspnet_maximum_attempts)
                {
                    current_pick_phase_ = GraspPhase::GRASPNET_SCAN;
                }
                else if(current_pick_phase_ == GraspPhase::SEND_GOAL)
                {
                    current_pick_phase_ = GraspPhase::FAILURE; 
                }
            }
            else
            {
                manipulation_state_ = TaskState::FAILURE;
            }
        }
        else
        {
            this->active_arm_handle_ = goal_handle;
            RCLCPP_INFO(this->get_logger(), "Goal PICK aceito.");
        }
    }

    void result_callback(const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces::action::PickObject>::WrappedResult & result)
    {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result->success)
        {
            if(use_graspnet == true)
            {
                if(current_pick_phase_ == GraspPhase::SEND_GOAL)
                {
                    current_pick_phase_ = GraspPhase::SUCCESS; 
                }
            }
            else
            {
                manipulation_state_ = TaskState::SUCCESS;
            }
            
            RCLCPP_INFO(this->get_logger(), "PICK SUCCESS");
        }
        else
        {
            // if(use_graspnet == true)
            // {
            //     if(current_pick_phase_ == GraspPhase::SEND_GOAL && this->grasp_context_.graspnet_attempts <= this->grasp_context_.graspnet_maximum_attempts)
            //     {
            //         current_pick_phase_ = GraspPhase::GRASPNET_SCAN;
            //     }
            //     else if(current_pick_phase_ == GraspPhase::SEND_GOAL)
            //     {
            //         current_pick_phase_ = GraspPhase::FAILURE; 
            //     }
            // }
            // else
            // {
            //     manipulation_state_ = TaskState::FAILURE;
            // }
            RCLCPP_ERROR(this->get_logger(), "PICK FAILED");
        }

        this->active_arm_handle_ = nullptr;
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

bool has_flag(const std::vector<std::string>& args, const std::string& flag) 
{
    return std::find(args.begin(), args.end(), flag) != args.end();
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    // --- Configuração das Opções dos Nós ---

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

    rclcpp::NodeOptions bridge_to_inference_opts;
    bridge_to_inference_opts.arguments({"--ros-args", "-r", "__node:=bridge_to_inference"});

    rclcpp::NodeOptions scan_object_opts;
    scan_object_opts.arguments({"--ros-args", "-r", "__node:=scan_object"});

    rclcpp::NodeOptions object_mapping_opts;
    object_mapping_opts.arguments({"--ros-args", "-r", "__node:=object_mapping"});

    rclcpp::NodeOptions world_state_node_opts;
    world_state_node_opts.arguments({"--ros-args", "-r", "__node:=world_state_node"});

    // --- Criação de Todos os Nós ---

    auto organize_node = std::make_shared<storage_manager::OrganizeNode>(organize_opts);
    
    auto storage_node = std::make_shared<storage_manager::StorageNode>(storage_opts);
    
    auto gripper_node = std::make_shared<manipulation::IsGripperHolding>(gripper_opts);
    
    auto reachability_node = std::make_shared<manipulation::ProjectedReachabilityAnalysis>(reachability_opts);
    
    auto obstacle_graph_node = std::make_shared<navigation::SharedObstacleGraph>(obstacle_graph_opts);
    
    auto ik_validator_node = std::make_shared<manipulation::IKValidator>(ik_validator_opts);
    
    auto bridge_to_inference_node = std::make_shared<drl_to_pick_cpp::BridgeToInference>(bridge_to_inference_opts);
    
    auto scan_object_node = std::make_shared<vision::GenerateScanPoses>(scan_object_opts);
    
    auto object_mapping_node = std::make_shared<vision::ObjectMapping>(object_mapping_opts);
    
    auto world_state_node = std::make_shared<llms::WorldStateNode>(world_state_node_opts);
    
    // --- Inicialização do ServerNode ---

    auto server_node = std::make_shared<ServerNode>(
        gripper_node, 
        reachability_node, 
        ik_validator_node, 
        obstacle_graph_node, 
        storage_node, 
        organize_node, 
        bridge_to_inference_node, 
        scan_object_node, 
        object_mapping_node, 
        world_state_node
    );

    // --- Execução ---

    rclcpp::executors::MultiThreadedExecutor executor;
    
    executor.add_node(organize_node);
    executor.add_node(storage_node);
    executor.add_node(gripper_node);
    executor.add_node(reachability_node);
    executor.add_node(obstacle_graph_node);
    executor.add_node(ik_validator_node);
    executor.add_node(bridge_to_inference_node);
    executor.add_node(scan_object_node);
    executor.add_node(object_mapping_node);
    executor.add_node(world_state_node);
    executor.add_node(server_node);
    
    executor.spin();

    rclcpp::shutdown();
    return 0;
}