#include "manipulation/IKValidator.hpp"
#include "navigation/SharedObstacleGraph.hpp" 

#include "rclcpp_components/register_node_macro.hpp"
#include <sensor_msgs/point_cloud2_iterator.hpp> 
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <moveit/planning_scene/planning_scene.h> 
#include <limits>
#include <cmath>
#include <thread>
#include <tuple>

namespace manipulation {

IKValidator::IKValidator(const rclcpp::NodeOptions & options)
: Node("ik_validator_node", options)
{
    this->declare_parameter<std::string>("group_name", "panda_arm");
    group_name_ = this->get_parameter("group_name").as_string();

    init_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&IKValidator::delayed_init, this)
    );

    std::this_thread::sleep_for(std::chrono::milliseconds(2500));

    rclcpp::QoS qos(10);
    qos.reliable();
    qos.transient_local(); 
    publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/viable_ik_points", qos);
}

void IKValidator::delayed_init()
{
    if (initialized_.load()) { init_timer_->cancel(); return; }

    try {
        if (!robot_model_loader_) {
            robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(shared_from_this(), "robot_description");
        }
        if (!psm_) {
            psm_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(shared_from_this(), robot_model_loader_);
            psm_->startSceneMonitor();
            psm_->startWorldGeometryMonitor(); 
            psm_->startStateMonitor();
        }

        if (!psm_->getRobotModel()) return; 

        robot_model_ = psm_->getRobotModel();

        if (!robot_model_->hasJointModelGroup(group_name_)) {
            RCLCPP_ERROR(this->get_logger(), "Grupo '%s' não existe!", group_name_.c_str());
            return;
        }

        const auto& virtual_joints = robot_model_->getJointModels();
        for (const auto* joint : virtual_joints) {
            if (joint->getType() == moveit::core::JointModel::PLANAR || 
                joint->getType() == moveit::core::JointModel::FLOATING) {
                virtual_joint_name_ = joint->getName();
                RCLCPP_INFO(this->get_logger(), "Virtual Joint encontrada: %s", virtual_joint_name_.c_str());
                break;
            }
        }

        initialized_.store(true); 
        init_timer_->cancel(); 
        RCLCPP_INFO(this->get_logger(), "IK Validator PRONTO.");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Erro Init: %s", e.what());
    }
}

std::optional<std::tuple<float, float, float>> IKValidator::find_best_base_position(
    const std::vector<std::tuple<float, float, float>>& robot_positions, 
    const geometry_msgs::msg::Pose& target_pose_global,
    bool seed_mode,
    const std::shared_ptr<navigation::SharedObstacleGraph>& graph_provider_node,
    std::string authorized_collision
)
{
   
    int attempts = 0;
    while (!initialized_.load()) {
        if (attempts++ > 50) return std::nullopt;
        if (!rclcpp::ok()) return std::nullopt;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    

    auto start_total = std::chrono::high_resolution_clock::now();

    
    if (!graph_provider_node || !graph_provider_node->get_current_map()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Mapa 2D indisponível.");
        return std::nullopt; 
    }
    auto map_snapshot = graph_provider_node->get_current_map();

    
    psm_->requestPlanningSceneState();
    planning_scene_monitor::LockedPlanningSceneRO parent_scene(psm_);
    if (!parent_scene) return std::nullopt;
    planning_scene::PlanningScenePtr temp_scene = parent_scene->diff();

    
    collision_detection::AllowedCollisionMatrix& acm = temp_scene->getAllowedCollisionMatrixNonConst();
    
    
    std::vector<std::string> gripper_links = {"panda_link8"}; 
    
    if (robot_model_->hasJointModelGroup("hand")) 
    {
        const auto& links = robot_model_->getJointModelGroup("hand")->getLinkModelNames();
        gripper_links.insert(gripper_links.end(), links.begin(), links.end());
    }

    if (!authorized_collision.empty()) 
    {
        
        for (const auto& link : gripper_links) 
        {
            if (robot_model_->hasLinkModel(link)) 
            {
                
                acm.setEntry(link, authorized_collision, true);
            }
        }
    }
    else 
    {
        
        RCLCPP_DEBUG(this->get_logger(), "Nenhum objeto autorizado fornecido. Colisão padrão mantida.");
    }
    // =================================================================================

    moveit::core::RobotState& local_state = temp_scene->getCurrentStateNonConst();
    const moveit::core::JointModelGroup* arm_jmg = local_state.getJointModelGroup(group_name_);

    
    moveit::core::GroupStateValidityCallbackFn validity_callback = 
        [&temp_scene, &acm](moveit::core::RobotState* state, const moveit::core::JointModelGroup* group, const double* values) -> bool
        {
            state->setJointGroupPositions(group, values);
            collision_detection::CollisionRequest req;
            req.group_name = group->getName();
            
            
            req.verbose = false; 
            req.contacts = false;
            
            collision_detection::CollisionResult res;
            
            
            temp_scene->checkCollision(req, res, *state, acm);
            return !res.collision;
        };

    std::vector<std::tuple<float, float, float>> successful_ik_points;
    std::optional<std::tuple<float, float, float>> best_base = std::nullopt;
    double min_dist_sq = std::numeric_limits<double>::max(); 
    int valid_count = 0;

    
    geometry_msgs::msg::Pose check_pose = target_pose_global;
    check_pose.position.z += 0.1; 

    tf2::Quaternion q_grip; 

    for (const auto& base_pos_3d : robot_positions) 
    {
        float bx = std::get<0>(base_pos_3d);
        float by = std::get<1>(base_pos_3d);
        float bz = std::get<2>(base_pos_3d);

        if (std::isnan(bx) || std::isnan(by) || std::isnan(bz)) continue;

        
        std::pair<float, float> base_pos_2d = {bx, by};
        if (map_snapshot->find(base_pos_2d) != map_snapshot->end()) continue; 

        double dx = check_pose.position.x - bx;
        double dy = check_pose.position.y - by;
        double dz = check_pose.position.z - bz;
        double dist_3d = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (dist_3d > 0.95 || dist_3d < 0.15) continue; 

        
        double yaw_to_target = std::atan2(dy, dx);

        if (!virtual_joint_name_.empty()) {
            const auto* vjoint = robot_model_->getJointModel(virtual_joint_name_);
            if (vjoint->getType() == moveit::core::JointModel::FLOATING) {
                Eigen::Quaterniond q_base(Eigen::AngleAxisd(yaw_to_target, Eigen::Vector3d::UnitZ()));
                std::vector<double> float_vals = { (double)bx, (double)by, (double)bz, q_base.x(), q_base.y(), q_base.z(), q_base.w() };
                local_state.setJointPositions(virtual_joint_name_, float_vals);
            } else { 
                local_state.setJointPositions(virtual_joint_name_, {(double)bx, (double)by, yaw_to_target});
            }
        }
        local_state.update(); 

        
        q_grip.setRPY(M_PI, 0.0, yaw_to_target); 
        q_grip.normalize();
        check_pose.orientation = tf2::toMsg(q_grip);

        if (!seed_mode) local_state.setToDefaultValues(arm_jmg, "ready");

        
        bool found_ik = local_state.setFromIK(arm_jmg, check_pose, 0.015, validity_callback);

        if (found_ik)
        {
            valid_count++;
            successful_ik_points.push_back(base_pos_3d);
            
            if (dist_3d < min_dist_sq) {
                min_dist_sq = dist_3d;
                best_base = base_pos_3d;
            }
        }
    }
    
   
    successful_ik_points.emplace_back(check_pose.position.x, check_pose.position.y, check_pose.position.z);

    auto end_total = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count() / 1000.0;
    
    RCLCPP_INFO(this->get_logger(), "IK Scan 3D: %d aceitos / %zu total (%.2f ms)", 
        valid_count, robot_positions.size(), ms);
    
    publish_viable_ik_points(successful_ik_points); 
    
    return best_base;
}

void IKValidator::publish_viable_ik_points(const std::vector<std::tuple<float, float, float>>& results)
{
    if (!publisher_ || results.empty()) return;
    try {
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.stamp = this->now();
        cloud.header.frame_id = "world";
        cloud.height = 1;
        cloud.width = results.size();
        cloud.is_dense = true;
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(cloud.width);
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x"), iter_y(cloud, "y"), iter_z(cloud, "z");
        for (const auto& p : results) {
            *iter_x = std::get<0>(p); *iter_y = std::get<1>(p); *iter_z = std::get<2>(p);
            ++iter_x; ++iter_y; ++iter_z;
        }
        publisher_->publish(cloud);
    } catch (...) {}
}

} // namespace
RCLCPP_COMPONENTS_REGISTER_NODE(manipulation::IKValidator)