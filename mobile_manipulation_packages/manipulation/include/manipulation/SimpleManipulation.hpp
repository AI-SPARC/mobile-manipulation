#ifndef SIMPLE_MANIPULATION_HPP
#define SIMPLE_MANIPULATION_HPP

#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <unordered_map>
#include <atomic>
#include <deque>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include "mobile_manipulation_interfaces/action/pick_object.hpp"
#include "mobile_manipulation_interfaces/srv/mobile_object_collision.hpp"
#include <moveit/robot_trajectory/robot_trajectory.h>

#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
namespace manipulation {

class SimpleManipulation : public rclcpp::Node 
{

public:
    SimpleManipulation();
    ~SimpleManipulation();

private:

    struct LocationData
    {
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size;
    };

    struct StorageData
    {
        int max_x_objects, max_y_objects, max_z_objects;
        int x;
        int y;
        int z = 0;
        int direction;
        geometry_msgs::msg::Pose pose;
        geometry_msgs::msg::Vector3 size;
    };

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr subscription_1;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr publisher_2;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
    rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_publisher_;
    
    rclcpp::Client<mobile_manipulation_interfaces::srv::MobileObjectCollision>::SharedPtr client_;
    rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr get_planning_scene_client_;

    rclcpp_action::Server<mobile_manipulation_interfaces::action::PickObject>::SharedPtr action_server_;

    rclcpp::TimerBase::SharedPtr init_timer_;
    rclcpp::TimerBase::SharedPtr timer_;
  
   
    planning_scene_monitor::PlanningSceneMonitorPtr psm_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_arm;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_gripper;

    rclcpp::Node::SharedPtr moveit_node_;
    rclcpp::Executor::SharedPtr executor_;
    std::thread executor_thread_;

    std::string yaml_file, storages_yaml_file;
    std::unordered_map<std::string, std::vector<geometry_msgs::msg::Pose>> pick_and_place_poses;
    std::pair<std::string, geometry_msgs::msg::Pose> object;
    geometry_msgs::msg::Pose object_pose;

    std::mutex object_pose_mutex;
    std::mutex contact_sensor_mutex;
    std::deque<float> contact_sensor_data;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::atomic<bool> moveit_ready_{false};

    bool use_graspnet = false;

    void object_pose_callback(const geometry_msgs::msg::Pose & msg);
    void loadLocationsFromYaml(const std::string &yaml_path);
    void initMoveGroup();
    void ready();
    void close_gripper();
    void open_gripper();
    bool attempt_cartesian_move(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool avoid_collisions);
    bool positions_for_arm(const geometry_msgs::msg::Pose &target_pose, float maxVelocity, bool computeCartesian);
    bool calculate_global_pose(std::string received_id, geometry_msgs::msg::Pose pose, bool pick);
    void set_collision_allowance(const std::string& id1, const std::string& id2, bool allow_collision);
    bool send_request(std::string received_obstacle_id, bool received_activate_movement);
    bool follow_path_with_consistent_ik(const std::vector<geometry_msgs::msg::Pose>& path_poses, 
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>>& goal_handle);


    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const mobile_manipulation_interfaces::action::PickObject::Goal> goal);
    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle);
    void handle_accepted(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle);
    void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<mobile_manipulation_interfaces::action::PickObject>> goal_handle);

    void topic_callback(const std_msgs::msg::Float32 & msg);
};

} // namespace manipulation

#endif // SIMPLE_MANIPULATION_HPP