#ifndef MANIPULATION_ADD_COLLISION_HPP_
#define MANIPULATION_ADD_COLLISION_HPP_

#include <memory>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <array>
#include <mutex>

// ROS2
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit_msgs/msg/collision_object.hpp"
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include "mobile_manipulation_interfaces/srv/mobile_object_collision.hpp"

// SQLite
#include <sqlite3.h>

namespace manipulation {

struct LabelRule {
    std::string label;
    bool is_prefix;
};

class AddCollision : public rclcpp::Node
{
public:
    explicit AddCollision(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~AddCollision();

private:
    
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    rclcpp::Service<mobile_manipulation_interfaces::srv::MobileObjectCollision>::SharedPtr service_;
    rclcpp::TimerBase::SharedPtr init_timer_;   
    rclcpp::TimerBase::SharedPtr db_timer_;    

   
    std::vector<LabelRule> authorized_labels_;
    std::vector<LabelRule> unauthorized_labels_;
    std::unordered_set<std::string> added;
    std::unordered_map<std::string, geometry_msgs::msg::Pose> last_known_poses_;

    std::string stop_moving_obstacle;
    bool activate_movement;

  
    sqlite3* db_;
    std::string db_path_;

 
    void load_labels_from_yaml(const std::string& file_path);
    void connect_database();
    void sync_from_database(); 
    
    
    std::vector<double> parse_string_to_vector(const std::string& s);

  
    void add_ground_plane();
    bool is_significant_change(const std::string& id, const geometry_msgs::msg::Pose& new_pose);
    bool is_authorized(const std::string& label);
    void add_collision_box(const std::string &id, const std::array<double, 3> &dimensions, const geometry_msgs::msg::Pose &pose);
    void move_collision_box(const std::string &id, const geometry_msgs::msg::Pose &pose);

    void handleStopService(
        const std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Request> request,
        std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Response> response);
};

} // namespace manipulation

#endif // MANIPULATION_ADD_COLLISION_HPP_