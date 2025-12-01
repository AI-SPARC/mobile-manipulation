#ifndef ADD_COLLISION_HPP
#define ADD_COLLISION_HPP

#include <memory>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include <shape_msgs/msg/solid_primitive.hpp>
#include "moveit_msgs/msg/collision_object.hpp"
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include "mobile_manipulation_interfaces/srv/mobile_object_collision.hpp"

namespace manipulation {

class AddCollision : public rclcpp::Node 
{
public:
    AddCollision();

private:
    struct LabelRule 
    {
        std::string label;
        bool is_prefix;
    };

    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;
    rclcpp::Service<mobile_manipulation_interfaces::srv::MobileObjectCollision>::SharedPtr service_;
    rclcpp::TimerBase::SharedPtr init_timer_;
    
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    std::string stop_moving_obstacle = "";
    std::unordered_set<std::string> added;
    std::unordered_map<std::string, geometry_msgs::msg::Pose> last_known_poses_;
    std::vector<LabelRule> authorized_labels_;
    std::vector<LabelRule> unauthorized_labels_;
    bool activate_movement = true;

    void load_labels_from_yaml(const std::string& file_path);
    void add_ground_plane();
    bool is_significant_change(const std::string& id, const geometry_msgs::msg::Pose& new_pose);
    void add_collision_box(const std::string &id, const std::array<double, 3> &dimensions, const geometry_msgs::msg::Pose &pose);
    void move_collision_box(const std::string &id, const geometry_msgs::msg::Pose &pose);
    bool is_authorized(const std::string& label);
    void detectionCallback(const vision_msgs::msg::Detection3DArray::SharedPtr msg);
    void handleStopService(
        const std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Request> request,
        std::shared_ptr<mobile_manipulation_interfaces::srv::MobileObjectCollision::Response> response);
};

} // namespace manipulation

#endif // ADD_COLLISION_HPP