#ifndef LLMS__WORLD_STATE_NODE_HPP_
#define LLMS__WORLD_STATE_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp" // <--- Mudança aqui
#include <sqlite3.h>
#include <mutex>
#include <string>

namespace llms
{

class WorldStateNode : public rclcpp::Node
{
public:
    explicit WorldStateNode(const rclcpp::NodeOptions & options);
    ~WorldStateNode();

private:
    // Callback agora recebe array de detecções 3D
    void handle_detections(const vision_msgs::msg::Detection3DArray::SharedPtr msg);

    void init_database();

    // Upsert simplificado (apenas ID, Pose, Size)
    bool upsert_object(const std::string& id, 
                       const std::string& pose, 
                       const std::string& size);

    sqlite3* db_;
    std::mutex db_mutex_;
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr subscription_;
};

} // namespace llms

#endif // LLMS__WORLD_STATE_NODE_HPP_