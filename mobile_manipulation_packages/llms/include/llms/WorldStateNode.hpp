#ifndef LLMS_WORLD_STATE_NODE_HPP_
#define LLMS_WORLD_STATE_NODE_HPP_

#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include <sqlite3.h>

namespace llms
{

class WorldStateNode : public rclcpp::Node
{
public:
    explicit WorldStateNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~WorldStateNode();

private:
    sqlite3* db_;
    std::mutex db_mutex_;

    std::vector<rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr> subscriptions_;

    void init_database();
    
    void handle_detections(const vision_msgs::msg::Detection3DArray::SharedPtr msg);

    bool upsert_object(const std::string& id, const std::string& pose, const std::string& size);
};

} // namespace llms

#endif // LLMS_WORLD_STATE_NODE_HPP_