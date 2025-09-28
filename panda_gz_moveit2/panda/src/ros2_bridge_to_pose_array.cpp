#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "vision_msgs/msg/detection3_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"
#include <vector>
#include <string>
#include <unordered_map>

class MultiBoxDetectionArray : public rclcpp::Node
{
public:
    MultiBoxDetectionArray()
    : Node("multi_box_detection_array")
    {
        num_boxes_ = this->declare_parameter<int>("num_boxes", 3);

        detection_pub_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("/boxes_detection_array", 10);

        for (int i = 0; i < num_boxes_; i++) {
            std::string topic;
            if (i == 0) topic = "/model/object/pose";
            else topic = "/model/object" + std::to_string(i) + "/pose";

            auto sub = this->create_subscription<geometry_msgs::msg::Pose>(
                topic, 10,
                [this, i](geometry_msgs::msg::Pose::SharedPtr msg) {
                    latest_poses_[i] = *msg;
                });

            subscribers_.push_back(sub);
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&MultiBoxDetectionArray::publishDetectionArray, this));
    }

private:
    void publishDetectionArray() 
    {
        vision_msgs::msg::Detection3DArray detection_array;
        detection_array.header.stamp = this->get_clock()->now();
        detection_array.header.frame_id = "world";

        for (int i = 0; i < num_boxes_; i++) 
        {
            vision_msgs::msg::Detection3D detection;

            // Se existe pose, usa; senão pose padrão
            if (latest_poses_.count(i)) {
                detection.bbox.center = latest_poses_[i];
            } else {
                detection.bbox.center.position.x = 0.0;
                detection.bbox.center.position.y = 0.0;
                detection.bbox.center.position.z = 0.0;
                detection.bbox.center.orientation.w = 1.0;
            }

            // Adiciona classe = ID do tópico
            vision_msgs::msg::ObjectHypothesisWithPose hypo;
            hypo.hypothesis.class_id = std::to_string(i);
            hypo.hypothesis.score = 1.0;  // opcional
            detection.results.push_back(hypo);

            detection_array.detections.push_back(detection);
        }

        detection_pub_->publish(detection_array);
    }

    int num_boxes_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detection_pub_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr> subscribers_;
    std::unordered_map<int, geometry_msgs::msg::Pose> latest_poses_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MultiBoxDetectionArray>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
