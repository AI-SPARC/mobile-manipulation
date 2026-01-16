#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <regex>
#include <map>
#include <string>
#include <vector>
#include <memory>

using Detection3DArray = vision_msgs::msg::Detection3DArray;
using StringMsg = std_msgs::msg::String;


class CameraProcessor
{
public:
    CameraProcessor(rclcpp::Node* node, int camera_id)
        : camera_id_(camera_id)
    {
        std::string id_str = std::to_string(camera_id_);

        
        std::string detections_topic = "/bbox_3d_" + id_str;
        std::string mapping_topic = "/semantic_labels_" + id_str;
        std::string output_topic = "/bbox_3d_with_labels_" + id_str;

        
        publisher_ = node->create_publisher<Detection3DArray>(output_topic, 10);

        
        detections_sub_ = node->create_subscription<Detection3DArray>(
            detections_topic, 10,
            std::bind(&CameraProcessor::detections_callback, this, std::placeholders::_1));

        
        mapping_sub_ = node->create_subscription<StringMsg>(
            mapping_topic, 10,
            std::bind(&CameraProcessor::mapping_callback, this, std::placeholders::_1));

        RCLCPP_INFO(node->get_logger(), "Configurado processador para Camera %d nos topicos: %s, %s -> %s", 
            camera_id_, detections_topic.c_str(), mapping_topic.c_str(), output_topic.c_str());
    }

private:
    int camera_id_;
    rclcpp::Publisher<Detection3DArray>::SharedPtr publisher_;
    rclcpp::Subscription<Detection3DArray>::SharedPtr detections_sub_;
    rclcpp::Subscription<StringMsg>::SharedPtr mapping_sub_;
    std::map<std::string, std::string> label_map_;

   
    void mapping_callback(const StringMsg::SharedPtr msg)
    {
        label_map_ = parse_label_map(msg->data);
        // RCLCPP_INFO(rclcpp::get_logger("CameraProcessor"), "Camera %d: Mapa de labels atualizado (%zu itens).", camera_id_, label_map_.size());
    }

    
    void detections_callback(const Detection3DArray::SharedPtr msg)
    {
        if (label_map_.empty())
        {
            
            return;
        }

        auto labeled_msg = *msg;

        for (auto &det : labeled_msg.detections)
        {
            if (det.results.empty())
                continue;

            std::string id = det.results[0].hypothesis.class_id;

            auto it = label_map_.find(id);
            if (it != label_map_.end())
            {
                det.results[0].hypothesis.class_id = it->second;
            }
            else
            {
                det.results[0].hypothesis.class_id = "UNMAPPED_" + id;
            }
        }

        publisher_->publish(labeled_msg);
    }

    
    std::map<std::string, std::string> parse_label_map(const std::string &input)
    {
        std::map<std::string, std::string> result;

        
        std::regex pair_regex("\"([0-9]+)\"\\s*:\\s*\\{[^}]*\"([A-Za-z0-9_]+)\"\\s*:\\s*\"([A-Za-z0-9_]+)\"");

        std::smatch match;
        std::string::const_iterator search_start(input.cbegin());

        while (std::regex_search(search_start, input.cend(), match, pair_regex))
        {
            std::string id = match[1].str();
            std::string last_word = match[3].str();

            result[id] = last_word;
            search_start = match.suffix().first;
        }

        return result;
    }
};


class TranslatorNode : public rclcpp::Node
{
public:
    TranslatorNode()
        : Node("synchronize_isaac_sim_labels")
    {
       
        this->declare_parameter("num_cameras", 3);
        
        int num_cameras = this->get_parameter("num_cameras").as_int();

        RCLCPP_INFO(this->get_logger(), "Iniciando TranslatorNode para %d cameras...", num_cameras);

        
        for (int i = 0; i < num_cameras; ++i)
        {
            camera_processors_.push_back(std::make_shared<CameraProcessor>(this, i));
        }
    }

private:
    
    std::vector<std::shared_ptr<CameraProcessor>> camera_processors_;
};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TranslatorNode>());
    rclcpp::shutdown();
    return 0;
}