#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <regex>
#include <map>
#include <string>

using Detection3DArray = vision_msgs::msg::Detection3DArray;
using StringMsg = std_msgs::msg::String;

class TranslatorNode : public rclcpp::Node
{
public:
    TranslatorNode()
        : Node("synchronize_isaac_sim_labels")
    {
        std::string detections_topic = "/bbox_3d";
        std::string mapping_topic = "/semantic_labels";
        std::string output_topic = "/bbox_3d_with_labels";

        publisher_ = this->create_publisher<Detection3DArray>(output_topic, 10);

        detections_sub_ = this->create_subscription<Detection3DArray>(
            detections_topic, 10,
            std::bind(&TranslatorNode::detections_callback, this, std::placeholders::_1));

        mapping_sub_ = this->create_subscription<StringMsg>(
            mapping_topic, 10,
            std::bind(&TranslatorNode::mapping_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Nó iniciado. Aguardando mensagens...");
    }

private:
    rclcpp::Publisher<Detection3DArray>::SharedPtr publisher_;
    rclcpp::Subscription<Detection3DArray>::SharedPtr detections_sub_;
    rclcpp::Subscription<StringMsg>::SharedPtr mapping_sub_;
    std::map<std::string, std::string> label_map_;

    // -------------------------------------------------------
    void mapping_callback(const StringMsg::SharedPtr msg)
    {
        label_map_ = parse_label_map(msg->data);
        // RCLCPP_INFO(this->get_logger(), "Mapa de labels atualizado (%zu itens).", label_map_.size());
    }

    // -------------------------------------------------------
    void detections_callback(const Detection3DArray::SharedPtr msg)
    {
        if (label_map_.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Mapa de labels vazio. Ignorando detecções.");
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

    // -------------------------------------------------------
    std::map<std::string, std::string> parse_label_map(const std::string &input)
    {
        std::map<std::string, std::string> result;

        // Captura tanto "X":{"class":"YYY"} quanto "X":{"YYY":"YYY"}
        std::regex pair_regex("\"([0-9]+)\"\\s*:\\s*\\{[^}]*\"([A-Za-z0-9_]+)\"\\s*:\\s*\"([A-Za-z0-9_]+)\"");

        std::smatch match;
        std::string::const_iterator search_start(input.cbegin());

        while (std::regex_search(search_start, input.cend(), match, pair_regex))
        {
            std::string id = match[1].str();
            std::string last_word = match[3].str(); // última palavra capturada

            result[id] = last_word;
            search_start = match.suffix().first;
        }

        return result;
    }
};

// -------------------------------------------------------
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TranslatorNode>());
    rclcpp::shutdown();
    return 0;
}
