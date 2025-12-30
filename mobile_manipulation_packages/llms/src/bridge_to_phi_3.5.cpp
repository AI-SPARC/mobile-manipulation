#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <sstream>

#include <llms/OllamaClient.hpp> 
#include <llms/DatabaseHandler.hpp>

using json = nlohmann::json;

const std::string DB_PATH = "/home/momesso/AQUI/robot_data.db";

class BrainNode : public rclcpp::Node {
public:
    BrainNode() : Node("brain_node") 
    {
        llm_client_ = std::make_unique<OllamaClient>();
        db_handler_ = std::make_unique<DatabaseHandler>(DB_PATH);

        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/human_command", 10, 
            std::bind(&BrainNode::handle_command, this, std::placeholders::_1));
        
        bt_publisher_ = this->create_publisher<std_msgs::msg::String>("/behavior_tree_xml", 10);
            
        RCLCPP_INFO(this->get_logger(), "Brain Node pronto.");
    }

private:
    std::unique_ptr<OllamaClient> llm_client_;
    std::unique_ptr<DatabaseHandler> db_handler_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bt_publisher_;

    void handle_command(const std_msgs::msg::String::SharedPtr msg) 
    {
        RCLCPP_INFO(this->get_logger(), "Comando: '%s'", msg->data.c_str());

        json plan = llm_client_->infer(msg->data);
        if (plan.empty() || !plan.contains("commands")) {
            RCLCPP_ERROR(this->get_logger(), "Plano inválido!");
            return;
        }

        // Debug: mostra o JSON recebido
        RCLCPP_INFO(this->get_logger(), "JSON da LLM:\n%s", plan.dump(2).c_str());

        std::string xml = build_bt_xml(plan);
        
        if (!xml.empty()) {
            std_msgs::msg::String xml_msg;
            xml_msg.data = xml;
            bt_publisher_->publish(xml_msg);
            RCLCPP_INFO(this->get_logger(), "BT XML publicado:\n%s", xml.c_str());
        }
    }

    std::string build_bt_xml(const json& plan) 
    {
        std::stringstream ss;
        ss << "<root BTCPP_format=\"4\" main_tree_to_execute=\"MainPlan\">\n";
        ss << "  <BehaviorTree ID=\"MainPlan\">\n";
        ss << "    <Sequence name=\"LLM_Sequence\">\n";

        for (const auto& cmd : plan["commands"]) 
        {
            // Validação do JSON
            if (!cmd.contains("skill")) {
                RCLCPP_ERROR(this->get_logger(), "Comando sem 'skill': %s", cmd.dump().c_str());
                continue;
            }
            if (!cmd.contains("params") || !cmd["params"].contains("id")) {
                RCLCPP_ERROR(this->get_logger(), "Comando sem 'params.id': %s", cmd.dump().c_str());
                continue;
            }

            std::string skill = cmd["skill"];
            std::string target_id = cmd["params"]["id"];

            auto props = db_handler_->get_object_data(target_id);
            
            std::string subtree = build_subtree(skill, target_id, props);
            if (subtree.empty()) {
                RCLCPP_ERROR(this->get_logger(), "Skill desconhecida: %s", skill.c_str());
                continue;
            }
            ss << subtree;
        }

        ss << "    </Sequence>\n";
        ss << "  </BehaviorTree>\n";
        ss << "</root>";
        return ss.str();
    }

    std::string build_subtree(const std::string& skill, 
                              const std::string& target_id,
                              const std::optional<ObjectProperties>& props) 
    {
        std::stringstream ss;
        
        std::string skill_lower = skill;
        std::transform(skill_lower.begin(), skill_lower.end(), skill_lower.begin(), ::tolower);

        std::string id = props ? props->id : target_id;
        std::string pose = props ? props->pose_str : "";
        std::string size = props ? props->size_str : "";

        if (!props) {
            RCLCPP_WARN(this->get_logger(), "Objeto '%s' não encontrado no DB!", target_id.c_str());
        }

        if (skill_lower == "pick") 
        {
            ss << "      <SubTree ID=\"Pick\" "
               << "target_id=\"" << id << "\" "
               << "target_pose=\"" << pose << "\" "
               << "target_size=\"" << size << "\" "
               << "/>\n";
        }
        else if (skill_lower == "place") 
        {
            ss << "      <SubTree ID=\"Place\" "
               << "storage_id=\"" << id << "\" "
               << "final_placement_pose=\"" << pose << "\" "
               << "/>\n";
        }
        else if (skill_lower == "goto_location" || skill_lower == "goto") 
        {
            ss << "      <SubTree ID=\"GoToLocation\" "
               << "nav_target=\"" << pose << "\" "
               << "/>\n";
        }
        else 
        {
            return "";
        }
        
        return ss.str();
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BrainNode>());
    rclcpp::shutdown();
    return 0;
}