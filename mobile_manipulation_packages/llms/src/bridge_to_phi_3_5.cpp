// ============================================================================
// brain_node.cpp
// ============================================================================

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <vector>

#include <llms/OllamaClient.hpp>

using json = nlohmann::json;

class BrainNode : public rclcpp::Node {
public:
    BrainNode() : Node("brain_node") 
    {
        llm_client_ = std::make_unique<OllamaClient>();

        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/human_command", 10, 
            std::bind(&BrainNode::handle_command, this, std::placeholders::_1));
        
        bt_publisher_ = this->create_publisher<std_msgs::msg::String>("/behavior_tree_xml", 10);
            
        RCLCPP_INFO(this->get_logger(), "Brain Node pronto.");
    }

private:
    std::unique_ptr<OllamaClient> llm_client_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr bt_publisher_;

    void handle_command(const std_msgs::msg::String::SharedPtr msg) 
    {
        RCLCPP_INFO(this->get_logger(), "Comando: '%s'", msg->data.c_str());

        json plan = llm_client_->infer(msg->data);

        if (plan.empty() || !plan.contains("commands")) 
        {
            RCLCPP_ERROR(this->get_logger(), "Plano inválido!");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "JSON da LLM:\n%s", plan.dump(2).c_str());

        std::string xml = build_bt_xml(plan);
        
        if (xml.empty()) 
        {
            RCLCPP_ERROR(this->get_logger(), "Falha ao gerar XML!");
            return;
        }

        std_msgs::msg::String xml_msg;
        xml_msg.data = xml;
        bt_publisher_->publish(xml_msg);
        RCLCPP_INFO(this->get_logger(), "BT XML publicado:\n%s", xml.c_str());
    }

    std::optional<std::string> extract_skill(const json& cmd)
    {
        if (cmd.contains("skill") && cmd["skill"].is_string()) 
        {
            return cmd["skill"].get<std::string>();
        }
        
        for (auto it = cmd.begin(); it != cmd.end(); ++it) 
        {
            std::string key = it.key();
            if (key.find("skill") != std::string::npos && it.value().is_string()) 
            {
                std::string value = it.value().get<std::string>();
                RCLCPP_WARN(this->get_logger(), "Campo 'skill' incorreto: '%s', usando: '%s'", 
                           key.c_str(), value.c_str());
                return value;
            }
        }
        
        return std::nullopt;
    }

    std::optional<std::string> extract_coordinates(const json& params)
    {
        if (params.contains("x") && params.contains("y") && params.contains("z"))
        {
            std::stringstream ss;
            ss << params["x"].get<double>() << ";"
               << params["y"].get<double>() << ";"
               << params["z"].get<double>();
            return ss.str();
        }
        
        if (params.contains("id") && params["id"].is_string())
        {
            std::string id = params["id"].get<std::string>();
            
            if (id.find(';') != std::string::npos) {
                return id;
            }
            
            if (id.find('(') != std::string::npos)
            {
                std::string clean = id;
                clean.erase(std::remove(clean.begin(), clean.end(), '('), clean.end());
                clean.erase(std::remove(clean.begin(), clean.end(), ')'), clean.end());
                clean.erase(std::remove(clean.begin(), clean.end(), ' '), clean.end());
                std::replace(clean.begin(), clean.end(), ',', ';');
                return clean;
            }
        }
        
        return std::nullopt;
    }

    std::optional<std::string> extract_object_id(const json& params)
    {
        if (!params.contains("id") || !params["id"].is_string()) {
            return std::nullopt;
        }
        
        std::string id = params["id"].get<std::string>();
        
        if (id.find('(') != std::string::npos || 
            id.find(';') != std::string::npos ||
            (id.find(',') != std::string::npos && id.find_first_of("0123456789") != std::string::npos))
        {
            return std::nullopt;
        }
        
        return id;
    }

    std::vector<std::string> expand_range(const std::string& range_str)
    {
        std::vector<std::string> result;
        
        size_t colon = range_str.find(':');

        if (colon == std::string::npos) 
        {
            result.push_back(range_str);
            return result;
        }
        
        std::string start = range_str.substr(0, colon);
        std::string end = range_str.substr(colon + 1);
        
        size_t num_pos = start.find_last_not_of("0123456789");

        if (num_pos == std::string::npos || num_pos == start.length() - 1) 
        {
            result.push_back(range_str);
            return result;
        }
        
        std::string prefix = start.substr(0, num_pos + 1);
        int start_num = std::stoi(start.substr(num_pos + 1));
        int end_num = std::stoi(end.substr(num_pos + 1));
        int width = start.length() - num_pos - 1;
        
        for (int i = start_num; i <= end_num; i++) 
        {
            std::ostringstream oss;
            oss << prefix << std::setfill('0') << std::setw(width) << i;
            result.push_back(oss.str());
        }
        
        return result;
    }

    std::string join_list(const std::vector<std::string>& list)
    {
        std::string result;
        for (size_t i = 0; i < list.size(); i++) {
            if (i > 0) result += "|";
            result += list[i];
        }
        return result;
    }

    std::string build_bt_xml(const json& plan) 
    {
        std::stringstream ss;
        ss << "<root BTCPP_format=\"4\" main_tree_to_execute=\"MainPlan\">\n";
        ss << "  <BehaviorTree ID=\"MainPlan\">\n";
        ss << "    <Sequence name=\"LLM_Sequence\">\n";

        for (const auto& cmd : plan["commands"]) 
        {
            if (cmd.contains("loop") && cmd.contains("do"))
            {
                std::string loop_xml = build_loop(cmd);
                if (!loop_xml.empty()) {
                    ss << loop_xml;
                }
                continue;
            }
            
            auto skill_opt = extract_skill(cmd);

            if (!skill_opt) 
            {
                RCLCPP_ERROR(this->get_logger(), "Comando sem 'skill': %s", cmd.dump().c_str());
                continue;
            }
            
            if (!cmd.contains("params")) 
            {
                RCLCPP_ERROR(this->get_logger(), "Comando sem 'params': %s", cmd.dump().c_str());
                continue;
            }

            std::string subtree = build_subtree(skill_opt.value(), cmd["params"]);

            if (subtree.empty()) 
            {
                RCLCPP_ERROR(this->get_logger(), "Falha ao construir subtree: %s", cmd.dump().c_str());
                continue;
            }
            ss << subtree;
        }

        ss << "    </Sequence>\n";
        ss << "  </BehaviorTree>\n";
        ss << "</root>";
        return ss.str();
    }

    std::string build_loop(const json& cmd)
    {
        std::stringstream ss;
        
        const json& loop_def = cmd["loop"];
        const json& actions = cmd["do"];
        
        // Expande items
        std::vector<std::string> items;
        if (loop_def.contains("item")) 
        {
            if (loop_def["item"].is_string()) 
            {
                items = expand_range(loop_def["item"].get<std::string>());
            } 
            else if (loop_def["item"].is_array()) 
            {
                for (const auto& it : loop_def["item"]) 
                {
                    items.push_back(it.get<std::string>());
                }
            }
        }
        
        if (items.empty()) 
        {
            RCLCPP_ERROR(this->get_logger(), "Loop sem items válidos!");
            return "";
        }
        
        // Expande destinos
        std::vector<std::string> dests;
        if (loop_def.contains("dest")) 
        {
            if (loop_def["dest"].is_string()) 
            {
                dests = expand_range(loop_def["dest"].get<std::string>());
            } 
            else if (loop_def["dest"].is_array()) 
            {
                for (const auto& it : loop_def["dest"]) 
                {
                    dests.push_back(it.get<std::string>());
                }
            }
        }
        
        // Constrói ForEach com output ports
        ss << "      <ForEach items=\"" << join_list(items) << "\" ";
        
        if (!dests.empty()) {
            ss << "dests=\"" << join_list(dests) << "\" ";
        }
        
        // Output ports para item
        ss << "item=\"{loop_item}\" pose=\"{loop_pose}\" size=\"{loop_size}\" ";
        
        // Output ports para destino
        if (!dests.empty()) {
            ss << "dest=\"{loop_dest}\" dest_pose=\"{loop_dest_pose}\" dest_size=\"{loop_dest_size}\" ";
        }
        ss << ">\n";
        
        ss << "        <Sequence>\n";
        
        // Processa ações do loop
        for (const auto& action : actions)
        {
            auto skill_opt = extract_skill(action);
            if (!skill_opt || !action.contains("params")) continue;
            
            std::string skill = skill_opt.value();
            std::string skill_lower = skill;
            std::transform(skill_lower.begin(), skill_lower.end(), skill_lower.begin(), ::tolower);
            
            const json& params = action["params"];
            std::string id_str = params.contains("id") ? params["id"].get<std::string>() : "";
            
            if (skill_lower == "pick")
            {
                ss << "          <SubTree ID=\"Pick\" target_id=\"{loop_item}\" "
                   << "target_pose=\"{loop_pose}\" target_size=\"{loop_size}\" />\n";
            }
            else if (skill_lower == "place")
            {
                if (id_str == "$dest") 
                {
                    // Destino variável do loop - usa pose e size do DB
                    ss << "          <SubTree ID=\"Place\" storage_id=\"{loop_dest}\" "
                       << "storage_pose=\"{loop_dest_pose}\" storage_size=\"{loop_dest_size}\" />\n";
                } 
                else 
                {
                    // Destino fixo - server_node vai expandir via XML preprocessing
                    ss << "          <SubTree ID=\"Place\" storage_id=\"" << id_str << "\" />\n";
                }
            }
            else if (skill_lower == "goto_location" || skill_lower == "goto")
            {
                auto coords = extract_coordinates(params);

                if (coords) 
                {
                    ss << "          <SubTree ID=\"GoToLocation\" target=\"" << coords.value() << "\" />\n";
                } 
                else 
                {
                    ss << "          <SubTree ID=\"GoToLocation\" target_id=\"{loop_item}\" />\n";
                }
            }
        }
        
        ss << "        </Sequence>\n";
        ss << "      </ForEach>\n";
        
        return ss.str();
    }

    std::string build_subtree(const std::string& skill, const json& params) 
    {
        std::stringstream ss;
        
        std::string skill_lower = skill;
        std::transform(skill_lower.begin(), skill_lower.end(), skill_lower.begin(), ::tolower);

        if (skill_lower == "pick") 
        {
            auto obj_id = extract_object_id(params);
            if (!obj_id) {
                RCLCPP_ERROR(this->get_logger(), "Pick requer ID de objeto!");
                return "";
            }
            
            ss << "      <SubTree ID=\"Pick\" target_id=\"" << obj_id.value() << "\" />\n";
        }
        else if (skill_lower == "place") 
        {
            auto coords = extract_coordinates(params);
            auto obj_id = extract_object_id(params);
            
            if (coords) 
            {
                // Coordenadas diretas
                ss << "      <SubTree ID=\"Place\" target=\"" << coords.value() << "\" />\n";
            }
            else if (obj_id) 
            {
                // ID do storage - server_node vai expandir via XML preprocessing
                ss << "      <SubTree ID=\"Place\" storage_id=\"" << obj_id.value() << "\" />\n";
            }
            else 
            {
                RCLCPP_ERROR(this->get_logger(), "Place requer coordenadas ou ID!");
                return "";
            }
        }
        else if (skill_lower == "goto_location" || skill_lower == "goto") 
        {
            auto coords = extract_coordinates(params);
            auto obj_id = extract_object_id(params);
            
            if (coords) 
            {
                ss << "      <SubTree ID=\"GoToLocation\" target=\"" << coords.value() << "\" />\n";
            }
            else if (obj_id) 
            {
                ss << "      <SubTree ID=\"GoToLocation\" target_id=\"" << obj_id.value() << "\" />\n";
            }
            else 
            {
                RCLCPP_ERROR(this->get_logger(), "GoTo requer coordenadas ou ID!");
                return "";
            }
        }
        else 
        {
            RCLCPP_ERROR(this->get_logger(), "Skill desconhecida: %s", skill.c_str());
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