#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <llms/OllamaClient.hpp> 

class BrainNode : public rclcpp::Node {
public:
    BrainNode() : Node("brain_node") {
        llm_client_ = std::make_unique<OllamaClient>();

        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/human_command", 10, 
            std::bind(&BrainNode::handle_command, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "Nó Cérebro Iniciado. Conectado ao Ollama (phi35_leve).");
    }

private:
    std::unique_ptr<OllamaClient> llm_client_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;

    void handle_command(const std_msgs::msg::String::SharedPtr msg) 
    {
        RCLCPP_INFO(this->get_logger(), "Processando comando: '%s'", msg->data.c_str());

        json plan = llm_client_->infer(msg->data);

        if (plan.empty()) {
            RCLCPP_ERROR(this->get_logger(), "A LLM não retornou um plano válido.");
            return;
        }

        std::string json_dump = plan.dump(2);
        RCLCPP_INFO(this->get_logger(), "\n--- PLANO GERADO ---\n%s\n--------------------", json_dump.c_str());
    
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BrainNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}