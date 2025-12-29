#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <llms/OllamaClient.hpp> // Inclui o arquivo acima

class BrainNode : public rclcpp::Node {
public:
    BrainNode() : Node("brain_node") {
        // Inicializa o cliente LLM
        llm_client_ = std::make_unique<OllamaClient>();

        // Subscriber: Ouve comandos de texto
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/human_command", 10, 
            std::bind(&BrainNode::handle_command, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "Nó Cérebro Iniciado. Conectado ao Ollama (phi35_leve).");
    }

private:
    std::unique_ptr<OllamaClient> llm_client_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;

    void handle_command(const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Processando comando: '%s'", msg->data.c_str());

        // 1. Chama a LLM (Bloqueante por ~1-2 segundos)
        json plan = llm_client_->infer(msg->data);

        if (plan.empty()) {
            RCLCPP_ERROR(this->get_logger(), "A LLM não retornou um plano válido.");
            return;
        }

        // 2. Mostra o JSON recebido (Aqui entraria seu BTBuilder)
        std::string json_dump = plan.dump(2); // Indentação de 2 espaços
        RCLCPP_INFO(this->get_logger(), "\n--- PLANO GERADO ---\n%s\n--------------------", json_dump.c_str());
        
        // TODO: Enviar esse JSON para o nó que monta a Behavior Tree
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BrainNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}