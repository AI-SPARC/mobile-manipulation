#include <manipulation/IsGripperHolding.hpp>
#include "rclcpp_components/register_node_macro.hpp"

using namespace std::chrono_literals;

namespace manipulation {

// DOC-START: IsGripperHolding
// Construtor: Inicializa o nó como um Componente ROS 2.
// Recebe 'NodeOptions' para permitir remapeamento de argumentos na composição.
IsGripperHolding::IsGripperHolding(const rclcpp::NodeOptions & options)
: Node("gripper_monitor_node", options) 
{
    RCLCPP_INFO(this->get_logger(), "Gripper Monitor Node inicializado (Composable).");

    // Assina o tópico do sensor de força/pressão.
    // O callback alimenta um buffer circular para filtragem de ruído.
    subscription_ = this->create_subscription<std_msgs::msg::Float32>(
        "contact_sensor", 10, 
        std::bind(&IsGripperHolding::topic_callback, this, std::placeholders::_1)
    );
}   
// DOC-END: IsGripperHolding

// DOC-START: checkIsHolding
// Função Pública: Verifica se o objeto está seguro na garra.
// Esta função é chamada DIRETAMENTE pelo ServerNode via ponteiro compartilhado,
// ignorando a pilha de comunicação do DDS (Zero-Copy / Zero-Latency).
bool IsGripperHolding::checkIsHolding() 
{
    // Protege o acesso ao deque compartilhado, pois o callback roda em outra thread
    std::lock_guard<std::mutex> lock(contact_sensor_mutex_);
    int contador = 0;

    if (contact_sensor_data_.empty()) 
    {
        return false;
    }

    // Lógica de Votação/Debounce:
    // Analisa o histórico recente (janela deslizante) para evitar falsos positivos
    // causados por picos momentâneos de ruído no sensor.
    for(size_t i = 0; i < contact_sensor_data_.size(); i++)
    {
        // Limiar empírico de pressão/força (0.1)
        if(contact_sensor_data_[i] > 0.1)
        {
            contador++;
        }
    }

    // Critério de Sucesso: Pelo menos 90% das leituras recentes indicam contato.
    if(contador >= 9)
    {
        return true;
    }
    
    return false;
}
// DOC-END: checkIsHolding

// DOC-START: topic_callback
// Callback do Sensor: Mantém o buffer circular atualizado.
void IsGripperHolding::topic_callback(const std_msgs::msg::Float32 & msg)
{
    std::lock_guard<std::mutex> lock(contact_sensor_mutex_); 

    // Mantém apenas as últimas 10 leituras (Janela Deslizante)
    if (contact_sensor_data_.size() >= 10) 
    {
        contact_sensor_data_.pop_front();
    }

    contact_sensor_data_.push_back(msg.data);
}
// DOC-END: topic_callback

} // namespace manipulation

// Macro para registrar o nó como um componente dinâmico.
// Permite que o 'ServerNode' carregue esta classe em tempo de execução.
RCLCPP_COMPONENTS_REGISTER_NODE(manipulation::IsGripperHolding)