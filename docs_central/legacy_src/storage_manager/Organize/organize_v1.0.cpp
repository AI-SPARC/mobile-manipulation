#include <storage_manager/Organize.hpp> 
#include "rclcpp_components/register_node_macro.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath> 

namespace storage_manager
{

// DOC-START: OrganizeNode
// Construtor: Inicializa o nó como um Componente.
// Este nó é projetado para ser carregado pelo 'ServerNode' via composição,
// permitindo chamadas diretas de função sem latência de tópicos.
OrganizeNode::OrganizeNode(const rclcpp::NodeOptions & options)
: Node("organize_node", options)
{
    RCLCPP_INFO(this->get_logger(), "Organize Node iniciado (Lógica de Bin Packing).");
}
// DOC-END: OrganizeNode

// DOC-START: placeObjectInBox
// Algoritmo de Bin Packing (Empacotamento)
// Calcula a posição exata (Pose) onde um objeto deve ser colocado dentro de uma caixa/prateleira,
// baseando-se em uma grade 3D virtual.
std::pair<geometry_msgs::msg::Pose, std::vector<int>> OrganizeNode::placeObjectInBox(
    const geometry_msgs::msg::Pose &storage_pose,    // Pose central da caixa de armazenamento
    const geometry_msgs::msg::Vector3 &storage_size, // Tamanho total da caixa (L, A, P)
    const geometry_msgs::msg::Vector3 &object_size,  // Tamanho do objeto a ser guardado
    const float object_padding,                      // Espaço extra entre objetos
    const float z_lift_offset,                       // Altura para levantar ao colocar (evitar colisão com a borda)
    int idx_x, int idx_y, int idx_z)                 // Índices atuais da grade (coluna, linha, camada)
{
    // 1. Definição da Célula da Grade
    // O tamanho ocupado por cada objeto é seu tamanho físico + margem de segurança (padding)
    double cell_x = object_size.x + object_padding;
    double cell_y = object_size.y + object_padding;

    // Validação de segurança para evitar divisão por zero ou lógica inválida
    if (cell_x <= 0 || cell_y <= 0 || object_size.z <= 0) 
    {
         RCLCPP_ERROR(rclcpp::get_logger("organize_node"), "Dimensões inválidas detectadas.");
         geometry_msgs::msg::Pose failed_pose;
         return std::make_pair(failed_pose, std::vector<int>{-1, -1, -1});
    }

    // 2. Cálculo de Capacidade Máxima
    // Quantos objetos cabem em cada eixo (X, Y, Z) dentro do Storage
    int max_idx_x = std::floor(storage_size.x / cell_x);
    int max_idx_y = std::floor(storage_size.y / cell_y);
    
    int raw_max_z = std::floor(storage_size.z / object_size.z);
    int max_idx_z = (raw_max_z == 0) ? 1 : raw_max_z; // Garante pelo menos 1 camada
    
    // Verifica se os índices solicitados estouram a capacidade da caixa
    bool z_overflow = (idx_z >= max_idx_z);

    // Permite colocar objetos maiores que a caixa no eixo Z se for a primeira camada (ex: em cima de uma mesa)
    if (storage_size.z < object_size.z && idx_z == 0) 
    {
        z_overflow = false; 
    }

    if (idx_x >= max_idx_x || idx_y >= max_idx_y || z_overflow)
    {
        RCLCPP_WARN(rclcpp::get_logger("organize_node"), 
                    "Storage Cheio! Índices [%d, %d, %d] excedem limite [%d, %d, %d].",
                    idx_x, idx_y, idx_z, max_idx_x, max_idx_y, max_idx_z);
        
        geometry_msgs::msg::Pose failed_pose;
        return std::make_pair(failed_pose, std::vector<int>{-1, -1, -1});
    }

    // 3. Cálculo da Origem Relativa (Canto Inferior Esquerdo)
    // O ROS usa o centro do objeto como (0,0,0). Para criar uma grade, precisamos
    // começar do canto da caixa.
    // Start X = -Metade_Caixa + Metade_Objeto (para centralizar o primeiro objeto na célula)
    double start_x = - (storage_size.x / 2.0) + (cell_x / 2.0);
    double start_y = - (storage_size.y / 2.0) + (cell_y / 2.0);
    
    // No Z, assumimos que storage_pose.z é o centro da caixa ou a base. 
    // Ajustamos para garantir que o objeto fique apoiado.
    double start_z = (object_size.z / 2.0) + z_lift_offset;

    // 4. Cálculo da Posição Relativa
    // Posição = Origem + (Índice * Tamanho da Célula)
    double pos_x_rel = start_x + (idx_x * cell_x);
    double pos_y_rel = start_y + (idx_y * cell_y);
    double pos_z_rel = start_z + (idx_z * object_size.z); 

    // 5. Transformação de Coordenadas (Rotação)
    // Se a caixa estiver rotacionada no mundo, a grade interna também deve rotacionar.
    geometry_msgs::msg::Pose final_pose;
    
    // Recupera a orientação da caixa (Quaternion)
    tf2::Quaternion q_storage(
        storage_pose.orientation.x, 
        storage_pose.orientation.y, 
        storage_pose.orientation.z, 
        storage_pose.orientation.w
    );
    
    // Aplica a rotação da caixa ao vetor de posição relativa calculado acima
    tf2::Matrix3x3 m_storage(q_storage);
    tf2::Vector3 vec_rel(pos_x_rel, pos_y_rel, pos_z_rel);
    tf2::Vector3 vec_world = m_storage * vec_rel; 
    
    // Soma o vetor rotacionado à posição central da caixa no mundo
    final_pose.position.x = storage_pose.position.x + vec_world.x();
    final_pose.position.y = storage_pose.position.y + vec_world.y();
    final_pose.position.z = storage_pose.position.z + vec_world.z();

    // 6. Definição da Orientação Final do Objeto
    // Mantemos o objeto "em pé" (Roll/Pitch = 0), mas alinhamos o Yaw com a caixa.
    double r_temp, p_temp, yaw_storage;
    m_storage.getRPY(r_temp, p_temp, yaw_storage);

    tf2::Quaternion q_final;
    q_final.setRPY(0.0, 0.0, yaw_storage); 
    q_final.normalize();
    final_pose.orientation = tf2::toMsg(q_final);
    
    // 7. Cálculo dos Próximos Índices (Lógica de Incremento)
    // Preenche: Eixo X -> Depois Eixo Y -> Depois Eixo Z (Camadas)
    int next_x = idx_x + 1;
    int next_y = idx_y;
    int next_z = idx_z;

    // Se acabou a linha (X), vai para a próxima coluna (Y) e reseta X
    if (next_x >= max_idx_x) 
    {
        next_x = 0;
        next_y++;

        // Se acabou a camada (Y), vai para cima (Z) e reseta Y
        if (next_y >= max_idx_y)
        {
            next_y = 0;
            next_z++;
        }
    }

    std::vector<int> next_indexes = {next_x, next_y, next_z};

    RCLCPP_INFO(rclcpp::get_logger("organize_node"), 
        "Objeto posicionado em [%d, %d, %d]. Próxima vaga: [%d, %d, %d]", 
        idx_x, idx_y, idx_z, next_x, next_y, next_z);

    return std::make_pair(final_pose, next_indexes);
}
// DOC-END: placeObjectInBox

} // namespace storage_manager

// Registro de Componente: Permite carregamento dinâmico sem re-compilação do main
RCLCPP_COMPONENTS_REGISTER_NODE(storage_manager::OrganizeNode)