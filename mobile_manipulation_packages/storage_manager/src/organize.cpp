#include <storage_manager/Organize.hpp> 
#include "rclcpp_components/register_node_macro.hpp"

// Incluir geometria para Vector3
#include "geometry_msgs/msg/vector3.hpp"
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath> // Para std::floor e M_PI

namespace storage_manager
{

OrganizeNode::OrganizeNode(const rclcpp::NodeOptions & options)
: Node("organize_node", options)
{
    RCLCPP_INFO(this->get_logger(), "Organize Node iniciado.");
}

std::pair<geometry_msgs::msg::Pose, std::vector<int>> OrganizeNode::placeObjectInBox(
    const geometry_msgs::msg::Pose &storage_pose, 
    const geometry_msgs::msg::Vector3 &storage_size,
    const geometry_msgs::msg::Vector3 &object_size,
    const float object_padding,
    const float z_lift_offset,
    int idx_x,
    int idx_y,
    int idx_z)
{
    double cell_x = object_size.x + object_padding;
    double cell_y = object_size.y + object_padding;
    

    if (cell_x <= 0 || cell_y <= 0 || object_size.z <= 0) 
    {
         RCLCPP_ERROR(rclcpp::get_logger("organize_node"), "Tamanho do objeto ou padding inválido (<= 0).");
         geometry_msgs::msg::Pose failed_pose;
         return std::make_pair(failed_pose, std::vector<int>{-1, -1, -1});
    }

    int max_idx_x = std::floor(storage_size.x / cell_x);
    int max_idx_y = std::floor(storage_size.y / cell_y);
    int max_idx_z = std::floor(storage_size.z / object_size.z);
    
    if (idx_x >= max_idx_x || idx_y >= max_idx_y || idx_z >= max_idx_z)
    {
        RCLCPP_WARN(rclcpp::get_logger("organize_node"), 
                    "Índices [%d, %d, %d] excedem o número total de células [%d, %d, %d]. Falha no Organize.",
                    idx_x, idx_y, idx_z, max_idx_x, max_idx_y, max_idx_z);
        
        geometry_msgs::msg::Pose failed_pose;
        return std::make_pair(failed_pose, std::vector<int>{-1, -1, -1});
    }


  
    double start_x = - (storage_size.x / 2.0) + (cell_x / 2.0);
    double start_y = - (storage_size.y / 2.0) + (cell_y / 2.0);
    
    double start_z;
    start_z = - (storage_size.z / 2.0) + (object_size.z / 2.0) + z_lift_offset;

    double pos_x_rel = start_x + (idx_x * cell_x);
    double pos_y_rel = start_y + (idx_y * cell_y);
    double pos_z_rel = start_z + (idx_z * object_size.z); 

    geometry_msgs::msg::Pose final_pose;
    
    
    tf2::Quaternion q_storage(
        storage_pose.orientation.x, 
        storage_pose.orientation.y, 
        storage_pose.orientation.z, 
        storage_pose.orientation.w
    );
    tf2::Matrix3x3 m_storage(q_storage);
    
    tf2::Vector3 vec_rel(pos_x_rel, pos_y_rel, pos_z_rel);
    tf2::Vector3 vec_world = m_storage * vec_rel; 
    
    final_pose.position.x = storage_pose.position.x + vec_world.x();
    final_pose.position.y = storage_pose.position.y + vec_world.y();
    final_pose.position.z = storage_pose.position.z + vec_world.z();


    double roll_box, pitch_box, yaw_box;
    m_storage.getRPY(roll_box, pitch_box, yaw_box);

    tf2::Quaternion q_final;

    q_final.setRPY(M_PI, 0.0, yaw_box); 
    
    q_final.normalize();
    final_pose.orientation = tf2::toMsg(q_final);
    

    std::vector<int> current_indexes = {idx_x, idx_y, idx_z};

    return std::make_pair(final_pose, current_indexes);
}

} // namespace storage_manager

RCLCPP_COMPONENTS_REGISTER_NODE(storage_manager::OrganizeNode)