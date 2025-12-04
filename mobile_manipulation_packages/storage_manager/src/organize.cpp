#include <storage_manager/Organize.hpp> 
#include "rclcpp_components/register_node_macro.hpp"

namespace storage_manager
{

OrganizeNode::OrganizeNode(const rclcpp::NodeOptions & options)
: Node("organize_node", options)
{
    RCLCPP_INFO(this->get_logger(), "Organize Node iniciado.");
}

geometry_msgs::msg::Pose OrganizeNode::placeObjectInBox(
    const geometry_msgs::msg::Pose &storage_pose, 
    const geometry_msgs::msg::Vector3 &storage_size,
    const geometry_msgs::msg::Quaternion &object_orientation,
    const geometry_msgs::msg::Vector3 &object_size,
    int idx_x,
    int idx_y,
    int idx_z)
{
    double cell_x = object_size.x + config_.object_padding;
    double cell_y = object_size.y + config_.object_padding;
    double cell_z = object_size.z + config_.object_padding; 

    double start_x = - (storage_size.x / 2.0) + (cell_x / 2.0);
    double start_y = - (storage_size.y / 2.0) + (cell_y / 2.0);
    

    double start_z;
    if (storage_size.z < object_size.z) 
    {
        start_z = (object_size.z / 2.0) + config_.z_lift_offset;
    }
    else
    {
        start_z = - (storage_size.z / 2.0) + (object_size.z / 2.0) + config_.z_lift_offset;
    }


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

    tf2::Quaternion q_obj(object_orientation.x, object_orientation.y, object_orientation.z, object_orientation.w);
    double r, p, y_obj;
    tf2::Matrix3x3(q_obj).getRPY(r, p, y_obj);

    tf2::Quaternion q_final;
    q_final.setRPY(M_PI, 0.0, -y_obj); 
    q_final.normalize();
    final_pose.orientation = tf2::toMsg(q_final);

    return final_pose;
}

} // namespace storage_manager

RCLCPP_COMPONENTS_REGISTER_NODE(storage_manager::OrganizeNode)