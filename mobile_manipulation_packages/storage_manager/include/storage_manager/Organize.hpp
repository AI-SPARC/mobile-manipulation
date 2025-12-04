#ifndef STORAGE_MANAGER__ORGANIZE_HPP_
#define STORAGE_MANAGER__ORGANIZE_HPP_

#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "geometry_msgs/msg/quaternion.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <yaml-cpp/yaml.h>

namespace storage_manager
{

class OrganizeNode : public rclcpp::Node
{
public:
    explicit OrganizeNode(const rclcpp::NodeOptions & options);
    ~OrganizeNode() override = default;

    /**
     * @brief Calcula a pose no mundo para um índice de grade específico dentro do storage.
     * * @param storage_pose Pose do centro da caixa/storage.
     * @param storage_size Dimensões da caixa.
     * @param object_orientation Orientação original do objeto.
     * @param object_size Dimensões do objeto.
     * @param idx_x Índice da coluna (0 a N).
     * @param idx_y Índice da linha (0 a M).
     * @param idx_z Índice da camada (0 a K).
     * @return geometry_msgs::msg::Pose A pose calculada.
     */
    geometry_msgs::msg::Pose placeObjectInBox(
        const geometry_msgs::msg::Pose &storage_pose, 
        const geometry_msgs::msg::Vector3 &storage_size,
        const geometry_msgs::msg::Quaternion &object_orientation,
        const geometry_msgs::msg::Vector3 &object_size,
        int idx_x,
        int idx_y,
        int idx_z);

private:
    struct Config
    {
        double object_padding = 0.02; 
        double z_lift_offset = 0.01; 
    } config_;

    std::string yaml_file_;
};

} // namespace storage_manager

#endif // STORAGE_MANAGER__ORGANIZE_HPP_