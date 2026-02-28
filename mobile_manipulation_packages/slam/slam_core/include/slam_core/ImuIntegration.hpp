#ifndef IMU_INTEGRATION_HPP
#define IMU_INTEGRATION_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>

// Inclusão obrigatória para os ponteiros do GTSAM
#include <boost/shared_ptr.hpp>

#include <mutex>
#include <memory>

namespace slam_core
{ 

class ImuIntegration : public rclcpp::Node
{
public:
    explicit ImuIntegration(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~ImuIntegration() override;
    
    // ATENÇÃO: Retorna boost::shared_ptr para o GTSAM aceitar
    boost::shared_ptr<gtsam::PreintegratedImuMeasurements> getAndResetPreintegratedMeasurements(
        const gtsam::imuBias::ConstantBias& current_bias);

private:
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // ROS 2 usa std::shared_ptr nativamente (escondido no typedef SharedPtr)
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    // GTSAM EXIGE boost::shared_ptr
    boost::shared_ptr<gtsam::PreintegratedImuMeasurements> imu_integrator_;
    boost::shared_ptr<gtsam::PreintegrationParams> imu_params_;

    // Variáveis padrão C++ (std)
    std::mutex imu_mutex_; 
    double last_imu_time_;
    bool is_first_imu_;
};

} // namespace slam_core

#endif // IMU_INTEGRATION_HPP