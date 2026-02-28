#include "slam_core/ImuIntegration.hpp"
#include <chrono>
#include "rclcpp_components/register_node_macro.hpp"

#include <boost/make_shared.hpp>

namespace slam_core
{

ImuIntegration::ImuIntegration(const rclcpp::NodeOptions & options)
: Node("imu_integration", options), last_imu_time_(-1.0), is_first_imu_(true)
{
    imu_params_ = gtsam::PreintegrationParams::MakeSharedU(9.81);
    
    double gyro_noise = 1e-5;

    double acc_noise = 100.0; 
    double integration_noise = 1.0; 

    imu_params_->accelerometerCovariance = gtsam::Matrix33::Identity() * std::pow(acc_noise, 2);
    imu_params_->gyroscopeCovariance     = gtsam::Matrix33::Identity() * std::pow(gyro_noise, 2);
    imu_params_->integrationCovariance   = gtsam::Matrix33::Identity() * std::pow(integration_noise, 2);

    gtsam::imuBias::ConstantBias prior_bias; 
    
    imu_integrator_ = boost::make_shared<gtsam::PreintegratedImuMeasurements>(imu_params_, prior_bias);

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/chassis/imu", 200, std::bind(&ImuIntegration::imu_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "No de Pre-Integracao da IMU (GTSAM) Iniciado!");
}

ImuIntegration::~ImuIntegration() {}

void ImuIntegration::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(imu_mutex_);

    double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    if (is_first_imu_)
    {
        last_imu_time_ = current_time;
        is_first_imu_ = false;
        return;
    }

    double dt = current_time - last_imu_time_;
    last_imu_time_ = current_time;

    if (dt <= 0.0) return;

    gtsam::Vector3 measured_acc(
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z);

    gtsam::Vector3 measured_omega(
        msg->angular_velocity.x,
        msg->angular_velocity.y,
        msg->angular_velocity.z);

    imu_integrator_->integrateMeasurement(measured_acc, measured_omega, dt);
}

// ATENÇÃO: Retorna boost::shared_ptr
boost::shared_ptr<gtsam::PreintegratedImuMeasurements> 
ImuIntegration::getAndResetPreintegratedMeasurements(const gtsam::imuBias::ConstantBias& current_bias)
{
    std::lock_guard<std::mutex> lock(imu_mutex_);

    // ATENÇÃO: Copiando com boost::make_shared
    auto preint_measurements_copy = boost::make_shared<gtsam::PreintegratedImuMeasurements>(*imu_integrator_);
    
    imu_integrator_->resetIntegrationAndSetBias(current_bias);

    return preint_measurements_copy;
}

} // namespace slam_core

RCLCPP_COMPONENTS_REGISTER_NODE(slam_core::ImuIntegration)