#include <chrono>
#include <cmath>
#include <memory>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

using namespace std::chrono_literals;

class NoisySlamSimulator : public rclcpp::Node
{
public:
  NoisySlamSimulator()
  : Node("noisy_slam_simulator")
  {
    // 1. Inicializa o estado do Ground Truth (x, y, theta)
    x_gt_ = 0.0;
    y_gt_ = 0.0;
    theta_gt_ = 0.0;

    // 2. Inicializa covariância padrão (pequena incerteza inicial)
    // Usamos a diagonal: [x, y, z, roll, pitch, yaw]
    current_std_dev_x_ = 0.05; 
    current_std_dev_y_ = 0.05;
    current_std_dev_yaw_ = 0.01;

    // Subscribers
    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10, std::bind(&NoisySlamSimulator::cmd_vel_callback, this, std::placeholders::_1));

    // Este tópico recebe a covariância que você quer testar
    cov_input_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/input/covariance", 10, std::bind(&NoisySlamSimulator::covariance_callback, this, std::placeholders::_1));

    // Publishers
    ground_truth_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/ground_truth/odom", 10);
    noisy_slam_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/slam/odom_noisy", 10);

    // Broadcaster TF (Para ver no RViz)
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Timer (Loop de Simulação - 50Hz)
    timer_ = this->create_wall_timer(
      20ms, std::bind(&NoisySlamSimulator::simulation_step, this));

    RCLCPP_INFO(this->get_logger(), "Simulador de Ruído Iniciado. Envie Twist para /cmd_vel e Covariancia para /input/covariance");
  }

private:
  // Callback de Movimento (Simula a física)
  void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    current_vel_ = *msg;
  }

  // Callback Matemático: Atualiza o gerador de ruído baseado na Covariância recebida
  void covariance_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    // A matriz de covariância é Sigma = desvio_padrao^2
    // Logo, desvio_padrao = sqrt(Sigma)
    // A msg.covariance é um array de 36 floats (6x6)
    
    // Posição 0: Var(X)
    double var_x = msg->pose.covariance[0];
    // Posição 7: Var(Y)
    double var_y = msg->pose.covariance[7];
    // Posição 35: Var(Yaw)
    double var_yaw = msg->pose.covariance[35];

    // Atualizamos os desvios padrão para o gerador de ruído
    // Adicionamos 1e-9 para evitar raiz de zero ou negativo
    current_std_dev_x_ = std::sqrt(std::max(var_x, 1e-9));
    current_std_dev_y_ = std::sqrt(std::max(var_y, 1e-9));
    current_std_dev_yaw_ = std::sqrt(std::max(var_yaw, 1e-9));

    RCLCPP_INFO(this->get_logger(), "Nova Covariancia Recebida! Sigma X: %.3f, Sigma Y: %.3f", current_std_dev_x_, current_std_dev_y_);
  }

  void simulation_step()
  {
    // --- 1. Atualizar Ground Truth (Física Perfeita) ---
    double dt = 0.02; // 20ms
    double v = current_vel_.linear.x;
    double w = current_vel_.angular.z;

    x_gt_ += v * std::cos(theta_gt_) * dt;
    y_gt_ += v * std::sin(theta_gt_) * dt;
    theta_gt_ += w * dt;

    // --- 2. Gerar Ruído Probabilístico (Baseado na Covariância atual) ---
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Cria distribuições normais N(0, sigma)
    std::normal_distribution<> noise_x(0.0, current_std_dev_x_);
    std::normal_distribution<> noise_y(0.0, current_std_dev_y_);
    std::normal_distribution<> noise_yaw(0.0, current_std_dev_yaw_);

    double noisy_x = x_gt_ + noise_x(gen);
    double noisy_y = y_gt_ + noise_y(gen);
    double noisy_theta = theta_gt_ + noise_yaw(gen);

    // --- 3. Publicar Ground Truth (Verde no Rviz) ---
    publish_odometryAndTF(x_gt_, y_gt_, theta_gt_, "ground_truth_link", ground_truth_pub_, false);

    // --- 4. Publicar Estimativa Ruidosa (Vermelho no Rviz) ---
    // Aqui enviamos a covariância real para o Rviz desenhar a elipse
    publish_odometryAndTF(noisy_x, noisy_y, noisy_theta, "noisy_slam_link", noisy_slam_pub_, true);
  }

  void publish_odometryAndTF(double x, double y, double theta, std::string child_frame, rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub, bool include_covariance)
  {
    rclcpp::Time now = this->get_clock()->now();

    // Quaternion
    tf2::Quaternion q;
    q.setRPY(0, 0, theta);

    // 1. Publicar TF (Para visualizar o robô se movendo)
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = now;
    t.header.frame_id = "map";
    t.child_frame_id = child_frame;
    t.transform.translation.x = x;
    t.transform.translation.y = y;
    t.transform.translation.z = 0.0;
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(t);

    // 2. Publicar Odometria
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = now;
    odom.header.frame_id = "map";
    odom.child_frame_id = child_frame;
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();

    if (include_covariance) {
      // Preenche a covariância na mensagem para o RViz desenhar a elipse
      // Diagonal:
      odom.pose.covariance[0] = std::pow(current_std_dev_x_, 2);
      odom.pose.covariance[7] = std::pow(current_std_dev_y_, 2);
      odom.pose.covariance[35] = std::pow(current_std_dev_yaw_, 2);
    }

    pub->publish(odom);
  }

  // Variáveis
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr cov_input_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ground_truth_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr noisy_slam_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  geometry_msgs::msg::Twist current_vel_;
  
  // Estado Ground Truth
  double x_gt_, y_gt_, theta_gt_;

  // Desvios padrão atuais (derivados da covariância de entrada)
  double current_std_dev_x_;
  double current_std_dev_y_;
  double current_std_dev_yaw_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NoisySlamSimulator>());
  rclcpp::shutdown();
  return 0;
}