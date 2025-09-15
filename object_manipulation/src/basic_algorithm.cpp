#include <string>
#include <random>
#include <algorithm>
#include <geometry_msgs/msg/point.hpp>
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>
#include <map>
#include <stack>
#include <unordered_map>
#include <optional>
#include <iostream>
#include <climits>
#include <iomanip>
#include <thread>
#include <queue>
#include <tuple>
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/path.hpp>
#include <cmath>
#include <cstring>
#include <utility> 
#include <iomanip>
#include <filesystem>
#include <barrier>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <future>
#include <barrier>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/odometry.hpp>                       
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cmath>
#include <cstring>
#include <utility> 
#include <iomanip>
#include <filesystem>
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include "vision_msgs/msg/detection3_d_array.hpp"

using namespace std::chrono_literals;



using namespace std::chrono_literals;

namespace std 
{
    template <>
    struct hash<std::tuple<float, float, float>> 
    {
        size_t operator()(const std::tuple<float, float, float>& t) const 
        {
            size_t h1 = hash<float>()(std::get<0>(t));
            size_t h2 = hash<float>()(std::get<1>(t));
            size_t h3 = hash<float>()(std::get<2>(t));
            
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

namespace std {
    template<>
    struct hash<std::tuple<std::pair<int, int>, bool>> {
        size_t operator()(const std::tuple<std::pair<int, int>, bool>& t) const {
            const auto& p = std::get<0>(t);
            bool b = std::get<1>(t);
            size_t h1 = std::hash<int>{}(p.first);
            size_t h2 = std::hash<int>{}(p.second);
            size_t h3 = std::hash<bool>{}(b);
            size_t seed = h1;
            seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
}

template <typename T1, typename T2>
struct pair_hash {
    std::size_t operator ()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);  
    }
};

template<typename T1, typename T2, typename T3>
std::ostream& operator<<(std::ostream& os, const std::tuple<T1, T2, T3>& t) {
    os << "(" << std::get<0>(t) << ", " 
       << std::get<1>(t) << ", " 
       << std::get<2>(t) << ")";
    return os;
}

struct TupleHash {
    std::size_t operator()(const std::tuple<float, float, float>& t) const {
        auto h1 = std::hash<float>{}(std::get<0>(t));
        auto h2 = std::hash<float>{}(std::get<1>(t));
        auto h3 = std::hash<float>{}(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct TupleEqual {
    bool operator()(const std::tuple<float,float,float>& a,
                    const std::tuple<float,float,float>& b) const noexcept {
        return std::get<0>(a) == std::get<0>(b) &&
               std::get<1>(a) == std::get<1>(b) &&
               std::get<2>(a) == std::get<2>(b);
    }
};


class BasicAlgorithm : public rclcpp::Node {

private:


    


    //Publishers.
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_rounded_points;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_pub;

    //Subscriptions.
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr depth_camera_subscription_1;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr depth_camera_subscription_2;
    rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr sub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr parameterTimer;    

    float resolution = 0.01;
    int decimals = 0;
    float x_min_lim, x_max_lim, y_min_lim, y_max_lim, z_min_lim, z_max_lim;

    std::vector<std::tuple<float, float, float>> points;
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    rclcpp::TimerBase::SharedPtr init_timer_;

  
    inline float round_to_multiple(float value, float multiple, int decimals) 
    {
        if (multiple == 0.0) return value; 
        
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0, decimals);
        result = std::round(result * factor) / factor;
        
        return result;
    }
    

    int count_decimals(float number) 
    {
      
        float fractional = std::fabs(number - std::floor(number));
        int decimals = 0;
        const float epsilon = 1e-9; 
    
  
        while (fractional > epsilon && decimals < 20) {
            fractional *= 10;
            fractional -= std::floor(fractional);
            decimals++;
        }
        return decimals;
    }


    void initMoveGroup() {
        try {
            move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "arm");  // deve existir no SRDF

            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface inicializado com sucesso.");
            init_timer_->cancel();  // para de tentar
        } catch (const std::exception &e) {
            RCLCPP_WARN(this->get_logger(), "Ainda não consegui inicializar MoveGroupInterface: %s", e.what());
        }

    }


    
    std::vector<double> getJointPositionsForPose(const geometry_msgs::msg::Pose &target_pose) 
    {
        if (!move_group_) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface não inicializado.");
            return {};
        }

        // Define a pose alvo
        move_group_->setPoseTarget(target_pose);

        // Planejamento
        move_group_->setMaxVelocityScalingFactor(1.0);
        move_group_->setMaxAccelerationScalingFactor(1.0);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        auto result = move_group_->plan(plan);

        if (result == moveit::core::MoveItErrorCode::SUCCESS) {
            // executa mais rápido
            move_group_->execute(plan);
        }

        // Pega as posições de juntas da última configuração da trajetória
        if (plan.trajectory.joint_trajectory.points.empty()) {
            RCLCPP_WARN(this->get_logger(), "Trajetória vazia retornada pelo planejador");
            return {};
        }

        return plan.trajectory.joint_trajectory.points.back().positions;
    }

    geometry_msgs::msg::Pose randomPose() 
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());

        std::uniform_real_distribution<double> pos_dist(-0.5, 0.5);   // intervalo posição
        std::uniform_real_distribution<double> ori_dist(-1.0, 1.0);   // intervalo orientação

        geometry_msgs::msg::Pose pose;
        pose.position.x = pos_dist(gen);
        pose.position.y = pos_dist(gen);
        pose.position.z = pos_dist(gen) + 0.6; // mantém z acima de 0.0

        double qx = ori_dist(gen);
        double qy = ori_dist(gen);
        double qz = ori_dist(gen);
        double qw = ori_dist(gen);

        double norm = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
        pose.orientation.x = qx / norm;
        pose.orientation.y = qy / norm;
        pose.orientation.z = qz / norm;
        pose.orientation.w = qw / norm;

        return pose;
    }


    /*

        PUBLISHERS.

    */



    void send_joint_positions()
    {
        // Pose e orientação desejadas do pegador (agora aleatória)
        geometry_msgs::msg::Pose target_pose = randomPose();

        // Obtem posições de juntas via MoveIt IK
        std::vector<double> positions = getJointPositionsForPose(target_pose);

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }


    void publish_created_vertices()
    {
        sensor_msgs::msg::PointCloud2 cloud_msgs_created_vertices;
        cloud_msgs_created_vertices.header.stamp = this->get_clock()->now();
        cloud_msgs_created_vertices.header.frame_id = "world";

        cloud_msgs_created_vertices.height = 1; 
        cloud_msgs_created_vertices.width = points.size(); 
        cloud_msgs_created_vertices.is_dense = true;
        cloud_msgs_created_vertices.is_bigendian = false;
        cloud_msgs_created_vertices.point_step = 3 * sizeof(float); 
        cloud_msgs_created_vertices.row_step = cloud_msgs_created_vertices.point_step * cloud_msgs_created_vertices.width;

        sensor_msgs::PointCloud2Modifier modifier(cloud_msgs_created_vertices);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(cloud_msgs_created_vertices.width);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msgs_created_vertices, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msgs_created_vertices, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msgs_created_vertices, "z");
        for (const auto& vertex : points) 
        {
           
                *iter_x = std::get<0>(vertex);
                *iter_y = std::get<1>(vertex);
                *iter_z = std::get<2>(vertex);

                ++iter_x;
                ++iter_y;
                ++iter_z;    
        }

        send_joint_positions();

        publisher_rounded_points->publish(cloud_msgs_created_vertices);
    }


  
    /*
    
        CALLBACKS.

    */

    void callback(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
    {
        for (const auto & detection : msg->detections) {
        // há uma lista de hypotheses, normalmente [0] é a principal
        if (!detection.results.empty() &&
            detection.results[0].hypothesis.class_id == "1")
        {
            const auto & size = detection.bbox.size;
            RCLCPP_INFO(this->get_logger(),
            "Class 1 -> size x: %.3f, y: %.3f, z: %.3f",
            size.x, size.y, size.z);
        }
        }
    }

 
            
    void callback_object_point_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        points.clear();

        // Limites da região de interesse

        float x_min = std::numeric_limits<float>::max();
        float x_max = std::numeric_limits<float>::lowest();
        float y_min = std::numeric_limits<float>::max();
        float y_max = std::numeric_limits<float>::lowest();
        float z_min = std::numeric_limits<float>::max();
        float z_max = std::numeric_limits<float>::lowest();

        bool found = false;

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;

            if (x >= x_min_lim && x <= x_max_lim && y >= y_min_lim && y <= y_max_lim && z >= z_min_lim && z <= z_max_lim)
            {
                found = true;

                if (x < x_min) x_min = x;
                if (x > x_max) x_max = x;
                if (y < y_min) y_min = y;
                if (y > y_max) y_max = y;
                if (z < z_min) z_min = z;
                if (z > z_max) z_max = z;
                
                std::tuple<float, float, float> index = std::make_tuple(x, y, z);
                
                points.push_back(index);
            }
        }

        if (found)
        {
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"),
                "ROI bounds: x[%f, %f], y[%f, %f], z[%f, %f]",
                x_min, x_max, y_min, y_max, z_min, z_max);
        }
        else
        {
            RCLCPP_WARN(rclcpp::get_logger("rclcpp"),
                "Nenhum ponto dentro da região de interesse.");
        }

        publish_created_vertices();
    }







public:
    BasicAlgorithm()
     : Node("basic_algorithm")
    {
        this->declare_parameter<double>("x_min", -2.0);
        this->declare_parameter<double>("x_max", 2.0);
        this->declare_parameter<double>("y_min", -2.0);
        this->declare_parameter<double>("y_max", 2.0);
        this->declare_parameter<double>("z_min", 0.0);
        this->declare_parameter<double>("z_max", 2.0);

        x_min_lim =  static_cast<float>(this->get_parameter("x_min").get_parameter_value().get<double>());
        x_max_lim =  static_cast<float>(this->get_parameter("x_max").get_parameter_value().get<double>());
        y_min_lim =  static_cast<float>(this->get_parameter("y_min").get_parameter_value().get<double>());
        y_max_lim =  static_cast<float>(this->get_parameter("y_max").get_parameter_value().get<double>());
        z_min_lim =  static_cast<float>(this->get_parameter("z_min").get_parameter_value().get<double>());
        z_max_lim =  static_cast<float>(this->get_parameter("z_max").get_parameter_value().get<double>());

  
        RCLCPP_INFO(this->get_logger(), "x_min is set to: %f", x_min_lim);
        RCLCPP_INFO(this->get_logger(), "x_max is set to: %f", x_max_lim);
        RCLCPP_INFO(this->get_logger(), "y_min is set to: %f", y_min_lim);
        RCLCPP_INFO(this->get_logger(), "y_max is set to: %f", y_max_lim);
        RCLCPP_INFO(this->get_logger(), "z_min is set to: %f", z_min_lim);
        RCLCPP_INFO(this->get_logger(), "z_max is set to: %f", z_max_lim);

        joint_trajectory_pub = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/joint_trajectory_controller/joint_trajectory", 10);

        decimals = count_decimals(resolution);

        publisher_rounded_points = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points", 10);

        depth_camera_subscription_1 = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/rounded_points", 10, std::bind(&BasicAlgorithm::callback_object_point_cloud, this, std::placeholders::_1));

        sub_ = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            "/boxes", 10,
            std::bind(&BasicAlgorithm::callback, this, std::placeholders::_1));

        
        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&BasicAlgorithm::initMoveGroup, this));
            

     
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }   
};


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("basic_algorithm"); // único

    
    rclcpp::spin(std::make_shared<BasicAlgorithm>());
    rclcpp::shutdown();
    return 0;
}