#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>

// Includes do PCL para gerar a nuvem de pontos
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

class DepthNoiseInjector : public rclcpp::Node
{
public:
    DepthNoiseInjector() : Node("depth_noise_injector_node"), info_received_(false)
    {
       
        this->declare_parameter<double>("baseline", 0.05);
        this->declare_parameter<double>("subpixel_error", 0.08);
        this->declare_parameter<bool>("publish_pointcloud", true);

        b_ = this->get_parameter("baseline").as_double();
        sigma_d_ = this->get_parameter("subpixel_error").as_double();
        publish_pointcloud_ = this->get_parameter("publish_pointcloud").as_bool();

        std::random_device rd;
        generator_ = std::mt19937(rd());

        rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();
        rclcpp::QoS info_qos(10);                         

       
        info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "camera_info_in", info_qos, std::bind(&DepthNoiseInjector::info_callback, this, std::placeholders::_1));

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "image_in", sensor_qos, std::bind(&DepthNoiseInjector::image_callback, this, std::placeholders::_1));

        
        info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info_out", info_qos);
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_out", sensor_qos);

        if (publish_pointcloud_) 
        {
            pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("noisy_cloud", sensor_qos);
            RCLCPP_INFO(this->get_logger(), "Publicacao de Nuvem de Pontos ATIVADA (SensorDataQoS).");
        }

        RCLCPP_INFO(this->get_logger(), "Injetor de Ruido Estereo Iniciado!");
    }

private:
    void info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg)
    {
        if (!info_received_) 
        {
            // Extrai a matriz intrínseca completa
            fx_ = msg->k[0]; 
            cx_ = msg->k[2];
            fy_ = msg->k[4];
            cy_ = msg->k[5];
            
            info_received_ = true;
            RCLCPP_INFO(this->get_logger(), "CameraInfo recebido! fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f", fx_, fy_, cx_, cy_);
        }

        info_pub_->publish(*msg);
    }

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
    {
        if (!info_received_) return;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Falha no cv_bridge: %s", e.what());
            return;
        }

        cv::Mat& img = cv_ptr->image;
        std::normal_distribution<float> standard_normal(0.0f, 1.0f);

        // Cria o contêiner da nuvem de pontos se estiver ativado
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

        if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) 
        {
            for (int y = 0; y < img.rows; ++y) 
            {
                uint16_t* row_ptr = img.ptr<uint16_t>(y);
                for (int x = 0; x < img.cols; ++x) 
                {
                    float z_m = row_ptr[x] / 1000.0f; 
                    if (z_m > 0.001f) 
                    {
                        // Injeta o ruído
                        float sigma_z = (z_m * z_m * sigma_d_) / (fx_ * b_);
                        float noise = standard_normal(generator_) * sigma_z;
                        float new_z_m = z_m + noise;
                        
                        if (new_z_m > 0.0f) {
                            row_ptr[x] = static_cast<uint16_t>(new_z_m * 1000.0f);
                            
                            // Adiciona o ponto 3D na nuvem
                            if (publish_pointcloud_) {
                                pcl::PointXYZ pt;
                                pt.z = new_z_m;
                                pt.x = (x - cx_) * new_z_m / fx_;
                                pt.y = (y - cy_) * new_z_m / fy_;
                                cloud->points.push_back(pt);
                            }
                        } else {
                            row_ptr[x] = 0;
                        }
                    }
                }
            }
        }
        else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) 
        {
            for (int y = 0; y < img.rows; ++y) 
            {
                float* row_ptr = img.ptr<float>(y);
                for (int x = 0; x < img.cols; ++x) 
                {
                    float z_m = row_ptr[x];
                    if (z_m > 0.001f) 
                    {
                        float sigma_z = (z_m * z_m * sigma_d_) / (fx_ * b_);
                        float noise = standard_normal(generator_) * sigma_z;
                        float new_z_m = z_m + noise;
                        
                        if (new_z_m > 0.0f) {
                            row_ptr[x] = new_z_m;
                            
                            if (publish_pointcloud_) {
                                pcl::PointXYZ pt;
                                pt.z = new_z_m;
                                pt.x = (x - cx_) * new_z_m / fx_;
                                pt.y = (y - cy_) * new_z_m / fy_;
                                cloud->points.push_back(pt);
                            }
                        } else {
                            row_ptr[x] = 0.0f;
                        }
                    }
                }
            }
        }
        
        image_pub_->publish(*cv_ptr->toImageMsg());

        // Se a publicação estiver ativada, converte a nuvem do PCL para ROS e publica
        if (publish_pointcloud_ && !cloud->points.empty()) 
        {
            cloud->width = cloud->points.size();
            cloud->height = 1;
            cloud->is_dense = false;

            sensor_msgs::msg::PointCloud2 pc_msg;
            pcl::toROSMsg(*cloud, pc_msg);
            
            // Garante que o timestamp e o frame_id sejam IGUAIS aos da imagem da câmera
            pc_msg.header = msg->header;
            pc_pub_->publish(pc_msg);
        }
    }

    bool info_received_;
    double fx_, fy_, cx_, cy_; // Parâmetros Intrínsecos
    double b_;
    double sigma_d_;
    bool publish_pointcloud_;
    std::mt19937 generator_; 

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_; // Novo publisher
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DepthNoiseInjector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}