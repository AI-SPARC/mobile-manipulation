#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <cmath>

struct PairHash {
    std::size_t operator()(const std::pair<float, float>& p) const noexcept {
        auto h1 = std::hash<float>{}(p.first);
        auto h2 = std::hash<float>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class OccupancyGridLoader : public rclcpp::Node {
public:
    OccupancyGridLoader()
        : rclcpp::Node("occupancy_grid_loader")
    {
        this->declare_parameter<std::string>("map_yaml_file", "map.yaml");
        this->declare_parameter<std::string>("map_image_file", "occupancy_grid.png");
        this->declare_parameter<double>("max_security_distance", 0.30);
        this->declare_parameter<double>("obstacle_graph_resolution", 0.05);

        std::string yaml_file = this->get_parameter("map_yaml_file").as_string();
        std::string image_file = this->get_parameter("map_image_file").as_string();
        maxSecurityDistance_ = static_cast<float>(this->get_parameter("max_security_distance").get_parameter_value().get<double>());
        distanceToObstacle_ = static_cast<float>(this->get_parameter("obstacle_graph_resolution").get_parameter_value().get<double>());

        RCLCPP_INFO(this->get_logger(), "YAML file: %s", yaml_file.c_str());
        RCLCPP_INFO(this->get_logger(), "Image file: %s", image_file.c_str());
        RCLCPP_INFO(this->get_logger(), "max_security_distance is set to %2f", maxSecurityDistance_);
        decimals = countDecimals(distanceToObstacle_);

        if (!loadOccupancyGrid(yaml_file, image_file)) {
            RCLCPP_ERROR(this->get_logger(), "Falha ao carregar o occupancy grid!");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Total de pontos ocupados carregados: %zu", occupied_points_.size());

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/obstacles_vertices", 10);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(500),
                                         std::bind(&OccupancyGridLoader::publishPointCloud, this));

        
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unordered_map<std::pair<float, float>, bool, PairHash> verticesCloudMap;

    std::unordered_set<std::pair<float, float>, PairHash> occupied_points_;
    float maxSecurityDistance_, distanceToObstacle_;
    int decimals = 0;

    int countDecimals(float number) 
    {
        float fractional = std::fabs(number - std::floor(number));
        int decimals = 0;
        const float epsilon = 1e-9; 
    
        while (fractional > epsilon && decimals < 20) 
        {
            fractional *= 10;
            fractional -= std::floor(fractional);
            decimals++;
        }

        return decimals;
    }

    inline float round_to_multiple(float value, float multiple, int decimals) 
    {
        if (multiple == 0.0f) return value; 
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0f, decimals);
        result = std::round(result * factor) / factor;
        return result;
    }

    bool loadOccupancyGrid(const std::string& yaml_path, const std::string& image_path) 
    {
        YAML::Node config = YAML::LoadFile(yaml_path);

        if (!config["resolution"] || !config["origin"]) 
        {
            std::cerr << "YAML inválido, precisa ter 'resolution' e 'origin'." << std::endl;
            return false;
        }

        double resolution = config["resolution"].as<double>();
        std::vector<double> origin = config["origin"].as<std::vector<double>>();
        double origin_x = origin[0];
        double origin_y = origin[1];

        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

        if (image.empty()) 
        {
            std::cerr << "Erro ao carregar imagem: " << image_path << std::endl;
            return false;
        }

        int width = image.cols;
        int height = image.rows;

        for (int y = 0; y < height; ++y) 
        {
            for (int x = 0; x < width; ++x) 
            {
                unsigned char pixel = image.at<unsigned char>(y, x);

                if (pixel < 50) 
                {
                    float wx = static_cast<float>(origin_x + x * resolution);
                    float wy = static_cast<float>(origin_y + (height - y - 1) * resolution); 

                    float x = round_to_multiple(wx, 0.05, decimals);
                    float y = round_to_multiple(wy, 0.05, decimals);

                    std::pair<float, float> index = std::make_pair(x, y);

                    if(verticesCloudMap.find(index) == verticesCloudMap.end())
                    {
                        verticesCloudMap[index] = false;
                        occupied_points_.insert(index);
                    }
                }
            }
        }

        createGraphFromPointCloud();
        return true;
    }

    void createGraphFromPointCloud() 
    {
        
        for(auto it = verticesCloudMap.begin(); it != verticesCloudMap.end(); it++)
        {
            if(it->second == false)
            {
                it->second = true;
                
                float toma = 0.0;
                int opa = 0;
              

                while(toma <= maxSecurityDistance_)
                {
                    for(int eita = 0; eita <= opa * 2; eita++)
                    {   
                        
                        std::pair<float, float> index10 = std::make_pair(round_to_multiple((std::get<0>(it->first) + toma) - (distanceToObstacle_ * eita), distanceToObstacle_, decimals), round_to_multiple((std::get<1>(it->first) + toma), distanceToObstacle_, decimals));
                        std::pair<float, float> index11 = std::make_pair(round_to_multiple((std::get<0>(it->first) + toma), distanceToObstacle_, decimals), round_to_multiple((std::get<1>(it->first) + toma) - (distanceToObstacle_ * eita), distanceToObstacle_, decimals));
                        
                        std::pair<float, float> index12 = std::make_pair(round_to_multiple((std::get<0>(it->first) - toma), distanceToObstacle_, decimals), round_to_multiple((std::get<1>(it->first) - toma) + (distanceToObstacle_ * eita), distanceToObstacle_, decimals));
                        std::pair<float, float> index13 = std::make_pair(round_to_multiple((std::get<0>(it->first) - toma) + (distanceToObstacle_ * eita), distanceToObstacle_, decimals), round_to_multiple((std::get<1>(it->first) - toma), distanceToObstacle_, decimals));
                        
                        
                        occupied_points_.insert(index10);
                        occupied_points_.insert(index11);
                        occupied_points_.insert(index12);
                        occupied_points_.insert(index13);
                    }

                    opa++;

                    toma += distanceToObstacle_;
                }
              
            }
            
        }

 
       
    }

   
    void publishPointCloud() 
    {
        if (occupied_points_.empty()) return;

      
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.stamp = this->get_clock()->now();
        cloud.header.frame_id = "world";

        cloud.height = 1;
        cloud.width = occupied_points_.size();
        cloud.is_dense = true;
        cloud.is_bigendian = false;
        cloud.point_step = 3 * sizeof(float);
        cloud.row_step = cloud.point_step * cloud.width;

        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(cloud.width);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

        for (const auto& p : occupied_points_) {
            *iter_x = p.first;
            *iter_y = p.second;
            *iter_z = 0.0f;
            ++iter_x; ++iter_y; ++iter_z;
        }

        

       
        publisher_->publish(cloud);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OccupancyGridLoader>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}