/**
 * @file obstacle_graph_with_occupancy_grid.cpp
 * @brief Nó responsável por carregar o mapa estático e gerar o grafo de obstáculos.
 * @details Este nó lê um arquivo YAML/PNG (formato ROS map_server), converte os pixels
 * ocupados em coordenadas de mundo, aplica uma inflação de segurança (padding) e publica
 * o resultado como uma PointCloud2 para ser consumida pelo A* ou D*.
 */

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

// DOC-START: PairHash
// Estrutura de Hash personalizada para permitir o uso de std::pair<float, float>
// como chave em unordered_map e unordered_set.
// Utiliza a combinação de bits (XOR e Shift) para gerar um hash único para coordenadas 2D.
struct PairHash {
    std::size_t operator()(const std::pair<float, float>& p) const noexcept {
        auto h1 = std::hash<float>{}(p.first);
        auto h2 = std::hash<float>{}(p.second);
        // Algoritmo de combinação de hash para reduzir colisões
        return h1 ^ (h2 << 1);
    }
};
// DOC-END: PairHash

class OccupancyGridLoader : public rclcpp::Node {
public:
    // DOC-START: Constructor
    OccupancyGridLoader()
        : rclcpp::Node("occupancy_grid_loader")
    {
        // Declaração de parâmetros:
        // map_yaml_file: Caminho para o arquivo de configuração do mapa.
        // map_image_file: Caminho para a imagem (PGM/PNG) do mapa.
        // max_security_distance: Raio de inflação dos obstáculos (padding de segurança).
        // obstacle_graph_resolution: Resolução da discretização (deve bater com a do A*).
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
        RCLCPP_INFO(this->get_logger(), "Safety Margin: %2f", maxSecurityDistance_);
        
        // Calcula casas decimais para evitar erros de ponto flutuante na chave do Hash
        decimals = countDecimals(distanceToObstacle_);

        // Carrega e processa o mapa imediatamente na inicialização
        if (!loadOccupancyGrid(yaml_file, image_file)) {
            RCLCPP_ERROR(this->get_logger(), "Falha ao carregar o occupancy grid!");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Total de pontos ocupados (inflados): %zu", occupied_points_.size());

        // Publisher para visualizar os obstáculos no RViz e alimentar o Planner
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/obstacles_vertices", 10);
        
        // Timer para publicar periodicamente a nuvem de pontos estática (Latched topic seria melhor, mas timer garante recebimento)
        timer_ = this->create_wall_timer(std::chrono::milliseconds(200),
                                         std::bind(&OccupancyGridLoader::publishPointCloud, this));
    }
    // DOC-END: Constructor

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Mapa auxiliar para marcar quais vértices originais já foram processados
    std::unordered_map<std::pair<float, float>, bool, PairHash> verticesCloudMap;

    // Conjunto final contendo todos os pontos bloqueados (originais + inflação)
    std::unordered_set<std::pair<float, float>, PairHash> occupied_points_;
    
    float maxSecurityDistance_, distanceToObstacle_;
    int decimals = 0;

    // DOC-START: Helpers
    // Conta casas decimais para ajuste de precisão
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

    // Função crítica: Discretiza coordenadas contínuas para o grid.
    // Ex: Se resolution = 0.05, 1.03 vira 1.05.
    // Isso garante que pontos próximos caiam na mesma chave do Hash.
    inline float round_to_multiple(float value, float multiple, int decimals) 
    {
        if (multiple == 0.0f) return value; 
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0f, decimals);
        result = std::round(result * factor) / factor;
        return result;
    }
    // DOC-END: Helpers

    // DOC-START: loadOccupancyGrid
    // Carrega o arquivo de Mapa (formato ROS Standard)
    bool loadOccupancyGrid(const std::string& yaml_path, const std::string& image_path) 
    {
        // 1. Lê metadados do YAML (Resolução, Origem)
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

        // 2. Lê a imagem usando OpenCV (Grayscale)
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

        if (image.empty()) 
        {
            std::cerr << "Erro ao carregar imagem: " << image_path << std::endl;
            return false;
        }

        int width = image.cols;
        int height = image.rows;

        // 3. Itera sobre cada pixel da imagem
        for (int y = 0; y < height; ++y) 
        {
            for (int x = 0; x < width; ++x) 
            {
                // No formato ROS map_server:
                // 0 = Livre (255 na imagem), 100 = Ocupado (0 na imagem)
                // Pixel < 50 é considerado obstáculo (Preto/Cinza escuro)
                unsigned char pixel = image.at<unsigned char>(y, x);

                if (pixel < 50) 
                {
                    // Converte índice de pixel para Coordenada de Mundo (Metros)
                    // Nota: Y na imagem cresce para baixo, no ROS cresce para cima (inversão)
                    float wx = static_cast<float>(origin_x + x * resolution);
                    float wy = static_cast<float>(origin_y + (height - y - 1) * resolution); 

                    // Arredonda para a resolução do nosso grafo de navegação
                    float rx = round_to_multiple(wx, 0.05, decimals);
                    float ry = round_to_multiple(wy, 0.05, decimals);

                    std::pair<float, float> index = std::make_pair(rx, ry);

                    // Insere no mapa de vértices brutos
                    if(verticesCloudMap.find(index) == verticesCloudMap.end())
                    {
                        verticesCloudMap[index] = false; // 'false' = ainda não inflado
                        occupied_points_.insert(index);
                    }
                }
            }
        }

        // 4. Aplica a inflação de obstáculos
        createGraphFromPointCloud();
        return true;
    }
    // DOC-END: loadOccupancyGrid

    // DOC-START: createGraphFromPointCloud
    // Algoritmo de Inflação de Obstáculos (Security Distance Expansion).
    // Para cada ponto de obstáculo real, cria camadas de "obstáculos virtuais" ao redor
    // até atingir a distância de segurança definida.
    void createGraphFromPointCloud() 
    {
        for(auto it = verticesCloudMap.begin(); it != verticesCloudMap.end(); it++)
        {
            // Se o ponto ainda não foi processado
            if(it->second == false)
            {
                it->second = true; // Marca como visitado
                
                float current_radius = 0.0;
                int layer_index = 0; // Índice da camada de expansão
              
                // Expande em camadas quadradas concêntricas
                while(current_radius <= maxSecurityDistance_)
                {
                    // Itera pelo perímetro do quadrado atual
                    for(int step = 0; step <= layer_index * 2; step++)
                    {   
                        // Calcula os 4 lados do quadrado de expansão
                        // Lado Direito/Cima
                        std::pair<float, float> p1 = std::make_pair(
                            round_to_multiple((std::get<0>(it->first) + current_radius) - (distanceToObstacle_ * step), distanceToObstacle_, decimals), 
                            round_to_multiple((std::get<1>(it->first) + current_radius), distanceToObstacle_, decimals)
                        );
                        
                        // Lado Direito/Baixo
                        std::pair<float, float> p2 = std::make_pair(
                            round_to_multiple((std::get<0>(it->first) + current_radius), distanceToObstacle_, decimals), 
                            round_to_multiple((std::get<1>(it->first) + current_radius) - (distanceToObstacle_ * step), distanceToObstacle_, decimals)
                        );
                        
                        // Lado Esquerdo/Baixo
                        std::pair<float, float> p3 = std::make_pair(
                            round_to_multiple((std::get<0>(it->first) - current_radius), distanceToObstacle_, decimals), 
                            round_to_multiple((std::get<1>(it->first) - current_radius) + (distanceToObstacle_ * step), distanceToObstacle_, decimals)
                        );

                        // Lado Esquerdo/Cima
                        std::pair<float, float> p4 = std::make_pair(
                            round_to_multiple((std::get<0>(it->first) - current_radius) + (distanceToObstacle_ * step), distanceToObstacle_, decimals), 
                            round_to_multiple((std::get<1>(it->first) - current_radius), distanceToObstacle_, decimals)
                        );
                        
                        // Adiciona os pontos calculados ao conjunto de bloqueio
                        occupied_points_.insert(p1);
                        occupied_points_.insert(p2);
                        occupied_points_.insert(p3);
                        occupied_points_.insert(p4);
                    }

                    layer_index++;
                    current_radius += distanceToObstacle_;
                }
            }
        }
    }
    // DOC-END: createGraphFromPointCloud

    // DOC-START: publishPointCloud
    // Converte o conjunto de pontos ocupados (Set) em uma mensagem ROS PointCloud2.
    // Esta mensagem é usada pelo A* para saber onde não pode pisar.
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
        
        // Define estrutura XYZ (float32)
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(cloud.width);

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

        // Preenche a mensagem
        for (const auto& p : occupied_points_) {
            *iter_x = p.first;
            *iter_y = p.second;
            *iter_z = 0.0f; // Mapa 2D, Z=0
            ++iter_x; ++iter_y; ++iter_z;
        }

        publisher_->publish(cloud);
    }
    // DOC-END: publishPointCloud
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OccupancyGridLoader>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}