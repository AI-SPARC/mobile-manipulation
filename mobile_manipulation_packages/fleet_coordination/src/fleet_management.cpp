// Inclusão de bibliotecas padrão do C++ para gerenciamento de memória, vetores, strings e matemática
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <set>
#include <algorithm>

// Inclusão das bibliotecas do ROS 2
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "mobile_manipulation_interfaces/msg/fleet_paths.hpp"

// Inclusão das bibliotecas de Transformação Geométrica (TF2)
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// Facilita o uso de placeholders para callbacks (ex: _1)
using std::placeholders::_1;
// Alias para um par de floats, representando um ponto 2D (x, y) na nuvem
using Point2D = std::pair<float, float>;

// --- CONFIGURAÇÕES E CONSTANTES GLOBAIS ---
constexpr double ROBOT_WIDTH = 0.60;       // Largura física do robô em metros (bounding box)
constexpr double LONGITUDINAL_PAD = 0.30;  // Espaço extra na frente/trás para os retângulos estáticos
constexpr double TIME_STEP = 0.02;         // Passo de tempo da simulação física (50Hz)
constexpr double ANIMATION_FREQ = 0.05;    // Frequência de atualização visual no RViz (20Hz)
constexpr double CLOUD_RES = 0.05;         // Resolução da nuvem de pontos (distância entre pontos em metros)
constexpr double SIM_LIMIT = 30.0;         // Tempo máximo que a simulação futura irá verificar (30 segundos)
constexpr float SECURITY_DISTANCE = 0.3f;  // Distância de segurança para expandir a nuvem de colisão
constexpr double APPROACH_DISTANCE = 0.4;  // Distância do robô até a colisão para começar a desacelerar (SEU VALOR)
constexpr double SAFETY_TIME_MARGIN = 0.5; // Tempo extra de folga para garantir que o outro robô passou (SEU VALOR)

// --- ESTRUTURAS DE DADOS ---

// Estrutura para matemática vetorial 2D simples
struct Vec2 
{
    double x, y; // Coordenadas
    // Sobrecarga do operador de subtração para calcular vetores entre pontos
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    // Função para produto escalar (dot product), usado no algoritmo SAT
    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
};

// Um retângulo é definido como um vetor de vértices 2D
using Rectangle = std::vector<Vec2>;

// Estado instantâneo do robô em um determinado tempo
struct RobotState 
{
    geometry_msgs::msg::Pose pose; // Posição e Orientação (ROS msg)
    int segment_idx = -1;          // Índice do segmento do caminho onde ele está
    double yaw = 0.0;              // Ângulo de rotação em radianos
};

// Dados para controle de velocidade e mitigação de colisão
struct MitigationData 
{
    bool active = false;              // Indica se este robô precisa ajustar velocidade
    double safe_speed = 0.0;          // A nova velocidade calculada para evitar batida
    double conflict_entry_dist = 0.0; // A distância percorrida onde começa o perigo
    
    // Dados para monitorar o robô parceiro (com quem ele vai bater)
    int partner_id = -1;              // ID do robô conflitante
    double partner_exit_dist = 0.0;   // A distância onde o parceiro sai da zona de perigo
};

// Dados completos de um robô na frota
struct RobotData 
{
    int id;                           // Identificador numérico
    std::string frame_id;             // Frame de referência (ex: "map")
    double base_speed = 0.5;          // Velocidade nominal/máxima do robô
    
    // Variáveis para a simulação de movimento independente na animação
    double current_dist = 0.0;        // Quanto o robô já andou na animação atual
    double current_speed = 0.0;       // Velocidade atual (pode ser menor que base_speed se estiver freando)
    
    nav_msgs::msg::Path path;         // O caminho completo recebido
    std::vector<Rectangle> static_rects; // Retângulos pré-calculados ao redor do caminho (Broad Phase)
    std::vector<bool> risky_segments;    // Marca quais segmentos cruzam com segmentos de outros robôs
    std::vector<double> distances;       // Cache de distâncias acumuladas para interpolação rápida
    double total_length = 0.0;           // Comprimento total do caminho em metros

    MitigationData mitigation;        // Estrutura de mitigação aninhada
};

// Informações sobre uma colisão detectada
struct RobotCollisionInfo 
{
    std::vector<Point2D> points;      // Pontos (x,y) que formam a nuvem de colisão
    std::vector<int> colliding_with;  // Lista de IDs dos robôs com quem colide
    double start_time = -1.0;         // Tempo da simulação onde a colisão começa
    double end_time = -1.0;           // Tempo da simulação onde a colisão termina
    
    // Mapas para guardar as distâncias exatas de entrada e saída da zona de colisão
    std::unordered_map<int, double> entry_distances;
    std::unordered_map<int, double> exit_distances;
};

// Estrutura auxiliar para cor (RGB)
struct RGB 
{
    uint8_t r, g, b; // Valores de 0 a 255
};

// --- CLASSE PRINCIPAL DO NÓ ---
class FleetManagement : public rclcpp::Node 
{
public:
    // Construtor do Nó
    FleetManagement() : Node("fleet_traffic_manager") 
    {
        // Cria o subscriber para receber os planos de caminho da frota
        sub_paths_ = create_subscription<mobile_manipulation_interfaces::msg::FleetPaths>(
            "/fleet/all_robot_plans", 10, std::bind(&FleetManagement::on_paths, this, _1));
        
        // Cria o publisher para os marcadores visuais (robôs, textos, linhas)
        pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/debug_markers", 10);
        
        // Cria um timer que chama a função 'animate' repetidamente (loop de visualização)
        timer_ = create_wall_timer(
            std::chrono::duration<double>(ANIMATION_FREQ),
            std::bind(&FleetManagement::animate, this));
        
        // Log informativo indicando que o nó iniciou
        RCLCPP_INFO(get_logger(), "Fleet Manager: Sistema Iniciado com Cálculo de Velocidade Corrigido.");
    }

private:
    // --- VARIÁVEIS DE ESTADO ---
    std::vector<RobotData> fleet_; // Vetor contendo todos os dados dos robôs
    
    // Mapa de colisões: Chave = ID do Robô, Valor = Info da Colisão
    std::unordered_map<int, RobotCollisionInfo> collision_data_;
    
    // Mapa de publishers de PointCloud: Um tópico separado por robô
    std::unordered_map<int, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> cloud_publishers_;
    
    // Mapa de cores fixas para cada robô
    std::unordered_map<int, RGB> robot_colors_;
    
    // Ponteiros para comunicação ROS
    rclcpp::Subscription<mobile_manipulation_interfaces::msg::FleetPaths>::SharedPtr sub_paths_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ==================================================================================
    // SEÇÃO 1: LÓGICA, FÍSICA E CÁLCULOS
    // ==================================================================================

    // Helper: Retorna ponteiro para os dados de um robô dado seu ID
    RobotData* get_robot_by_id(int id) {
        for (auto& r : fleet_) {
            if (r.id == id) return &r; // Retorna se encontrar
        }
        return nullptr; // Retorna nulo se não existir
    }

    // Helper: Arredonda um float para o múltiplo mais próximo (usado na geração de grid da PointCloud)
    inline float round_to_multiple(float value, float multiple, int decimals) 
    {
        if (multiple == 0.0f) return value; // Evita divisão por zero
        float result = std::round(value / multiple) * multiple; // Arredonda
        float factor = std::pow(10.0f, decimals); // Fator de casas decimais
        return std::round(result * factor) / factor; // Trunca casas decimais extras
    }

    // Cria um retângulo orientado (OBB) ao redor de um segmento de reta entre p1 e p2
    Rectangle make_static_rect(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2) 
    {
        double dx = p2.x - p1.x; // Diferença em X
        double dy = p2.y - p1.y; // Diferença em Y
        double len = std::hypot(dx, dy); // Comprimento do segmento
        if (len < 0.001) return {}; // Retorna vazio se pontos forem iguais
        
        Vec2 u = {dx/len, dy/len}; // Vetor unitário diretor (direção do caminho)
        Vec2 n = {u.y, -u.x};      // Vetor unitário normal (perpendicular ao caminho)
        
        // Calcula pontos expandidos com preenchimento longitudinal (frente/trás)
        Vec2 s = {p1.x - u.x*LONGITUDINAL_PAD, p1.y - u.y*LONGITUDINAL_PAD};
        Vec2 e = {p2.x + u.x*LONGITUDINAL_PAD, p2.y + u.y*LONGITUDINAL_PAD};
        double hw = ROBOT_WIDTH / 2.0; // Meia largura do robô
        
        // Retorna os 4 cantos do retângulo
        return {{e.x+n.x*hw, e.y+n.y*hw}, {e.x-n.x*hw, e.y-n.y*hw},
                {s.x-n.x*hw, s.y-n.y*hw}, {s.x+n.x*hw, s.y+n.y*hw}};
    }

    // Cria a Bounding Box do robô baseada na sua posição e rotação atual
    Rectangle make_dynamic_box(const geometry_msgs::msg::Pose& pose) 
    {
        // Converte Quaternion (ROS) para Ângulos de Euler para pegar o Yaw (rotação Z)
        tf2::Quaternion q(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
        double r, p, yaw; tf2::Matrix3x3(q).getRPY(r, p, yaw);
        
        double hw = ROBOT_WIDTH / 2.0; // Meia largura
        double c = std::cos(yaw);      // Cosseno do ângulo
        double s = std::sin(yaw);      // Seno do ângulo
        Rectangle rect;
        
        // Vetor de cantos relativos (quadrado na origem)
        std::vector<std::pair<double,double>> corners = {{hw,hw},{hw,-hw},{-hw,-hw},{-hw,hw}};
        
        // Aplica rotação e translação para cada canto
        for (auto& [lx, ly] : corners)
            rect.push_back({pose.position.x + lx*c - ly*s, pose.position.y + lx*s + ly*c});
            
        return rect; // Retorna retângulo rotacionado no mapa
    }

    // Verifica se um ponto (px, py) está dentro de um robô (matemática de rotação inversa)
    bool point_in_rect(double px, double py, const geometry_msgs::msg::Pose& pose, double yaw) 
    {
        double dx = px - pose.position.x; // Delta X
        double dy = py - pose.position.y; // Delta Y
        double c = std::cos(-yaw);        // Cosseno da rotação inversa
        double s = std::sin(-yaw);        // Seno da rotação inversa
        
        // Rotaciona o ponto para o referencial local do robô
        double lx = dx*c - dy*s;
        double ly = dx*s + dy*c;
        
        double hw = ROBOT_WIDTH / 2.0;
        // Verifica se está dentro da caixa alinhada aos eixos (AABB) local
        return std::abs(lx) <= hw && std::abs(ly) <= hw;
    }

    // Projeta um polígono em um eixo para o algoritmo SAT
    void project(const Vec2& axis, const Rectangle& poly, double& min, double& max) 
    {
        min = std::numeric_limits<double>::infinity(); 
        max = -std::numeric_limits<double>::infinity();
        for (auto& p : poly) {
            double proj = p.dot(axis); // Produto escalar = projeção
            min = std::min(min, proj); // Guarda mínimo
            max = std::max(max, proj); // Guarda máximo
        }
    }

    // Algoritmo SAT (Separating Axis Theorem) para detecção de colisão convexa
    bool sat_intersect(const Rectangle& r1, const Rectangle& r2) 
    {
        // Testa eixos normais de ambos os retângulos
        for (auto* poly : {&r1, &r2}) {
            for (size_t i = 0; i < poly->size(); ++i) {
                // Pega uma aresta
                Vec2 edge = (*poly)[(i+1)%poly->size()] - (*poly)[i];
                // Calcula a normal (perpendicular) à aresta
                Vec2 normal = {-edge.y, edge.x};
                
                double min1, max1, min2, max2;
                // Projeta ambos os retângulos nessa normal
                project(normal, r1, min1, max1);
                project(normal, r2, min2, max2);
                
                // Se houver separação (gap) entre as projeções, não há colisão
                if (max1 < min2 || max2 < min1) return false;
            }
        }
        return true; // Se não achou separação em nenhum eixo, colidem
    }

    // Calcula estado do robô baseado no tempo (usado na pré-simulação)
    RobotState get_state_by_time(const RobotData& robot, double t) 
    {
        double target_dist = t * robot.base_speed; // d = v * t
        return get_state_by_distance(robot, target_dist);
    }

    // Calcula a Pose exata do robô interpolando no caminho dada uma distância percorrida
    RobotState get_state_by_distance(const RobotData& robot, double dist) 
    {
        RobotState state;
        
        // Se a distância pedida for maior que o caminho, retorna o final
        if (dist >= robot.total_length) {
            if (!robot.path.poses.empty()) {
                state.pose = robot.path.poses.back().pose;
                state.pose.position.z = 0.15; // Altura visual
                state.segment_idx = robot.static_rects.size() - 1;
                // Calcula orientação final baseada nos ultimos 2 pontos
                if (robot.path.poses.size() >= 2) {
                    auto& p1 = robot.path.poses[robot.path.poses.size()-2].pose.position;
                    auto& p2 = robot.path.poses.back().pose.position;
                    state.yaw = std::atan2(p2.y - p1.y, p2.x - p1.x);
                }
            }
            return state;
        }
        
        // Busca em qual segmento do caminho a distância cai
        for (size_t i = 0; i < robot.distances.size() - 1; ++i) {
            if (dist >= robot.distances[i] && dist <= robot.distances[i+1]) {
                double len = robot.distances[i+1] - robot.distances[i]; // Comprimento do segmento
                double ratio = (len > 0.001) ? (dist - robot.distances[i]) / len : 0.0; // Porcentagem percorrida
                
                auto& p1 = robot.path.poses[i].pose.position;
                auto& p2 = robot.path.poses[i+1].pose.position;
                
                // Interpolação Linear da Posição
                state.pose.position.x = p1.x + (p2.x - p1.x) * ratio;
                state.pose.position.y = p1.y + (p2.y - p1.y) * ratio;
                state.pose.position.z = 0.15;
                
                // Cálculo da Orientação (Yaw)
                state.yaw = std::atan2(p2.y - p1.y, p2.x - p1.x);
                tf2::Quaternion q; q.setRPY(0, 0, state.yaw);
                state.pose.orientation = tf2::toMsg(q);
                state.segment_idx = i;
                return state;
            }
        }
        
        // Fallback (segurança)
        if (!robot.path.poses.empty()) state.pose = robot.path.poses[0].pose;
        return state;
    }

    // Gera pontos brutos (x,y) dentro da caixa de colisão
    void generate_raw_points(const Rectangle& box, const RobotState& state, std::vector<Point2D>& out) 
    {
        // Encontra os limites (Min/Max) da caixa para varredura
        double minx = std::numeric_limits<double>::infinity();
        double miny = std::numeric_limits<double>::infinity();
        double maxx = -std::numeric_limits<double>::infinity();
        double maxy = -std::numeric_limits<double>::infinity();

        for (auto& v : box) {
            minx = std::min(minx, v.x); maxx = std::max(maxx, v.x);
            miny = std::min(miny, v.y); maxy = std::max(maxy, v.y);
        }
        
        // Varre a área retangular (scanline) com a resolução definida
        for (double px = minx; px <= maxx; px += CLOUD_RES) {
            for (double py = miny; py <= maxy; py += CLOUD_RES) {
                // Verifica se o ponto está geometricamente dentro do robô rotacionado
                if (point_in_rect(px, py, state.pose, state.yaw)) {
                    out.emplace_back(static_cast<float>(px), static_cast<float>(py));
                }
            }
        }
    }

    // Expande os pontos gerados adicionando uma margem de segurança (lógica toma/opa/eita)
    void expand_with_security_zone(std::vector<Point2D>& points) 
    {
        if (points.empty()) return;

        std::set<Point2D> unique_points; // Usa set para evitar pontos duplicados
        float obstacle_dist = static_cast<float>(CLOUD_RES); 
        float maxSecurityDistance_ = SECURITY_DISTANCE;
        int decimals = 2;

        // Insere pontos originais
        for (const auto& p : points) unique_points.insert(p);
        
        // Algoritmo de expansão em cruz/losango ao redor de cada ponto
        for (const auto& point : points) {
            float toma = 0.0f;
            int opa = 0;
            while (toma <= maxSecurityDistance_) {
                for (int eita = 0; eita <= opa * 2; eita++) {
                    // Gera 4 pontos simétricos expandidos
                    Point2D p1 = {
                        round_to_multiple((point.first + toma) - (obstacle_dist * eita), obstacle_dist, decimals),
                        round_to_multiple((point.second + toma), obstacle_dist, decimals)
                    };
                    Point2D p2 = {
                        round_to_multiple((point.first + toma), obstacle_dist, decimals),
                        round_to_multiple((point.second + toma) - (obstacle_dist * eita), obstacle_dist, decimals)
                    };
                    Point2D p3 = {
                        round_to_multiple((point.first - toma), obstacle_dist, decimals),
                        round_to_multiple((point.second - toma) + (obstacle_dist * eita), obstacle_dist, decimals)
                    };
                    Point2D p4 = {
                        round_to_multiple((point.first - toma) + (obstacle_dist * eita), obstacle_dist, decimals),
                        round_to_multiple((point.second - toma), obstacle_dist, decimals)
                    };
                    unique_points.insert(p1); unique_points.insert(p2);
                    unique_points.insert(p3); unique_points.insert(p4);
                }
                opa++; toma += obstacle_dist;
            }
        }
        // Atualiza o vetor original com os pontos expandidos únicos
        points.assign(unique_points.begin(), unique_points.end());
    }

    // Callback principal: Recebe caminhos e orquestra toda a lógica
    void on_paths(const mobile_manipulation_interfaces::msg::FleetPaths::SharedPtr msg) 
    {
        // Limpa dados anteriores
        fleet_.clear();
        collision_data_.clear();
        robot_colors_.clear();
        
        // 1. Parse e Inicialização dos Robôs
        for (size_t i = 0; i < msg->paths.size(); ++i) {
            RobotData data;
            data.id = static_cast<int>(msg->robot_ids[i]);
            data.frame_id = msg->paths[i].header.frame_id;
            data.path = msg->paths[i];
            data.base_speed = (i < msg->robot_speeds.size()) ? msg->robot_speeds[i] : 0.5;
            
            // Estado inicial da animação
            data.current_dist = 0.0;
            data.current_speed = data.base_speed;
            
            if (data.path.poses.size() < 2) continue; // Ignora caminhos vazios
            
            // Define cor e cria publisher para este robô
            robot_colors_[data.id] = get_robot_color(data.id);
            ensure_publisher(data.id);
            
            // Pré-calcula distâncias acumuladas e retângulos estáticos
            data.distances.push_back(0.0);
            for (size_t j = 0; j < data.path.poses.size() - 1; ++j) {
                auto& p1 = data.path.poses[j].pose.position;
                auto& p2 = data.path.poses[j+1].pose.position;
                
                // Cria retângulo estático (Broad Phase)
                data.static_rects.push_back(make_static_rect(p1, p2));
                data.risky_segments.push_back(false); // Inicialmente seguro
                
                // Acumula distância
                data.total_length += std::hypot(p2.x - p1.x, p2.y - p1.y);
                data.distances.push_back(data.total_length);
            }
            fleet_.push_back(std::move(data)); // Adiciona à frota
        }
        
        // 2. Broad Phase (Verificação Estática de Cruzamento de Caminhos)
        for (size_t i = 0; i < fleet_.size(); ++i) {
            for (size_t k = i + 1; k < fleet_.size(); ++k) {
                // Compara todos os retângulos de caminho de i com k
                for (size_t r = 0; r < fleet_[i].static_rects.size(); ++r) {
                    for (size_t s = 0; s < fleet_[k].static_rects.size(); ++s) {
                        // Se cruzam geometricamente, marca como "risky" (arriscado)
                        if (sat_intersect(fleet_[i].static_rects[r], fleet_[k].static_rects[s])) {
                            fleet_[i].risky_segments[r] = true;
                            fleet_[k].risky_segments[s] = true;
                        }
                    }
                }
            }
        }
        
        // 3. Narrow Phase (Simulação no Tempo) + Geração de Nuvens
        run_simulation_and_generate_clouds();

        // 4. Calcular Mitigação (Quem freia e quanto)
        solve_conflicts();
        
        // 5. Publicar Nuvens de Colisão (Visualização)
        publish_collision_clouds();
    }

    // Simula o futuro para detectar colisões e gerar os pontos
    void run_simulation_and_generate_clouds() 
    {
        if (fleet_.empty()) return;
        
        // Loop de tempo simulado (0 a 30s) com passo de 20ms
        for (double t = 0.0; t <= SIM_LIMIT; t += TIME_STEP) {
            std::vector<RobotState> states(fleet_.size());
            std::vector<Rectangle> boxes(fleet_.size());
            
            // Passo A: Calcula onde todos estariam no tempo 't' (sem desaceleração)
            for (size_t r = 0; r < fleet_.size(); ++r) {
                states[r] = get_state_by_time(fleet_[r], t);
                boxes[r] = make_dynamic_box(states[r].pose);
            }
            
            // Passo B: Verifica colisão par-a-par
            for (size_t i = 0; i < fleet_.size(); ++i) {
                for (size_t k = i + 1; k < fleet_.size(); ++k) {
                    // Otimização: Só verifica se ambos estiverem em segmentos marcados como 'risky'
                    int s1 = states[i].segment_idx, s2 = states[k].segment_idx;
                    bool risky = s1 >= 0 && s2 >= 0 &&
                                 s1 < (int)fleet_[i].risky_segments.size() &&
                                 s2 < (int)fleet_[k].risky_segments.size() &&
                                 fleet_[i].risky_segments[s1] && fleet_[k].risky_segments[s2];
                    
                    if (risky && sat_intersect(boxes[i], boxes[k])) {
                        // Colisão detectada!
                        int id1 = fleet_[i].id, id2 = fleet_[k].id;
                        
                        // Atualiza info para Robot 1
                        auto& info1 = collision_data_[id1];
                        info1.colliding_with.push_back(id2);
                        if (info1.start_time < 0) { // Se for o primeiro instante da batida
                            info1.start_time = t;
                            info1.entry_distances[id1] = t * fleet_[i].base_speed; // Guarda onde começa
                        }
                        info1.end_time = t; // Atualiza fim
                        info1.exit_distances[id1] = t * fleet_[i].base_speed; // Guarda onde termina
                        generate_raw_points(boxes[i], states[i], info1.points); // Gera pontos
                        
                        // Atualiza info para Robot 2
                        auto& info2 = collision_data_[id2];
                        info2.colliding_with.push_back(id1);
                        if (info2.start_time < 0) {
                            info2.start_time = t;
                            info2.entry_distances[id2] = t * fleet_[k].base_speed;
                        }
                        info2.end_time = t;
                        info2.exit_distances[id2] = t * fleet_[k].base_speed;
                        generate_raw_points(boxes[k], states[k], info2.points);
                    }
                }
            }
        }
    }

    // Calcula a lógica de quem deve parar/desacelerar (Mitigação de Conflito)
    void solve_conflicts()
    {
        for (auto& [robot_id, info] : collision_data_) {
            if (info.colliding_with.empty()) continue;

            for (int other_id : info.colliding_with) {
                // REGRA DE OURO: Robô com menor ID cede passagem (é quem desacelera)
                if (robot_id < other_id) {
                    
                    RobotData* myself = get_robot_by_id(robot_id);
                    RobotData* other = get_robot_by_id(other_id);

                    if (!myself || !other) continue;

                    auto& other_info = collision_data_[other_id];
                    
                    // -- CÁLCULO FÍSICO CORRIGIDO (Lógica de aproximação) --
                    
                    // Distância até o ponto de parada/frenagem
                    double dist_to_collision = info.entry_distances[robot_id];
                    double dist_braking_point = dist_to_collision - APPROACH_DISTANCE;
                    
                    // Tempo que leva pra chegar no ponto de frenagem na velocidade normal
                    double time_at_braking = dist_braking_point / myself->base_speed;
                    if (time_at_braking < 0) time_at_braking = 0; // Já passou do ponto

                    // Tempo alvo: Quando o outro robô sai + margem
                    double target_arrival_time = other_info.end_time + SAFETY_TIME_MARGIN;

                    // Quanto tempo tenho que "queimar" percorrendo apenas a APPROACH_DISTANCE?
                    // (Tempo Alvo) - (Tempo que cheguei no ponto de freio)
                    double time_to_cover_approach = target_arrival_time - time_at_braking;

                    // Se o tempo for negativo ou muito curto, significa que já estou muito atrasado
                    if (time_to_cover_approach < 0.1) time_to_cover_approach = 0.1;

                    // V = d / t -> Velocidade necessária nesse trecho curto
                    double safe_speed = APPROACH_DISTANCE / time_to_cover_approach;

                    // Limite inferior de segurança (não parar totalmente)
                    if (safe_speed < 0.05) safe_speed = 0.05;

                    // Configura dados de mitigação no robô
                    myself->mitigation.active = true;
                    myself->mitigation.safe_speed = safe_speed;
                    myself->mitigation.conflict_entry_dist = dist_to_collision;
                    
                    // Guarda dados para monitoramento em tempo real (para retomar velocidade)
                    myself->mitigation.partner_id = other_id;
                    myself->mitigation.partner_exit_dist = other_info.exit_distances[other_id];

                    RCLCPP_WARN(get_logger(), "MITIGAÇÃO: R%d reduz para %.2f m/s nos ultimos %.1fm (Espera R%d).",
                        robot_id, safe_speed, APPROACH_DISTANCE, other_id);
                }
            }
        }
    }

    // ==================================================================================
    // SEÇÃO 2: VISUALIZAÇÃO E ANIMAÇÃO
    // ==================================================================================

    // Garante que o publisher do robô existe no mapa
    void ensure_publisher(int robot_id) 
    {
        if (cloud_publishers_.find(robot_id) == cloud_publishers_.end()) {
            std::string topic = "/fleet/collision_cloud/robot_" + std::to_string(robot_id);
            cloud_publishers_[robot_id] = create_publisher<sensor_msgs::msg::PointCloud2>(topic, 10);
        }
    }

    // Retorna uma cor fixa baseada no ID do robô para os marcadores
    RGB get_robot_color(int id) 
    {
        static const std::vector<RGB> palette = {
            {255, 0, 0}, {0, 0, 255}, {0, 255, 0}, {255, 165, 0},
            {128, 0, 128}, {0, 255, 255}, {255, 0, 255}, {255, 255, 0}
        };
        return palette[std::abs(id) % palette.size()];
    }

    // Converte vetor de pontos C++ para mensagem ROS PointCloud2
    sensor_msgs::msg::PointCloud2 to_pointcloud2(const std::vector<Point2D>& points, 
                                                  const std::string& frame_id, 
                                                  const RGB& color,
                                                  float z = 0.05f) 
    {
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.frame_id = frame_id;
        cloud.header.stamp = now();
        cloud.height = 1;
        cloud.width = points.size();
        cloud.is_dense = true;
        cloud.is_bigendian = false;
        
        // Configura campos x, y, z, rgb
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(points.size());
        
        // Iteradores para preencher dados binários
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb(cloud, "rgb");
        
        for (size_t i = 0; i < points.size(); ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb) {
            *iter_x = points[i].first;
            *iter_y = points[i].second;
            *iter_z = z; // Altura fixa
            iter_rgb[0] = color.r; iter_rgb[1] = color.g; iter_rgb[2] = color.b;
        }
        return cloud;
    }

    // Função dedicada para publicar as nuvens no RViz após cálculo
    void publish_collision_clouds() 
    {
        for (auto& [robot_id, info] : collision_data_) {
            // Filtro opcional: Mostra apenas robô 0 se desejado
            if (robot_id != 0) continue; 

            if (!info.points.empty()) {
                // Expande pontos com margem de segurança antes de publicar
                expand_with_security_zone(info.points);
                
                if (cloud_publishers_.count(robot_id)) {
                    const std::string& frame_id = fleet_.empty() ? "map" : fleet_[0].frame_id;
                    auto& color = robot_colors_[robot_id];
                    
                    auto cloud_msg = to_pointcloud2(info.points, frame_id, color, 0.05f);
                    cloud_publishers_[robot_id]->publish(cloud_msg);
                }
            }
        }
    }

    // Loop de Animação: Atualiza posições e desenha no RViz
    void animate() 
    {
        if (fleet_.empty()) return;
        
        // Prepara array de marcadores para deletar os antigos e adicionar novos
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker del;
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        markers.markers.push_back(del);
        int marker_id = 0;
        
        bool all_finished = true;

        for (auto& robot : fleet_) {
            // Verifica se o robô terminou o caminho para reiniciar o loop
            if (robot.current_dist >= robot.total_length) {
                robot.current_dist = 0.0;
            } else {
                all_finished = false;
            }

            // --- LÓGICA DE VELOCIDADE DINÂMICA (Resume Speed) ---
            double target_speed = robot.base_speed;
            
            // Se houver mitigação ativa para este robô
            if (robot.mitigation.active) {
                double dist_to_entry = robot.mitigation.conflict_entry_dist - robot.current_dist;
                
                // Verifica se o robô parceiro já limpou a área
                bool partner_cleared = false;
                RobotData* partner = get_robot_by_id(robot.mitigation.partner_id);
                if (partner) {
                    // Se o parceiro já passou da saída da colisão (+ margem pequena)
                    if (partner->current_dist > robot.mitigation.partner_exit_dist + 0.1) {
                        partner_cleared = true;
                    }
                }

                if (partner_cleared) {
                    // SE O CAMINHO ESTÁ LIVRE, VOLTA A ACELERAR!
                    target_speed = robot.base_speed;
                } 
                else if (dist_to_entry <= APPROACH_DISTANCE && dist_to_entry > -0.5) {
                    // SE ESTÁ CHEGANDO (1.5m) E AINDA TEM PERIGO -> MANTÉM VELOCIDADE SEGURA
                    target_speed = robot.mitigation.safe_speed;
                }
                // Se ainda está longe da colisão (> 1.5m), mantém velocidade normal
            }

            // Atualiza velocidade e posição
            robot.current_speed = target_speed;
            double step = robot.current_speed * ANIMATION_FREQ; // d = v * dt
            robot.current_dist += step;

            // Obtém Pose visual para desenhar
            auto state = get_state_by_distance(robot, robot.current_dist);
            auto& color = robot_colors_[robot.id];
            std::string id_str = std::to_string(robot.id);
            
            // --- DESENHO DOS MARCADORES ---
            
            // 1. Cubo representando o Robô
            visualization_msgs::msg::Marker bot;
            bot.header.frame_id = robot.frame_id;
            bot.header.stamp = now();
            bot.ns = "bot_" + id_str;
            bot.id = 0;
            bot.type = visualization_msgs::msg::Marker::CUBE;
            bot.action = visualization_msgs::msg::Marker::ADD;
            bot.pose = state.pose;
            bot.scale.x = bot.scale.y = ROBOT_WIDTH; bot.scale.z = 0.3;
            
            // Muda cor para Laranja se estiver mitigando velocidade, senão usa cor do ID
            if (std::abs(robot.current_speed - robot.base_speed) > 0.01) {
                bot.color.r = 1.0; bot.color.g = 0.5; bot.color.b = 0.0; bot.color.a = 1.0; 
            } else {
                bot.color.r = color.r/255.0; bot.color.g = color.g/255.0; bot.color.b = color.b/255.0; bot.color.a = 0.8;
            }
            markers.markers.push_back(bot);
            
            // 2. Texto de Velocidade acima do robô
            visualization_msgs::msg::Marker txt;
            txt.header.frame_id = robot.frame_id;
            txt.header.stamp = now();
            txt.ns = "spd_" + id_str;
            txt.id = 0;
            txt.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            txt.action = visualization_msgs::msg::Marker::ADD;
            txt.pose = state.pose; txt.pose.position.z += 0.5;
            txt.scale.z = 0.2; txt.color.r=1.0; txt.color.g=1.0; txt.color.b=1.0; txt.color.a=1.0;
            std::stringstream ss; ss << std::fixed << std::setprecision(2) << robot.current_speed << " m/s";
            txt.text = ss.str();
            markers.markers.push_back(txt);

            // 3. Desenho do Caminho Estático no chão
            for (size_t r = 0; r < robot.static_rects.size(); ++r) {
                visualization_msgs::msg::Marker rect;
                rect.header.frame_id = robot.frame_id;
                rect.header.stamp = now();
                rect.ns = "path_" + id_str;
                rect.id = marker_id++;
                rect.type = visualization_msgs::msg::Marker::LINE_STRIP;
                rect.action = visualization_msgs::msg::Marker::ADD;
                rect.scale.x = 0.02; // Espessura da linha
                
                // Se segmento é perigoso (cruzamento), fica vermelho transparente
                if (robot.risky_segments[r]) { rect.color.r = 1.0; rect.color.a = 0.3; }
                else { rect.color.r = color.r/255.0; rect.color.g = color.g/255.0; rect.color.b = color.b/255.0; rect.color.a = 0.15; }
                
                // Converte vértices do retângulo para pontos do marcador
                for (auto& v : robot.static_rects[r]) {
                    geometry_msgs::msg::Point p; p.x=v.x; p.y=v.y; rect.points.push_back(p);
                }
                rect.points.push_back(rect.points[0]); // Fecha o loop
                markers.markers.push_back(rect);
            }
        }
        // Publica todos os marcadores de uma vez
        pub_markers_->publish(markers);
    }
};

// Função principal (Entry Point)
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv); // Inicializa ROS 2
    rclcpp::spin(std::make_shared<FleetManagement>()); // Mantém o nó rodando
    rclcpp::shutdown(); // Encerra
    return 0;
}