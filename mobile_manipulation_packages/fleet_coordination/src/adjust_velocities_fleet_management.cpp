#include <memory>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "mobile_manipulation_interfaces_fleet/msg/fleet_paths.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_srvs/srv/trigger.hpp"
// --- ESTRUTURAS AUXILIARES ---

struct GridKey {
    int x, y;
    bool operator==(const GridKey& other) const { return x == other.x && y == other.y; }
    struct Hash {
        std::size_t operator()(const GridKey& k) const {
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
        }
    };
};

struct RobotVisit {
    int robot_id;
    float timestamp;
};

struct ResolutionLog {
    int zone_id;
    int step_number;
    int r_priority;
    int r_victim;
    float priority_end;
    float victim_shifted_start;   
    float victim_final_start;     
    float victim_total_delay;     
    float dist_traveled;
    float required_speed;
    bool is_physical_violation;
};

struct CollisionZone {
    int id;
    std::set<int> involved_robots; 
    std::vector<GridKey> points;
    std::unordered_set<GridKey, GridKey::Hash> points_set; 
    std::map<int, std::vector<std::pair<float, float>>> robot_intervals; 
    
    float get_earliest_entry() const {
        float min_t = std::numeric_limits<float>::max();
        for(const auto& [rid, intervals] : robot_intervals) {
            if(!intervals.empty()) min_t = std::min(min_t, intervals[0].first);
        }
        return min_t;
    }
};

struct TrajectoryPoint {
    float time;
    float x;
    float y;
};

// --- CLASSE PRINCIPAL ---

class FleetManager : public rclcpp::Node {
public:
    FleetManager() : Node("cbs_fleet_manager") 
    {
        // 1. CONFIGURAÇÃO E PARÂMETROS
        this->declare_parameter<double>("path_resolution", 0.05);       
        this->declare_parameter<double>("simulation_base_speed", 2.0);  
        this->declare_parameter<double>("min_robot_gap", 0.5);          
        this->declare_parameter<double>("robot_radius", 0.3);           
        this->declare_parameter<double>("time_gap_tolerance", 2.0);     
        this->declare_parameter<int>("animation_rate_ms", 20);          
        this->declare_parameter<bool>("viz_show_zones", true);
        this->declare_parameter<bool>("viz_show_paths", true);

        config_.resolution = static_cast<float>(this->get_parameter("path_resolution").as_double());
        config_.base_speed = static_cast<float>(this->get_parameter("simulation_base_speed").as_double());
        config_.min_robot_gap = static_cast<float>(this->get_parameter("min_robot_gap").as_double());
        config_.robot_radius = static_cast<float>(this->get_parameter("robot_radius").as_double());
        config_.time_gap_tolerance = static_cast<float>(this->get_parameter("time_gap_tolerance").as_double());
        config_.animation_rate_ms = this->get_parameter("animation_rate_ms").as_int();
        
        config_.show_zones = this->get_parameter("viz_show_zones").as_bool();
        config_.show_paths = this->get_parameter("viz_show_paths").as_bool();

        decimals = count_decimals(config_.resolution);

        sub_fleet_ = this->create_subscription<mobile_manipulation_interfaces_fleet::msg::FleetPaths>(
            "/fleet/all_robot_plans", 
            10, 
            std::bind(&FleetManager::fleet_callback, this, std::placeholders::_1));
            
        pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/simulation_markers", 10);
        pub_zone_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/viz_zones", 10);
        pub_path_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/viz_static_paths", 10);
        
        scenario_client_ = this->create_client<std_srvs::srv::Trigger>("/fleet/generate_scenario");
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(config_.animation_rate_ms), 
            std::bind(&FleetManager::animation_loop, this));

        RCLCPP_INFO(this->get_logger(), "Fleet Manager: Modo Lookahead (Retorno automatico a velocidade maxima).");
        request_new_scenario();
    }

private:
    struct FleetConfig {
        float resolution;
        float base_speed;
        float min_robot_gap;
        float robot_radius;
        float time_gap_tolerance;
        int animation_rate_ms;
        bool show_zones;
        bool show_paths;
    } config_;

    int decimals;
    rclcpp::Subscription<mobile_manipulation_interfaces_fleet::msg::FleetPaths>::SharedPtr sub_fleet_; 
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_zone_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_path_markers_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr scenario_client_;

    std::unordered_map<GridKey, std::vector<RobotVisit>, GridKey::Hash> global_occupancy_map;
    std::map<int, std::vector<std::pair<float, float>>> original_paths_;
    std::map<int, std::vector<TrajectoryPoint>> animated_trajectories_;
    
    float sim_time_ = 0.0f;
    float max_sim_time_ = 0.0f;
    bool is_simulating_ = false;

    struct IntervalRef { int robot_id; float start; float end; float original_start; };

    void request_new_scenario() {
        if (!scenario_client_->wait_for_service(std::chrono::seconds(1))) {
            RCLCPP_WARN(this->get_logger(), "Gerador de cenarios nao disponivel.");
            return;
        }
        
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        
        // Chama assincronamente para não bloquear o spin do nó
        auto future_result = scenario_client_->async_send_request(request, 
            [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
                try {
                    auto response = future.get();
                    if(response->success) {
                        RCLCPP_INFO(this->get_logger(), "Novo cenario solicitado com sucesso!");
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Gerador recusou: %s", response->message.c_str());
                    }
                } catch (const std::exception &e) {
                    RCLCPP_ERROR(this->get_logger(), "Falha na chamada do servico: %s", e.what());
                }
            });
    }

    // --- UTILS ---
    GridKey to_key(float x, float y) {
        return { static_cast<int>(std::round(x / config_.resolution)), static_cast<int>(std::round(y / config_.resolution)) };
    }
    
    std::set<int> get_robots_at_point(const GridKey& key) {
        std::set<int> robots;
        if (global_occupancy_map.find(key) != global_occupancy_map.end()) {
            for (const auto& visit : global_occupancy_map.at(key)) robots.insert(visit.robot_id);
        }
        return robots;
    }

    std_msgs::msg::ColorRGBA get_color_for_id(int id, float alpha = 1.0) {
        std_msgs::msg::ColorRGBA color;
        color.a = alpha;
        float h = std::fmod((id * 0.618033988749895), 1.0f);
        float s = 0.8; float v = 0.95;
        int i = (int)(h * 6); float f = h * 6 - i;
        float p = v * (1 - s); float q = v * (1 - f * s); float t = v * (1 - (1 - f) * s);
        switch (i % 6) {
            case 0: color.r = v; color.g = t; color.b = p; break;
            case 1: color.r = q; color.g = v; color.b = p; break;
            case 2: color.r = p; color.g = v; color.b = t; break;
            case 3: color.r = p; color.g = q; color.b = v; break;
            case 4: color.r = t; color.g = p; color.b = v; break;
            case 5: color.r = v; color.g = p; color.b = q; break;
        }
        return color;
    }

    // --- CALLBACK PRINCIPAL ---
    void fleet_callback(const mobile_manipulation_interfaces_fleet::msg::FleetPaths::SharedPtr msg)
    {
        global_occupancy_map.clear();
        original_paths_.clear();

        for (size_t i = 0; i < msg->paths.size(); ++i) {
            int r_id = (i < msg->robot_ids.size()) ? msg->robot_ids[i] : (int)i;
            float speed_sim = config_.base_speed; 
            const auto& nav_path = msg->paths[i];
            if (nav_path.poses.empty()) continue;

            std::vector<std::pair<float, float>> raw_points;
            for (const auto& pose_stamped : nav_path.poses) {
                raw_points.push_back({
                    static_cast<float>(pose_stamped.pose.position.x),
                    static_cast<float>(pose_stamped.pose.position.y)
                });
            }
            original_paths_[r_id] = raw_points;
            process_robot_volume(r_id, raw_points, speed_sim);
        }

        std::vector<CollisionZone> zones_vector = detect_and_cluster_collisions();
        auto result = analyze_and_resolve_conflicts(zones_vector);
        
        print_super_detailed_report(zones_vector, result.first, result.second);
        
        // Geração de trajetória usando o controlador Lookahead
        generate_visualization_trajectories_lookahead(result.first);

        publish_zone_visuals(zones_vector);
        publish_robot_footprints();
        // request_new_scenario();
    }

    // --- VISUALIZACAO ESTATICA ---
    void publish_zone_visuals(const std::vector<CollisionZone>& zones) {
        if (!config_.show_zones) return;
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker del; del.action = 3; markers.markers.push_back(del);

        for (const auto& zone : zones) {
            visualization_msgs::msg::Marker mk;
            mk.header.frame_id = "world"; mk.header.stamp = this->now();
            mk.ns = "zones"; mk.id = zone.id; mk.type = 6; mk.action = 0;
            mk.scale.x = config_.resolution; mk.scale.y = config_.resolution; mk.scale.z = config_.resolution;
            mk.color = get_color_for_id(zone.id * 10, 0.6);

            for (const auto& pt : zone.points) {
                geometry_msgs::msg::Point p;
                p.x = pt.x * config_.resolution; p.y = pt.y * config_.resolution; p.z = 0.05;
                mk.points.push_back(p);
            }
            markers.markers.push_back(mk);
        }
        pub_zone_markers_->publish(markers);
    }

    void publish_robot_footprints() {
        if (!config_.show_paths) return;
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker del; del.action = 3; markers.markers.push_back(del);

        std::map<int, std::vector<geometry_msgs::msg::Point>> robot_points;
        for (const auto& entry : global_occupancy_map) {
            for (const auto& visit : entry.second) {
                geometry_msgs::msg::Point p;
                p.x = entry.first.x * config_.resolution;
                p.y = entry.first.y * config_.resolution;
                p.z = -0.05; 
                robot_points[visit.robot_id].push_back(p);
            }
        }

        for (const auto& [rid, points] : robot_points) {
            visualization_msgs::msg::Marker mk;
            mk.header.frame_id = "world"; mk.header.stamp = this->now();
            mk.ns = "static_paths"; mk.id = rid; mk.type = 6; mk.action = 0;
            mk.scale.x = config_.resolution; mk.scale.y = config_.resolution; mk.scale.z = 0.02;
            mk.color = get_color_for_id(rid, 0.4);
            mk.points = points;
            markers.markers.push_back(mk);
        }
        pub_path_markers_->publish(markers);
    }

    // --- NOVA ANIMAÇÃO: CONTROLADOR LOOKAHEAD ---
    // Verifica a cada instante: "Tenho algum compromisso de horario na minha frente?"
    // Se não, volta a base_speed.
    void generate_visualization_trajectories_lookahead(const std::vector<ResolutionLog>& logs) 
    {
        animated_trajectories_.clear();
        sim_time_ = 0.0f;
        max_sim_time_ = 0.0f;

        // 1. Organiza os checkpoints (constraints) para cada robô
        std::map<int, std::vector<std::pair<float, float>>> robot_checkpoints;
        for(const auto& log : logs) {
            robot_checkpoints[log.r_victim].push_back({log.dist_traveled, log.victim_final_start});
        }
        for(auto& [rid, checkpoints] : robot_checkpoints) {
            std::sort(checkpoints.begin(), checkpoints.end());
        }

        // 2. Simula o robô andando no caminho
        for(const auto& [rid, path] : original_paths_) {
            if(path.empty()) continue;
            
            float current_sim_time = 0.0f;
            float current_dist_traveled = 0.0f;
            
            // Ponto inicial
            animated_trajectories_[rid].push_back({0.0f, path[0].first, path[0].second});

            for(size_t i = 0; i < path.size() - 1; ++i) {
                float dx = path[i+1].first - path[i].first;
                float dy = path[i+1].second - path[i].second;
                float segment_total_dist = std::hypot(dx, dy);
                
                // Vamos simular pequenos passos dentro deste segmento para detectar mudanças de velocidade
                float dist_in_segment = 0.0f;
                float sim_step = 0.05f; // Resolução da simulação de movimento (metros)

                while(dist_in_segment < segment_total_dist) {
                    float step = std::min(sim_step, segment_total_dist - dist_in_segment);
                    
                    // --- LÓGICA LOOKAHEAD ---
                    float target_speed = config_.base_speed;
                    
                    // Verifica se existe alguma restrição futura
                    if (robot_checkpoints.count(rid)) {
                        for (const auto& cp : robot_checkpoints[rid]) {
                            float target_dist = cp.first;
                            float target_time = cp.second;

                            // Se a restrição está à frente
                            if (target_dist > current_dist_traveled) {
                                float dist_to_go = target_dist - current_dist_traveled;
                                float time_to_go = target_time - current_sim_time;

                                if (time_to_go > 0.001f) {
                                    float req_speed = dist_to_go / time_to_go;
                                    // Se precisarmos andar mais devagar que o maximo para cumprir horario, obedecemos
                                    if (req_speed < target_speed) {
                                        target_speed = req_speed;
                                    }
                                }
                                // Encontramos a primeira restrição ativa, paramos de procurar (as outras estao depois)
                                break; 
                            }
                        }
                    }

                    // Aplica velocidade (com mínimo de segurança para não travar divisão por zero)
                    if(target_speed < 0.01f) target_speed = 0.01f;

                    float dt = step / target_speed;
                    current_sim_time += dt;
                    current_dist_traveled += step;
                    dist_in_segment += step;

                    // Interpolação da posição
                    float ratio = dist_in_segment / segment_total_dist;
                    float px = path[i].first + ratio * dx;
                    float py = path[i].second + ratio * dy;

                    animated_trajectories_[rid].push_back({current_sim_time, px, py});
                }
            }
            if(current_sim_time > max_sim_time_) max_sim_time_ = current_sim_time;
        }
        
        max_sim_time_ += 2.0f; 
        is_simulating_ = true;
    }

    void animation_loop() {
        if(!is_simulating_ || animated_trajectories_.empty()) return;
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker del_mk; del_mk.action = 3; markers.markers.push_back(del_mk);

        sim_time_ += (config_.animation_rate_ms / 1000.0f) * 2.0f; 
        if(sim_time_ > max_sim_time_) sim_time_ = 0.0f; 

        for(const auto& [rid, traj] : animated_trajectories_) {
            if(traj.empty()) continue;
            float x = traj.back().x; float y = traj.back().y;
            
            for(size_t i = 0; i < traj.size() - 1; ++i) {
                if(sim_time_ >= traj[i].time && sim_time_ <= traj[i+1].time) {
                    float total_dt = traj[i+1].time - traj[i].time;
                    if(total_dt > 0.0001) {
                        float ratio = (sim_time_ - traj[i].time) / total_dt;
                        x = traj[i].x + ratio * (traj[i+1].x - traj[i].x);
                        y = traj[i].y + ratio * (traj[i+1].y - traj[i].y);
                    }
                    break;
                }
            }
            
            visualization_msgs::msg::Marker robot_mk;
            robot_mk.header.frame_id = "world"; robot_mk.header.stamp = this->now();
            robot_mk.ns = "simulated_robots"; robot_mk.id = rid;
            robot_mk.type = 3; robot_mk.action = 0;
            robot_mk.pose.position.x = x; robot_mk.pose.position.y = y; robot_mk.pose.position.z = 0.2;
            robot_mk.scale.x = 0.4; robot_mk.scale.y = 0.4; robot_mk.scale.z = 0.4;
            robot_mk.color = get_color_for_id(rid, 1.0); 
            markers.markers.push_back(robot_mk);

            visualization_msgs::msg::Marker text_mk = robot_mk;
            text_mk.type = 9; text_mk.ns = "ids"; text_mk.id = rid+100;
            text_mk.pose.position.z = 0.6; text_mk.scale.z = 0.3;
            text_mk.color.r=1; text_mk.color.g=1; text_mk.color.b=1;
            std::stringstream ss; ss << "R" << rid; text_mk.text = ss.str();
            markers.markers.push_back(text_mk);
        }
        pub_markers_->publish(markers);
    }

    std::pair<float, std::pair<float, float>> calculate_approach_metrics(int robot_id, const CollisionZone& zone) 
    {
        if (original_paths_.find(robot_id) == original_paths_.end()) return {0.0f, {0,0}};
        const auto& path = original_paths_[robot_id];
        if (path.empty()) return {0.0f, {0,0}};
        GridKey start_k = to_key(path[0].first, path[0].second);
        if (zone.points_set.count(start_k)) return {0.0f, path[0]}; 
        float total_dist_traveled = 0.0f;
        for (size_t i = 0; i < path.size() - 1; ++i) {
            float ax = path[i].first; float ay = path[i].second;
            float bx = path[i+1].first; float by = path[i+1].second;
            float dist = std::hypot(bx - ax, by - ay);
            if (dist < 1e-6) continue;
            float t_dist = config_.resolution;
            float ux = (bx - ax) / dist; float uy = (by - ay) / dist;
            while (t_dist < dist) {
                float cur_x = ax + t_dist * ux;
                float cur_y = ay + t_dist * uy;
                GridKey k = to_key(cur_x, cur_y);
                if (zone.points_set.count(k)) return {total_dist_traveled + t_dist, {cur_x, cur_y}};
                t_dist += config_.resolution;
            }
            total_dist_traveled += dist;
        }
        return {total_dist_traveled, path.back()}; 
    }

    // --- SOLVER PRIORIZADO ---
    std::pair<std::vector<ResolutionLog>, std::map<int, float>> analyze_and_resolve_conflicts(std::vector<CollisionZone>& zones)
    {
        std::vector<ResolutionLog> logs;
        std::map<int, float> global_delays; 
        int log_step = 1;

        std::set<int> all_robot_ids;
        for(const auto& zone : zones) for(int rid : zone.involved_robots) all_robot_ids.insert(rid);

        struct ReservedSlot { float start; float end; int owner_id; };
        std::map<int, std::vector<ReservedSlot>> reservation_table;

        for(int current_robot : all_robot_ids) {
            float speed_factor = 1.0f; 
            bool factor_changed = true;

            while(factor_changed) {
                factor_changed = false;
                for(const auto& zone : zones) {
                    if(zone.robot_intervals.find(current_robot) == zone.robot_intervals.end()) continue;

                    for(const auto& my_interval : zone.robot_intervals.at(current_robot)) {
                        float original_duration = my_interval.second - my_interval.first;
                        float new_duration = original_duration / speed_factor;
                        float my_start = my_interval.first / speed_factor;
                        float my_end = my_start + new_duration;

                        if(reservation_table.count(zone.id)) {
                            auto& reserved = reservation_table[zone.id];
                            std::sort(reserved.begin(), reserved.end(), [](auto& a, auto& b){ return a.start < b.start; });

                            for(const auto& slot : reserved) {
                                float overlap_start = std::max(my_start, slot.start);
                                float overlap_end = std::min(my_end, slot.end);
                                bool collision = (overlap_start < overlap_end + config_.min_robot_gap);

                                if(collision) {
                                    float required_arrival = slot.end + config_.min_robot_gap;
                                    float new_factor = my_interval.first / required_arrival;

                                    if(new_factor < speed_factor - 0.001) {
                                        auto metrics = calculate_approach_metrics(current_robot, zone);
                                        float dist = metrics.first;
                                        float req_speed = config_.base_speed * new_factor;

                                        ResolutionLog log;
                                        log.zone_id = zone.id;
                                        log.step_number = log_step++;
                                        log.r_priority = slot.owner_id;
                                        log.r_victim = current_robot;
                                        log.priority_end = slot.end;
                                        log.victim_final_start = required_arrival;
                                        log.dist_traveled = dist;
                                        log.required_speed = req_speed;
                                        log.is_physical_violation = false; 

                                        logs.push_back(log);

                                        speed_factor = new_factor;
                                        factor_changed = true;
                                        goto restart_checks;
                                    }
                                }
                            }
                        }
                    }
                }
                restart_checks:;
            }

            for(const auto& zone : zones) {
                if(zone.robot_intervals.count(current_robot)) {
                    for(const auto& interval : zone.robot_intervals.at(current_robot)) {
                        float original_duration = interval.second - interval.first;
                        float new_duration = original_duration / speed_factor;
                        float my_start = interval.first / speed_factor;
                        
                        reservation_table[zone.id].push_back({my_start, my_start + new_duration, current_robot});
                    }
                }
            }
        }
        return {logs, {}}; 
    }

    void print_super_detailed_report(
        const std::vector<CollisionZone>& zones, 
        const std::vector<ResolutionLog>& logs,
        const std::map<int, float>& final_delays) 
    {
        std::cout << "\n\n";
        std::cout << "================================================================================\n";
        std::cout << "||   RELATÓRIO DE TRÁFEGO (VELOCIDADE DINÂMICA)   ||\n";
        std::cout << "================================================================================\n";
        
        std::cout << ">>> LOG DE RESOLUÇÃO:\n";
        if(logs.empty()) std::cout << "    [Fluxo livre.]\n";
        
        for(const auto& log : logs) {
            std::cout << "    +--------------------------------------------------------------------------+\n";
            std::cout << "    | ZONA " << log.zone_id << " (Evento #" << log.step_number << ")\n";
            std::cout << "    | [CONFLITO]   Robo " << log.r_victim << " bateria no Robo " << log.r_priority << ".\n";
            std::cout << "    | [SOLUÇÃO]    Reduzir velocidade para chegar em " 
                      << std::fixed << std::setprecision(2) << log.victim_final_start << "s.\n";
            std::cout << "    |              - VELOCIDADE NECESSARIA: >>> " << log.required_speed << " m/s <<<\n";
            std::cout << "    |              - (Apos passar, retoma velocidade maxima)\n";
            std::cout << "    +--------------------------------------------------------------------------+\n";
        }
        std::cout << "================================================================================\n\n";
    }

    void process_robot_volume(int robot_id, const std::vector<std::pair<float, float>>& waypoints, float speed) {
        if(waypoints.empty()) return;
        float current_time = 0.0f; 
        expand_point_to_map(robot_id, waypoints[0], current_time);
        for(size_t i = 0; i < waypoints.size() - 1; i++) {
            float ax = waypoints[i].first; float ay = waypoints[i].second;
            float bx = waypoints[i+1].first; float by = waypoints[i+1].second;
            float dist = std::hypot(bx - ax, by - ay);
            if (dist < 1e-6) continue;
            float t_dist = config_.resolution;
            float ux = (bx - ax) / dist; float uy = (by - ay) / dist;
            while (t_dist < dist) {
                float rx = ax + t_dist * ux; float ry = ay + t_dist * uy;
                float px = round_to_multiple(rx, config_.resolution, decimals);
                float py = round_to_multiple(ry, config_.resolution, decimals);
                float time_at_point = current_time + (t_dist / speed);
                expand_point_to_map(robot_id, {px, py}, time_at_point);
                t_dist += config_.resolution;
            }
            current_time += (dist / speed);
            expand_point_to_map(robot_id, {bx, by}, current_time);
        }
    }

    void expand_point_to_map(int robot_id, std::pair<float, float> center, float timestamp) {
        insert_visit(robot_id, center, timestamp);
        float toma = 0.0; int opa = 0;
        while(toma <= config_.robot_radius) {
            for(int eita = 0; eita <= opa * 2; eita++) {   
                std::vector<std::pair<float, float>> expansion = {
                    { (center.first + toma) - (config_.resolution * eita), (center.second + toma) },
                    { (center.first + toma), (center.second + toma) - (config_.resolution * eita) },
                    { (center.first - toma), (center.second - toma) + (config_.resolution * eita) },
                    { (center.first - toma) + (config_.resolution * eita), (center.second - toma) }
                };
                for(auto& p : expansion) {
                    p.first = round_to_multiple(p.first, config_.resolution, decimals);
                    p.second = round_to_multiple(p.second, config_.resolution, decimals);
                    insert_visit(robot_id, p, timestamp);
                }
            }
            opa++; toma += config_.resolution;
        }
    }

    void insert_visit(int robot_id, std::pair<float, float> p, float time) {
        GridKey k = to_key(p.first, p.second);
        global_occupancy_map[k].push_back({robot_id, time});
    }

    std::vector<CollisionZone> detect_and_cluster_collisions() {
        std::vector<CollisionZone> zones;
        std::unordered_set<GridKey, GridKey::Hash> visited_keys; 
        std::vector<GridKey> collision_candidates;
        for(const auto& entry : global_occupancy_map) {
            std::set<int> unique_robots;
            for(const auto& visit : entry.second) unique_robots.insert(visit.robot_id);
            if(unique_robots.size() > 1) collision_candidates.push_back(entry.first);
        }
        std::unordered_set<GridKey, GridKey::Hash> candidate_set(collision_candidates.begin(), collision_candidates.end());
        int zone_counter = 0;
        for(const auto& start_key : collision_candidates) {
            if(visited_keys.count(start_key)) continue;
            std::set<int> zone_signature = get_robots_at_point(start_key);
            CollisionZone current_zone;
            current_zone.id = zone_counter++;
            current_zone.involved_robots = zone_signature;
            std::vector<GridKey> queue;
            queue.push_back(start_key);
            visited_keys.insert(start_key);
            while(!queue.empty()) {
                GridKey curr = queue.back(); queue.pop_back();
                current_zone.points.push_back(curr);
                current_zone.points_set.insert(curr); 
                int dx[] = {1, -1, 0, 0}; int dy[] = {0, 0, 1, -1};
                for(int i=0; i<4; i++) {
                    GridKey neighbor = {curr.x + dx[i], curr.y + dy[i]};
                    if(candidate_set.count(neighbor) && visited_keys.find(neighbor) == visited_keys.end()) {
                        if(get_robots_at_point(neighbor) == zone_signature) {
                            visited_keys.insert(neighbor); queue.push_back(neighbor);
                        }
                    }
                }
            }
            std::map<int, std::vector<float>> raw_times;
            for(const auto& pt : current_zone.points) {
                for(const auto& v : global_occupancy_map[pt]) {
                    if(current_zone.involved_robots.count(v.robot_id)) raw_times[v.robot_id].push_back(v.timestamp);
                }
            }
            for(auto& [rid, times] : raw_times) {
                if(times.empty()) continue;
                std::sort(times.begin(), times.end());
                float start_t = times[0]; float end_t = times[0];
                for(size_t i = 1; i < times.size(); i++) {
                    if((times[i] - end_t) > config_.time_gap_tolerance) {
                        current_zone.robot_intervals[rid].push_back({start_t, end_t});
                        start_t = times[i]; 
                    }
                    end_t = times[i]; 
                }
                current_zone.robot_intervals[rid].push_back({start_t, end_t});
            }
            zones.push_back(current_zone);
        }
        return zones;
    }

    inline float round_to_multiple(float value, float multiple, int decimals) {
        if (multiple == 0.0) return value; 
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0, decimals);
        result = std::round(result * factor) / factor;
        return result;
    }
    
    int count_decimals(float number) {
        float fractional = std::fabs(number - std::floor(number));
        int decimals = 0;
        const float epsilon = 1e-9; 
        while (fractional > epsilon && decimals < 20) {
            fractional *= 10; fractional -= std::floor(fractional); decimals++;
        }
        return decimals;
    }
};

int main(int argc, char* argv[]) 
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FleetManager>());
    rclcpp::shutdown();
    return 0;
}