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
#include <omp.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav_msgs/msg/path.hpp"
#include "mobile_manipulation_interfaces_fleet/msg/fleet_paths.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "mobile_manipulation_interfaces_fleet/action/path.hpp"
#include <navigation/FleetManagementAStar.hpp>

// =============================================================================
// ESTRUTURAS AUXILIARES
// =============================================================================

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
            if(!intervals.empty()) {
                if (intervals[0].first < min_t) min_t = intervals[0].first;
            }
        }
        return min_t;
    }
};

// =============================================================================
// CLASSE FLEET MANAGER (LÓGICA)
// =============================================================================

class FleetManager : public rclcpp::Node {
public:
    FleetManager(
        std::shared_ptr<navigation::FleetManagementAStar> fleet_management_a_star_node
    ) 
    : Node("fleet_manager"),
    fleet_management_a_star_node_(fleet_management_a_star_node)
    {
        // Declaração de Parâmetros de Lógica
        this->declare_parameter<double>("path_resolution", 0.05);       
        this->declare_parameter<double>("simulation_base_speed", 5.0);  
        this->declare_parameter<double>("min_robot_gap", 1.5);          
        this->declare_parameter<double>("robot_radius", 0.3);           
        this->declare_parameter<double>("time_gap_tolerance", 2.0);     

        // Leitura de Parâmetros
        config_.resolution = static_cast<float>(this->get_parameter("path_resolution").as_double());
        config_.base_speed = static_cast<float>(this->get_parameter("simulation_base_speed").as_double());
        config_.min_robot_gap = static_cast<float>(this->get_parameter("min_robot_gap").as_double());
        config_.robot_radius = static_cast<float>(this->get_parameter("robot_radius").as_double());
        config_.time_gap_tolerance = static_cast<float>(this->get_parameter("time_gap_tolerance").as_double());

        decimals = count_decimals(config_.resolution);

        // Clients
        fleet_path_client = rclcpp_action::create_client<mobile_manipulation_interfaces_fleet::action::Path>(this, "multiple_paths");
        scenario_client_ = this->create_client<std_srvs::srv::Trigger>("/fleet/generate_scenario");
            
        RCLCPP_INFO(this->get_logger(), "Fleet Manager Iniciado (Logic Node).");

        odom_timer_ = this->create_wall_timer(std::chrono::milliseconds(20), std::bind(&FleetManager::check_terminal_state_collisions, this));

        
     
        request_new_scenario();
    }

private:
    struct FleetConfig {
        float resolution;
        float base_speed;
        float min_robot_gap;
        float robot_radius;
        float time_gap_tolerance;
    } config_;

    int decimals;
    std::shared_ptr<navigation::FleetManagementAStar> fleet_management_a_star_node_;

    rclcpp::TimerBase::SharedPtr odom_timer_;

    rclcpp::Subscription<mobile_manipulation_interfaces_fleet::msg::FleetPaths>::SharedPtr sub_fleet_; 
    
    // Publisher para os planos validados com velocidades ajustadas
    rclcpp::Publisher<mobile_manipulation_interfaces_fleet::msg::FleetPaths>::SharedPtr pub_plans_;

    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr scenario_client_;
    rclcpp_action::Client<mobile_manipulation_interfaces_fleet::action::Path>::SharedPtr fleet_path_client;
    rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces_fleet::action::Path>::SharedPtr active_fleet_goal_handle_;

    std::unordered_map<GridKey, std::vector<RobotVisit>, GridKey::Hash> global_occupancy_map;
    std::map<int, std::vector<std::pair<float, float>>> original_paths_;

    bool sent_goal = false;

    inline float round_to_multiple(float value, float multiple, int decimals) {
        if (multiple == 0.0) return value; 
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0, decimals);
        return std::round(result * factor) / factor;
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

    GridKey to_key(float x, float y) {
        return { static_cast<int>(std::round(x / config_.resolution)), static_cast<int>(std::round(y / config_.resolution)) };
    }


    void publish_adjusted_plans(const std::vector<ResolutionLog>& logs)
    {
        mobile_manipulation_interfaces_fleet::msg::FleetPaths output_msg;

        // 1. Calcula velocidades finais
        // Inicializa todos com a velocidade base
        std::map<int, float> final_speeds;
        for(const auto& [rid, path] : original_paths_) {
            final_speeds[rid] = config_.base_speed;
        }

        // Aplica restrições dos logs (escolhe sempre a menor velocidade exigida)
        for(const auto& log : logs) {
            if(final_speeds.count(log.r_victim)) {
                // Se o log pede 1.5 e o atual é 2.0, vira 1.5. 
                // Se outro log pede 1.0, vira 1.0.
                if (log.required_speed < final_speeds[log.r_victim]) {
                    final_speeds[log.r_victim] = log.required_speed;
                }
            }
        }

        // 2. Monta a mensagem
        for(const auto& [rid, path_vec] : original_paths_) 
        {
            if(path_vec.empty()) continue;

            output_msg.robot_ids.push_back(rid);
            
            // ATENÇÃO: Assume-se que o .msg tem o campo 'float32[] velocities'
            output_msg.robot_speeds.push_back(final_speeds[rid]); 

            // Reconstrói nav_msgs::Path
            nav_msgs::msg::Path ros_path;
            ros_path.header.frame_id = "world";
            ros_path.header.stamp = this->now();
            
            for(const auto& pt : path_vec) {
                geometry_msgs::msg::PoseStamped ps;
                ps.header = ros_path.header;
                ps.pose.position.x = pt.first;
                ps.pose.position.y = pt.second;
                ps.pose.position.z = 0.0;
                // Orientação padrão (identidade)
                ps.pose.orientation.w = 1.0; 
                ros_path.poses.push_back(ps);
            }
            output_msg.paths.push_back(ros_path);
        }

        pub_plans_->publish(output_msg);
    }


    void request_new_scenario() {
        if (!scenario_client_->wait_for_service(std::chrono::seconds(1))) {
            RCLCPP_WARN(this->get_logger(), "Gerador de cenarios nao disponivel.");
            return;
        }
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        scenario_client_->async_send_request(request, [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
            try {
                auto response = future.get();
                if(response->success) RCLCPP_INFO(this->get_logger(), "Novo cenario solicitado com sucesso!");
                else RCLCPP_WARN(this->get_logger(), "Gerador recusou: %s", response->message.c_str());
            } catch (const std::exception &e) {
                RCLCPP_ERROR(this->get_logger(), "Falha na chamada do servico: %s", e.what());
            }
        });
    }

    std::vector<GridKey> get_footprint_keys(float x, float y, std::vector<std::pair<float, float>>& all_points) {
        std::vector<GridKey> keys;
        keys.push_back(to_key(x, y));
        all_points.push_back(std::make_pair(x, y));
        float toma = 0.0; int opa = 0;
        while(toma <= config_.robot_radius) {
            for(int eita = 0; eita <= opa * 2; eita++) {   
                std::vector<std::pair<float, float>> expansion = {
                    { (x + toma) - (config_.resolution * eita), (y + toma) },
                    { (x + toma), (y + toma) - (config_.resolution * eita) },
                    { (x - toma), (y - toma) + (config_.resolution * eita) },
                    { (x - toma) + (config_.resolution * eita), (y - toma) }
                };
                for(auto& p : expansion) {
                    keys.push_back(to_key(p.first, p.second));
                    all_points.push_back(p);
                }
            }
            opa++; toma += config_.resolution;
        }
        return keys;
    }

    void check_terminal_state_collisions() 
    {
        // RCLCPP_INFO(this->get_logger(), "--- VERIFICACAO DE DESTINOS ---");
        
        std::vector<std::pair<float, float>> all_goal_points;
        std::unordered_set<int> robots;

        // 1. Identifica onde cada robô quer chegar e cria a lista de obstáculos
        for(const auto& [robot_id, path] : original_paths_) 
        {
            if(path.empty()) continue;
            
            std::pair<float, float> goal = path.back();
            // Preenche all_goal_points com o footprint do destino
            std::vector<GridKey> goal_footprint = get_footprint_keys(goal.first, goal.second, all_goal_points);
            
            std::set<int> conflicting_robots;
            
            // Verifica se a posição final colide com alguém no mapa atual
            for(const auto& key : goal_footprint) 
            {
                if(global_occupancy_map.count(key)) 
                {
                    for(const auto& visit : global_occupancy_map.at(key)) 
                    {
                        if(visit.robot_id != robot_id) 
                        {
                            conflicting_robots.insert(visit.robot_id);
                            robots.insert(visit.robot_id);
                        }
                    }
                }
            }

            if(!conflicting_robots.empty()) 
            {
                std::stringstream ss;
                ss << "PERIGO CRITICO: O destino do Robo " << robot_id 
                   << " em (" << goal.first << ", " << goal.second << ") colide com robos: ";
                for(int id : conflicting_robots) ss << id << " ";
                RCLCPP_ERROR(this->get_logger(), "%s", ss.str().c_str());
            }
        }

        // 2. Se houver robôs em conflito, solicita replanejamento
        if (!robots.empty() && sent_goal == false) 
        {
            std::vector<int32_t> replan_ids;
            
            // Prepara os obstáculos (destinos dos outros robôs)
            std::vector<geometry_msgs::msg::Point32> dynamic_obstacles; 
            dynamic_obstacles.reserve(all_goal_points.size());
            
            for(const auto& p : all_goal_points) 
            {
                geometry_msgs::msg::Point32 pt;
                pt.x = p.first; 
                pt.y = p.second; 
                pt.z = 0.0f; 
                dynamic_obstacles.push_back(pt);
            }

            // Seleciona os robôs que precisam replanejar
            for (int r_id : robots) 
            {
                if (original_paths_.count(r_id) && !original_paths_[r_id].empty()) 
                {
                    replan_ids.push_back(r_id);
                    // NOTA: Não precisamos mais montar o vetor de Poses aqui,
                    // pois o Server já tem o destino salvo no mapa dele.
                }
            }

            // Envia a action apenas com IDs e Obstáculos
            if (!replan_ids.empty()) 
            {
                RCLCPP_WARN(this->get_logger(), 
                    "Enviando replanejamento para %zu robos com %zu novos obstaculos (Point32).", 
                    replan_ids.size(), dynamic_obstacles.size());
                
                // send_fleet_path_goal(replan_ids, dynamic_obstacles);
            }
        }
    }

   
    std::vector<CollisionZone> detect_and_cluster_collisions() {
        auto t_start_total = std::chrono::high_resolution_clock::now();
        std::vector<CollisionZone> zones;
        std::vector<GridKey> collision_candidates;
        std::unordered_map<GridKey, std::vector<int>, GridKey::Hash> signature_cache;

        for(const auto& entry : global_occupancy_map) {
            std::set<int> unique_check;
            for(const auto& visit : entry.second) unique_check.insert(visit.robot_id);
            if(unique_check.size() > 1) {
                collision_candidates.push_back(entry.first);
                signature_cache[entry.first] = std::vector<int>(unique_check.begin(), unique_check.end());
            }
        }

        std::unordered_set<GridKey, GridKey::Hash> visited_keys; 
        std::unordered_set<GridKey, GridKey::Hash> candidate_set(collision_candidates.begin(), collision_candidates.end());
        int zone_counter = 0;

        for(const auto& start_key : collision_candidates) {
            if(visited_keys.count(start_key)) continue;
            const std::vector<int>& current_sig = signature_cache[start_key];
            CollisionZone current_zone;
            current_zone.id = zone_counter++;
            current_zone.involved_robots.insert(current_sig.begin(), current_sig.end());
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
                        if(signature_cache[neighbor] == current_sig) {
                            visited_keys.insert(neighbor); queue.push_back(neighbor);
                        }
                    }
                }
            }
            zones.push_back(current_zone);
        }

        #pragma omp parallel for schedule(dynamic)
        for(size_t z_idx = 0; z_idx < zones.size(); z_idx++) {
            CollisionZone& zone = zones[z_idx];
            std::map<int, std::vector<float>> raw_times;
            for(const auto& pt : zone.points) {
                auto it = global_occupancy_map.find(pt);
                if (it != global_occupancy_map.end()) {
                    for(const auto& v : it->second) {
                        if(zone.involved_robots.count(v.robot_id)) raw_times[v.robot_id].push_back(v.timestamp);
                    }
                }
            }
            for(auto& [rid, times] : raw_times) {
                if(times.empty()) continue;
                std::sort(times.begin(), times.end());
                float start_t = times[0]; float end_t = times[0];
                for(size_t i = 1; i < times.size(); i++) {
                    if((times[i] - end_t) > config_.time_gap_tolerance) {
                        zone.robot_intervals[rid].push_back({start_t, end_t}); start_t = times[i];
                    }
                    end_t = times[i];
                }
                zone.robot_intervals[rid].push_back({start_t, end_t});
            }
        }
        auto t_end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = t_end_total - t_start_total;
        if (!collision_candidates.empty()) RCLCPP_INFO(this->get_logger(), "Collision Detection (Optimized): %.6f s | Zones Found: %zu", duration.count(), zones.size());
        return zones;
    }

    std::pair<std::vector<ResolutionLog>, std::map<int, float>> analyze_and_resolve_conflicts(std::vector<CollisionZone>& zones) {
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
                                        log.zone_id = zone.id; log.step_number = log_step++; log.r_priority = slot.owner_id; log.r_victim = current_robot;
                                        log.priority_end = slot.end; log.victim_final_start = required_arrival; log.dist_traveled = dist;
                                        log.required_speed = req_speed; log.is_physical_violation = false; 
                                        logs.push_back(log);
                                        speed_factor = new_factor; factor_changed = true; goto restart_checks;
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

    std::pair<float, std::pair<float, float>> calculate_approach_metrics(int robot_id, const CollisionZone& zone) {
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
                float cur_x = ax + t_dist * ux; float cur_y = ay + t_dist * uy;
                GridKey k = to_key(cur_x, cur_y);
                if (zone.points_set.count(k)) return {total_dist_traveled + t_dist, {cur_x, cur_y}};
                t_dist += config_.resolution;
            }
            total_dist_traveled += dist;
        }
        return {total_dist_traveled, path.back()}; 
    }

    void print_super_detailed_report(
        const std::vector<CollisionZone>& zones, 
        const std::vector<ResolutionLog>& logs,
        const std::map<int, float>& final_delays) 
    {
        std::cout << "\n\n";
        std::cout << "================================================================================\n";
        std::cout << "||   RELATÓRIO DE TRÁFEGO (VELOCIDADE VARIÁVEL REAL)   ||\n";
        std::cout << "================================================================================\n";
        std::cout << ">>> LOG DE RESOLUÇÃO:\n";
        if(logs.empty()) std::cout << "    [Fluxo livre.]\n";
        for(const auto& log : logs) {
            std::cout << "    +--------------------------------------------------------------------------+\n";
            std::cout << "    | ZONA " << log.zone_id << " (Evento #" << log.step_number << ")\n";
            std::cout << "    | [CONFLITO]   Robo " << log.r_victim << " bateria no Robo " << log.r_priority << ".\n";
            std::cout << "    | [SOLUÇÃO]    Reduzir velocidade para chegar em " << std::fixed << std::setprecision(2) << log.victim_final_start << "s.\n";
            std::cout << "    |              - VELOCIDADE NECESSARIA: >>> " << log.required_speed << " m/s <<<\n";
            std::cout << "    +--------------------------------------------------------------------------+\n";
        }
        std::cout << "================================================================================\n\n";
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions fleet_management_a_star_opts;
    fleet_management_a_star_opts.arguments({"--ros-args", "-r", "__node:=fleet_management_a_star_node"});


    std::shared_ptr<navigation::FleetManagementAStar> fleet_management_a_star_node = nullptr;


    rclcpp::executors::MultiThreadedExecutor executor;


    fleet_management_a_star_node = std::make_shared<navigation::FleetManagementAStar>(fleet_management_a_star_opts);
    executor.add_node(fleet_management_a_star_node);

    auto server_node = std::make_shared<FleetManager>(fleet_management_a_star_node);

    executor.add_node(server_node);

    executor.spin();

    rclcpp::shutdown();
    return 0;
}