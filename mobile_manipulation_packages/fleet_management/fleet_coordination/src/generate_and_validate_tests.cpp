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
#include <random>
#include <omp.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "mobile_manipulation_interfaces_fleet/msg/fleet_paths.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/point32.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "mobile_manipulation_interfaces_fleet/action/path.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

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
};

struct TrajectoryPoint { 
    float time; 
    float x, y; 
};

// Contexto do Robô para Visualização e Publicação
struct RobotContext {
    int id;
    float speed;
    std::string frame_id;
    std::string label_text;
    
    // Publishers individuais
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pub_dest; 
    
    std::vector<TrajectoryPoint> path;
    geometry_msgs::msg::Pose final_pose; 
    
    // Estado atual
    float current_x = 0.0;
    float current_y = 0.0;
    float current_yaw = 0.0;
    bool is_moving = false;
};

// =============================================================================
// CLASSE UNIFICADA
// =============================================================================

class FleetCompleteNode : public rclcpp::Node {
public:
    FleetCompleteNode() : Node("fleet_complete_node"), gen_(rd_())
    {
        // ---------------------------------------------------------
        // 1. PARÂMETROS
        // ---------------------------------------------------------
        this->declare_parameter<double>("path_resolution", 0.05);       
        this->declare_parameter<double>("min_robot_gap", 0.5);          
        this->declare_parameter<double>("robot_radius", 0.3);           
        this->declare_parameter<double>("time_gap_tolerance", 2.0);     
        this->declare_parameter<double>("simulation_base_speed", 2.0);
        this->declare_parameter<int>("odom_rate_ms", 10); 
        this->declare_parameter<int>("viz_rate_ms", 50);  
        this->declare_parameter<int>("min_robot_count", 10);
        this->declare_parameter<int>("max_robot_count", 30);
        this->declare_parameter<double>("map_limit_x", 25.0); 
        this->declare_parameter<double>("map_limit_y", 25.0);
        this->declare_parameter<double>("min_travel_dist", 10.0);

        config_.resolution = this->get_parameter("path_resolution").as_double();
        config_.base_speed = this->get_parameter("simulation_base_speed").as_double();
        config_.min_robot_gap = this->get_parameter("min_robot_gap").as_double();
        config_.robot_radius = this->get_parameter("robot_radius").as_double();
        config_.time_gap_tolerance = this->get_parameter("time_gap_tolerance").as_double();
        
        odom_rate_ms_ = this->get_parameter("odom_rate_ms").as_int();
        viz_rate_ms_ = this->get_parameter("viz_rate_ms").as_int();
        decimals_ = count_decimals(config_.resolution);

        // ---------------------------------------------------------
        // 2. COMUNICAÇÃO
        // ---------------------------------------------------------

        // [Manager] Subscriber EXTERNO (caso venha de fora)
        sub_raw_scenario_ = this->create_subscription<mobile_manipulation_interfaces_fleet::msg::FleetPaths>(
            "/fleet/all_robot_plans", 
            10, 
            std::bind(&FleetCompleteNode::manager_input_callback, this, std::placeholders::_1));

        // [Visualizer] Publishers de Visualização
        pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/simulation_markers", 10);
        pub_zone_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/viz_zones", 10);
        pub_footprint_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/viz_static_paths", 10);

        // [Services & Actions]
        fleet_path_client_ = rclcpp_action::create_client<mobile_manipulation_interfaces_fleet::action::Path>(this, "multiple_paths");
        
        srv_generate_ = this->create_service<std_srvs::srv::Trigger>(
            "/fleet/generate_scenario",
            std::bind(&FleetCompleteNode::handle_generation_request, this, std::placeholders::_1, std::placeholders::_2));

        // ---------------------------------------------------------
        // 3. TIMERS
        // ---------------------------------------------------------
        
        // Timer rápido para Odometria
        odom_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(odom_rate_ms_), 
            std::bind(&FleetCompleteNode::odom_loop, this));

        // Timer lento para Markers e Destinations
        viz_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(viz_rate_ms_), 
            std::bind(&FleetCompleteNode::viz_loop, this));

        RCLCPP_INFO(this->get_logger(), "=== FLEET NODE (INTERNAL GEN) INICIADO ===");
        
        // Disparo inicial automático
        one_shot_timer_ = this->create_wall_timer(
            std::chrono::seconds(2),
            [this]() { 
                this->publish_random_scenario(); 
                this->one_shot_timer_->cancel(); 
            });
    }

private:
    struct FleetConfig {
        float resolution;
        float base_speed;
        float min_robot_gap;
        float robot_radius;
        float time_gap_tolerance;
    } config_;

    int decimals_;
    int odom_rate_ms_;
    int viz_rate_ms_;

    rclcpp::Subscription<mobile_manipulation_interfaces_fleet::msg::FleetPaths>::SharedPtr sub_raw_scenario_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_zone_markers_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_footprint_markers_;

    rclcpp_action::Client<mobile_manipulation_interfaces_fleet::action::Path>::SharedPtr fleet_path_client_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr srv_generate_;
    
    rclcpp::TimerBase::SharedPtr odom_timer_;
    rclcpp::TimerBase::SharedPtr viz_timer_;
    rclcpp::TimerBase::SharedPtr one_shot_timer_;

    std::unordered_map<GridKey, std::vector<RobotVisit>, GridKey::Hash> global_occupancy_map_;
    std::map<int, std::vector<std::pair<float, float>>> original_paths_;
    std::map<int, std::shared_ptr<RobotContext>> robots_;
    
    float sim_time_ = 0.0f;
    float max_sim_time_ = 0.0f;
    bool is_simulating_ = false;

    std::random_device rd_;
    std::mt19937 gen_;

    // =========================================================================
    // PARTE 1: GERADOR INTERNO
    // =========================================================================

    void handle_generation_request(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        (void)request;
        RCLCPP_INFO(this->get_logger(), "[Generator] Gerando novo cenario...");
        publish_random_scenario();
        response->success = true;
        response->message = "Cenario Gerado Internamente";
    }

    void publish_random_scenario() 
    {
        int min_r = this->get_parameter("min_robot_count").as_int();
        int max_r = this->get_parameter("max_robot_count").as_int();
        double limit_x = this->get_parameter("map_limit_x").as_double();
        double limit_y = this->get_parameter("map_limit_y").as_double();
        double min_dist = this->get_parameter("min_travel_dist").as_double();

        std::uniform_int_distribution<> count_dist(min_r, max_r);
        int total_robots = count_dist(gen_);

        std::uniform_real_distribution<double> pos_x_dist(-limit_x, limit_x);
        std::uniform_real_distribution<double> pos_y_dist(-limit_y, limit_y);

        auto msg = std::make_shared<mobile_manipulation_interfaces_fleet::msg::FleetPaths>();
        msg->header.frame_id = "world";
        msg->header.stamp = this->now();

        for (int i = 0; i < total_robots; ++i) {
            double sx, sy, gx, gy;
            int attempts = 0;
            do {
                sx = pos_x_dist(gen_); sy = pos_y_dist(gen_);
                gx = pos_x_dist(gen_); gy = pos_y_dist(gen_);
                attempts++;
            } while (std::hypot(gx - sx, gy - sy) < min_dist && attempts < 100);

            nav_msgs::msg::Path path;
            path.header = msg->header;
            
            geometry_msgs::msg::PoseStamped p1, p2;
            p1.pose.position.x = sx; p1.pose.position.y = sy; p1.pose.orientation.w = 1.0;
            p2.pose.position.x = gx; p2.pose.position.y = gy; p2.pose.orientation.w = 1.0;
            
            path.poses.push_back(p1);
            path.poses.push_back(p2);

            msg->robot_ids.push_back(i);
            msg->robot_speeds.push_back(config_.base_speed);
            msg->paths.push_back(path);
        }

        RCLCPP_INFO(this->get_logger(), "[Generator] Gerado: %d robos. Passando para Manager...", total_robots);
        
        // CHAMA O MANAGER DIRETAMENTE
        manager_input_callback(msg);
    }

    // =========================================================================
    // PARTE 2: MANAGER (LÓGICA)
    // =========================================================================

    void manager_input_callback(const mobile_manipulation_interfaces_fleet::msg::FleetPaths::SharedPtr msg)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        global_occupancy_map_.clear();
        original_paths_.clear();

        for (size_t i = 0; i < msg->paths.size(); ++i) 
        {
            int r_id = (i < msg->robot_ids.size()) ? msg->robot_ids[i] : (int)i;
            const auto& nav_path = msg->paths[i];
            if (nav_path.poses.empty()) continue;

            std::vector<std::pair<float, float>> raw_points;
            for (const auto& ps : nav_path.poses) {
                raw_points.push_back({(float)ps.pose.position.x, (float)ps.pose.position.y});
            }
            original_paths_[r_id] = raw_points;
            process_robot_volume(r_id, raw_points, config_.base_speed);
        }

        check_terminal_state_collisions(); 
        
        auto zones = detect_and_cluster_collisions();
        auto resolution = analyze_and_resolve_conflicts(zones);
        auto logs = resolution.first;

        print_report(zones, logs);
        publish_static_viz(zones);

        // ATUALIZA VISUALIZER INTERNAMENTE
        update_visualizer_state(msg, logs);

        auto t_end = std::chrono::high_resolution_clock::now();
        RCLCPP_INFO(this->get_logger(), "[Manager] Logica concluida em %.4fs", (std::chrono::duration<float>(t_end - t_start)).count());
    }

    // =========================================================================
    // PARTE 3: VISUALIZER
    // =========================================================================

    void update_visualizer_state(const mobile_manipulation_interfaces_fleet::msg::FleetPaths::SharedPtr original_msg, 
                                 const std::vector<ResolutionLog>& logs)
    {
        sim_time_ = 0.0f;
        max_sim_time_ = 0.0f;

        std::map<int, float> final_speeds;
        for(int id : original_msg->robot_ids) final_speeds[id] = config_.base_speed;
        
        for(const auto& log : logs) {
            if(final_speeds.count(log.r_victim)) {
                if (log.required_speed < final_speeds[log.r_victim]) {
                    final_speeds[log.r_victim] = log.required_speed;
                }
            }
        }

        for (size_t i = 0; i < original_msg->paths.size(); ++i) 
        {
            int r_id = (i < original_msg->robot_ids.size()) ? original_msg->robot_ids[i] : (int)i;
            const auto& nav_path = original_msg->paths[i];
            if (nav_path.poses.empty()) continue;

            float robot_speed = final_speeds[r_id];
            if (robot_speed < 0.01f) robot_speed = 0.01f;

            if (robots_.find(r_id) == robots_.end()) {
                auto ctx = std::make_shared<RobotContext>();
                ctx->id = r_id;
                ctx->frame_id = "robot_" + std::to_string(r_id) + "_base_link";
                std::stringstream ss; ss << "R" << r_id; ctx->label_text = ss.str();
                
                std::string odom_topic = "/robot_" + std::to_string(r_id) + "/odom";
                ctx->pub_odom = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);
                
                std::string dest_topic = "/robot_" + std::to_string(r_id) + "/destination";
                ctx->pub_dest = this->create_publisher<geometry_msgs::msg::Pose>(dest_topic, 10);
                
                robots_[r_id] = ctx;
            }

            auto& ctx = robots_[r_id];
            ctx->path.clear();
            ctx->speed = robot_speed;
            ctx->final_pose = nav_path.poses.back().pose;

            float current_t = 0.0f;
            float px = nav_path.poses[0].pose.position.x;
            float py = nav_path.poses[0].pose.position.y;
            ctx->path.push_back({current_t, px, py});

            for(size_t k = 0; k < nav_path.poses.size() - 1; ++k) {
                float ax = nav_path.poses[k].pose.position.x;
                float ay = nav_path.poses[k].pose.position.y;
                float bx = nav_path.poses[k+1].pose.position.x;
                float by = nav_path.poses[k+1].pose.position.y;
                float dist = std::hypot(bx-ax, by-ay);
                if (dist > 1e-6) {
                    float dt = dist / robot_speed;
                    current_t += dt;
                    ctx->path.push_back({current_t, bx, by});
                }
            }
            if(current_t > max_sim_time_) max_sim_time_ = current_t;
        }
        max_sim_time_ += 2.0f;
        is_simulating_ = true;
        RCLCPP_INFO(this->get_logger(), "[Visualizer] Simulacao reiniciada.");
    }

    void odom_loop() 
    {
        if(!is_simulating_) return;
        sim_time_ += (odom_rate_ms_ / 1000.0f);
        if(sim_time_ > max_sim_time_) sim_time_ = 0.0f; 

        auto now_ros = this->now();

        for(auto& [id, ctx] : robots_) 
        {
            if(ctx->path.empty()) continue;
            float x = ctx->path.back().x, y = ctx->path.back().y, yaw = 0.0f;
            bool moving = false;

            for(size_t i=0; i<ctx->path.size()-1; ++i) {
                if(sim_time_ >= ctx->path[i].time && sim_time_ <= ctx->path[i+1].time) {
                    float dt = ctx->path[i+1].time - ctx->path[i].time;
                    if(dt > 1e-5) {
                        float r = (sim_time_ - ctx->path[i].time)/dt;
                        x = ctx->path[i].x + r*(ctx->path[i+1].x - ctx->path[i].x);
                        y = ctx->path[i].y + r*(ctx->path[i+1].y - ctx->path[i].y);
                        yaw = std::atan2(ctx->path[i+1].y - ctx->path[i].y, ctx->path[i+1].x - ctx->path[i].x);
                    } else { x = ctx->path[i].x; y = ctx->path[i].y; }
                    moving = true; break;
                }
            }
            ctx->current_x = x; ctx->current_y = y; ctx->current_yaw = yaw; ctx->is_moving = moving;

            nav_msgs::msg::Odometry odom;
            odom.header.stamp = now_ros; odom.header.frame_id = "world"; odom.child_frame_id = ctx->frame_id;
            odom.pose.pose.position.x = x; odom.pose.pose.position.y = y;
            tf2::Quaternion q; q.setRPY(0, 0, yaw); odom.pose.pose.orientation = tf2::toMsg(q);
            odom.twist.twist.linear.x = moving ? ctx->speed : 0.0;
            ctx->pub_odom->publish(odom);
        }
    }

    void viz_loop()
    {
        if(!is_simulating_ || robots_.empty()) return;
        
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker del; del.action = 3; markers.markers.push_back(del);
        auto now_ros = this->now();

        for(const auto& [id, ctx] : robots_) {
            if(ctx->path.empty()) continue;
            
            visualization_msgs::msg::Marker mk;
            mk.header.frame_id = "world"; mk.header.stamp = now_ros;
            mk.ns = "simulated_robots"; mk.id = id; mk.type = 3; mk.action = 0;
            mk.pose.position.x = ctx->current_x; mk.pose.position.y = ctx->current_y; mk.pose.position.z = 0.2;
            mk.scale.x = 0.4; mk.scale.y = 0.4; mk.scale.z = 0.4;
            mk.color = get_color_for_id(id, 1.0);
            markers.markers.push_back(mk);

            visualization_msgs::msg::Marker txt = mk;
            txt.type = 9; txt.ns = "robot_ids"; txt.id = id+1000;
            txt.pose.position.z = 0.6; txt.scale.z = 0.3;
            txt.color.r=1; txt.color.g=1; txt.color.b=1; txt.text = ctx->label_text;
            markers.markers.push_back(txt);

            if(ctx->pub_dest) {
                ctx->pub_dest->publish(ctx->final_pose);
            }
        }
        pub_markers_->publish(markers);
    }

    // --- Helpers Internos ---
    void process_robot_volume(int robot_id, const std::vector<std::pair<float, float>>& waypoints, float speed) {
        if(waypoints.empty()) return;
        float current_time = 0.0f; 
        expand_point_to_map(robot_id, waypoints[0], current_time);
        for(size_t i = 0; i < waypoints.size() - 1; i++) {
            float dist = std::hypot(waypoints[i+1].first - waypoints[i].first, waypoints[i+1].second - waypoints[i].second);
            if (dist < 1e-6) continue;
            float t_dist = config_.resolution;
            float ux = (waypoints[i+1].first - waypoints[i].first) / dist; 
            float uy = (waypoints[i+1].second - waypoints[i].second) / dist;
            while (t_dist < dist) {
                float rx = waypoints[i].first + t_dist * ux; 
                float ry = waypoints[i].second + t_dist * uy;
                float time_at_point = current_time + (t_dist / speed);
                expand_point_to_map(robot_id, {rx, ry}, time_at_point);
                t_dist += config_.resolution;
            }
            current_time += (dist / speed);
            expand_point_to_map(robot_id, waypoints[i+1], current_time); 
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
                    p.first = round_to_multiple(p.first); p.second = round_to_multiple(p.second);
                    insert_visit(robot_id, p, timestamp);
                }
            }
            opa++; toma += config_.resolution;
        }
    }

    void insert_visit(int robot_id, std::pair<float, float> p, float time) {
        GridKey k = to_key(p.first, p.second);
        global_occupancy_map_[k].push_back({robot_id, time});
    }

    void check_terminal_state_collisions() {
        std::vector<std::pair<float, float>> all_goal_points;
        std::unordered_set<int> robots;
        for(const auto& [robot_id, path] : original_paths_) {
            if(path.empty()) continue;
            std::pair<float, float> goal = path.back();
            std::vector<GridKey> goal_footprint = get_footprint_keys(goal.first, goal.second, all_goal_points);
            std::set<int> conflicting_robots;
            for(const auto& key : goal_footprint) {
                if(global_occupancy_map_.count(key)) {
                    for(const auto& visit : global_occupancy_map_.at(key)) {
                        if(visit.robot_id != robot_id) {
                            conflicting_robots.insert(visit.robot_id); robots.insert(visit.robot_id);
                        }
                    }
                }
            }
            if(!conflicting_robots.empty()) RCLCPP_WARN(this->get_logger(), "Conflito terminal para Robo %d", robot_id);
        }
        if (!robots.empty()) {
            std::vector<int32_t> replan_ids;
            std::vector<geometry_msgs::msg::Point32> obs;
            for(const auto& p : all_goal_points) {
                geometry_msgs::msg::Point32 pt; pt.x = p.first; pt.y = p.second; pt.z = 0.0f; obs.push_back(pt);
            }
            for (int r_id : robots) {
                if (original_paths_.count(r_id) && !original_paths_[r_id].empty()) replan_ids.push_back(r_id);
            }
            if (!replan_ids.empty()) send_fleet_path_goal(replan_ids, obs);
        }
    }

    void send_fleet_path_goal(const std::vector<int32_t> & ids, const std::vector<geometry_msgs::msg::Point32> & obstacles) {
        if (!this->fleet_path_client_->wait_for_action_server(std::chrono::seconds(1))) {
            RCLCPP_ERROR(this->get_logger(), "Action server indisponivel."); return;
        }
        auto goal_msg = mobile_manipulation_interfaces_fleet::action::Path::Goal();
        goal_msg.robot_ids = ids; goal_msg.new_obstacles = obstacles;
        auto send_goal_options = rclcpp_action::Client<mobile_manipulation_interfaces_fleet::action::Path>::SendGoalOptions();
        send_goal_options.result_callback = [this, ids](const rclcpp_action::ClientGoalHandle<mobile_manipulation_interfaces_fleet::action::Path>::WrappedResult & result) {
            if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
                for(size_t i = 0; i < result.result->path.size(); i++) {
                    int id = ids[i];
                    if(!result.result->path[i].poses.empty()) {
                        std::vector<std::pair<float, float>> new_p;
                        for(const auto& ps : result.result->path[i].poses) new_p.push_back({(float)ps.pose.position.x, (float)ps.pose.position.y});
                        original_paths_[id] = new_p;
                    }
                }
                RCLCPP_INFO(this->get_logger(), "[Manager] Replanejamento recebido com sucesso.");
            }
        };
        this->fleet_path_client_->async_send_goal(goal_msg, send_goal_options);
    }

    std::vector<CollisionZone> detect_and_cluster_collisions() {
        std::vector<CollisionZone> zones;
        std::vector<GridKey> candidates;
        std::unordered_map<GridKey, std::vector<int>, GridKey::Hash> sig_cache;
        for(const auto& entry : global_occupancy_map_) {
            std::set<int> u; for(const auto& v : entry.second) u.insert(v.robot_id);
            if(u.size() > 1) {
                candidates.push_back(entry.first);
                sig_cache[entry.first] = std::vector<int>(u.begin(), u.end());
            }
        }
        std::unordered_set<GridKey, GridKey::Hash> visited; 
        std::unordered_set<GridKey, GridKey::Hash> candidate_set(candidates.begin(), candidates.end());
        int z_cnt = 0;
        for(const auto& start : candidates) {
            if(visited.count(start)) continue;
            CollisionZone z; z.id = z_cnt++;
            z.involved_robots.insert(sig_cache[start].begin(), sig_cache[start].end());
            std::vector<GridKey> q; q.push_back(start); visited.insert(start);
            while(!q.empty()) {
                GridKey curr = q.back(); q.pop_back();
                z.points.push_back(curr); z.points_set.insert(curr);
                int dx[]={1,-1,0,0}, dy[]={0,0,1,-1};
                for(int i=0; i<4; i++) {
                    GridKey n = {curr.x+dx[i], curr.y+dy[i]};
                    if(candidate_set.count(n) && !visited.count(n)) {
                        if(sig_cache[n] == sig_cache[start]) { visited.insert(n); q.push_back(n); }
                    }
                }
            }
            zones.push_back(z);
        }
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0; i<zones.size(); i++) {
            CollisionZone& z = zones[i];
            std::map<int, std::vector<float>> times;
            for(const auto& pt : z.points) {
                if(global_occupancy_map_.count(pt)) {
                    for(const auto& v : global_occupancy_map_.at(pt)) {
                        if(z.involved_robots.count(v.robot_id)) times[v.robot_id].push_back(v.timestamp);
                    }
                }
            }
            for(auto& [rid, t] : times) {
                if(t.empty()) continue;
                std::sort(t.begin(), t.end());
                float start = t[0], end = t[0];
                for(size_t k=1; k<t.size(); k++) {
                    if(t[k]-end > config_.time_gap_tolerance) { z.robot_intervals[rid].push_back({start, end}); start=t[k]; }
                    end=t[k];
                }
                z.robot_intervals[rid].push_back({start, end});
            }
        }
        return zones;
    }

    std::pair<std::vector<ResolutionLog>, std::map<int, float>> analyze_and_resolve_conflicts(std::vector<CollisionZone>& zones) {
        std::vector<ResolutionLog> logs;
        std::set<int> all_ids;
        for(const auto& z : zones) for(int id : z.involved_robots) all_ids.insert(id);
        struct Slot { float s, e; int owner; };
        std::map<int, std::vector<Slot>> table;

        for(int rid : all_ids) {
            float factor = 1.0f;
            bool changed = true;
            while(changed) {
                changed = false;
                for(const auto& z : zones) {
                    if(!z.robot_intervals.count(rid)) continue;
                    for(const auto& interval : z.robot_intervals.at(rid)) {
                        float dur = interval.second - interval.first;
                        float my_s = interval.first / factor;
                        float my_e = my_s + dur / factor;
                        if(table.count(z.id)) {
                            auto& slots = table[z.id];
                            std::sort(slots.begin(), slots.end(), [](auto& a, auto& b){ return a.s < b.s; });
                            for(const auto& s : slots) {
                                float os = std::max(my_s, s.s), oe = std::min(my_e, s.e);
                                if(os < oe + config_.min_robot_gap) {
                                    float req_arrival = s.e + config_.min_robot_gap;
                                    float new_factor = interval.first / req_arrival;
                                    if(new_factor < factor - 0.001) {
                                        ResolutionLog l; l.zone_id = z.id; l.r_victim = rid; l.r_priority = s.owner;
                                        l.required_speed = config_.base_speed * new_factor; l.victim_final_start = req_arrival;
                                        logs.push_back(l);
                                        factor = new_factor; changed = true; goto next_loop;
                                    }
                                }
                            }
                        }
                    }
                }
                next_loop:;
            }
            for(const auto& z : zones) {
                if(z.robot_intervals.count(rid)) {
                    for(const auto& interval : z.robot_intervals.at(rid)) {
                        float my_s = interval.first / factor;
                        float dur = (interval.second - interval.first) / factor;
                        table[z.id].push_back({my_s, my_s + dur, rid});
                    }
                }
            }
        }
        return {logs, {}};
    }

    void publish_static_viz(const std::vector<CollisionZone>& zones) {
        if(!pub_zone_markers_->get_subscription_count()) return;
        visualization_msgs::msg::MarkerArray m;
        visualization_msgs::msg::Marker del; del.action=3; m.markers.push_back(del);
        for(const auto& z : zones) {
            visualization_msgs::msg::Marker mk; mk.header.frame_id="world"; mk.ns="zones"; mk.id=z.id;
            mk.type=6; mk.action=0; mk.scale.x=config_.resolution; mk.scale.y=config_.resolution; mk.scale.z=config_.resolution;
            mk.color=get_color_for_id(z.id*10, 0.6);
            for(const auto& p:z.points) { geometry_msgs::msg::Point pt; pt.x=p.x*config_.resolution; pt.y=p.y*config_.resolution; pt.z=0.05; mk.points.push_back(pt); }
            m.markers.push_back(mk);
        }
        pub_zone_markers_->publish(m);
    }

    std_msgs::msg::ColorRGBA get_color_for_id(int id, float alpha) {
        std_msgs::msg::ColorRGBA c; c.a = alpha;
        float h = std::fmod(id * 0.618033988749895f, 1.0f);
        float s=0.8f, v=0.95f;
        int i=(int)(h*6); float f=h*6-i, p=v*(1-s), q=v*(1-f*s), t=v*(1-(1-f)*s);
        switch(i%6){ case 0:c.r=v;c.g=t;c.b=p;break; case 1:c.r=q;c.g=v;c.b=p;break; case 2:c.r=p;c.g=v;c.b=t;break;
                     case 3:c.r=p;c.g=q;c.b=v;break; case 4:c.r=t;c.g=p;c.b=v;break; case 5:c.r=v;c.g=p;c.b=q;break; }
        return c;
    }

    void print_report(const std::vector<CollisionZone>& zones, const std::vector<ResolutionLog>& logs) {
        std::cout << "\n========================================================\n";
        std::cout << ">>> RELATORIO DE TRAFEGO (DESTINATION MODE) <<<\n";
        std::cout << "Zonas de Colisao: " << zones.size() << "\n";
        if(logs.empty()) std::cout << "Fluxo Livre.\n";
        for(const auto& l : logs) {
            std::cout << "Zone " << l.zone_id << ": R" << l.r_victim << " reduz para " << std::fixed << std::setprecision(2) << l.required_speed << " m/s (Causa: R" << l.r_priority << ")\n";
        }
        std::cout << "========================================================\n\n";
    }

    std::vector<GridKey> get_footprint_keys(float x, float y, std::vector<std::pair<float, float>>& all_points) {
        std::vector<GridKey> keys;
        keys.push_back(to_key(x, y)); all_points.push_back({x,y});
        float t = 0.0; int o = 0;
        while(t <= config_.robot_radius) {
            for(int e=0; e<=o*2; e++) {
                std::vector<std::pair<float, float>> exp = {
                    {(x+t)-(config_.resolution*e), y+t}, {x+t, (y+t)-(config_.resolution*e)},
                    {x-t, (y-t)+(config_.resolution*e)}, {(x-t)+(config_.resolution*e), y-t}
                };
                for(auto& p : exp) { keys.push_back(to_key(p.first, p.second)); all_points.push_back(p); }
            }
            o++; t+=config_.resolution;
        }
        return keys;
    }

    inline float round_to_multiple(float v) {
        if(config_.resolution==0) return v;
        return std::round(std::round(v/config_.resolution)*config_.resolution * std::pow(10,decimals_))/std::pow(10,decimals_);
    }
    int count_decimals(float n) { 
        float f = std::fabs(n - std::floor(n)); int d=0; 
        while(f > 1e-9 && d<10) { f*=10; f-=std::floor(f); d++; } return d; 
    }
    GridKey to_key(float x, float y) { return { (int)std::round(x/config_.resolution), (int)std::round(y/config_.resolution) }; }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FleetCompleteNode>());
    rclcpp::shutdown();
    return 0;
}