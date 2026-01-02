// Inclusão de bibliotecas padrão
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

// ROS 2
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "mobile_manipulation_interfaces/msg/fleet_paths.hpp"

// TF2
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using std::placeholders::_1;
using Point2D = std::pair<float, float>;

// --- CONFIGURAÇÕES ---
constexpr double ROBOT_WIDTH = 0.60;
constexpr double LONGITUDINAL_PAD = 0.30;
constexpr double TIME_STEP = 0.02;
constexpr double ANIMATION_FREQ = 0.05;
constexpr double CLOUD_RES = 0.05;
constexpr double SIM_LIMIT = 40.0;
constexpr float SECURITY_DISTANCE = 0.3f;

// Parâmetros Híbridos (Ajustados para Robustez)
constexpr double APPROACH_DISTANCE = 0.5; // Distância curta de teste
constexpr double SAFETY_TIME_MARGIN = 0.2; 
constexpr double SAFETY_DIST_MARGIN = 0.2; 
constexpr double MIN_CRAWL_SPEED = 0.1;

// --- ESTRUTURAS ---
struct Vec2 {
    double x, y;
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
    double length() const { return std::hypot(x, y); }
};
using Rectangle = std::vector<Vec2>;

struct RobotState {
    geometry_msgs::msg::Pose pose;
    int segment_idx = -1;
    double yaw = 0.0;
};

struct MitigationData {
    bool active = false;
    bool stop_required = false;
    double target_speed = 0.0;
    double trigger_dist = 0.0;
    double stop_line = 0.0;
    int partner_id = -1; 
    double partner_exit_dist = 0.0;
};

struct SegmentTime {
    double start_t;
    double end_t;
};

struct RobotData {
    int id;
    std::string frame_id;
    double base_speed = 0.5;
    double current_dist = 0.0;
    double current_speed = 0.0;
    nav_msgs::msg::Path path;
    std::vector<Rectangle> static_rects;
    std::vector<SegmentTime> segment_times; 
    std::vector<double> accumulated_dists;  
    double total_length = 0.0;
    MitigationData mitigation;
};

struct RobotCollisionInfo {
    std::vector<int> colliding_with;
    std::vector<Rectangle> overlap_areas;
    double conflict_entry_dist = std::numeric_limits<double>::infinity();
    double conflict_exit_dist = -std::numeric_limits<double>::infinity();
    double max_other_exit_time = 0.0;
    std::unordered_map<int, double> specific_partner_exit_dists;
};

struct RGB { uint8_t r, g, b; };

class FleetManagement : public rclcpp::Node {
public:
    FleetManagement() : Node("fleet_traffic_manager") {
        sub_paths_ = create_subscription<mobile_manipulation_interfaces::msg::FleetPaths>(
            "/fleet/all_robot_plans", 10, std::bind(&FleetManagement::on_paths, this, _1));
        
        pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/fleet/debug_markers", 10);
        
        timer_ = create_wall_timer(
            std::chrono::duration<double>(ANIMATION_FREQ),
            std::bind(&FleetManagement::animate, this));
        
        RCLCPP_INFO(get_logger(), "Fleet Manager: Lógica de Parada Conservadora Ativa.");
    }

private:
    std::vector<RobotData> fleet_;
    std::unordered_map<int, RobotCollisionInfo> collision_data_;
    std::unordered_map<int, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> cloud_publishers_;
    std::unordered_map<int, RGB> robot_colors_;
    
    rclcpp::Subscription<mobile_manipulation_interfaces::msg::FleetPaths>::SharedPtr sub_paths_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::TimerBase::SharedPtr timer_;

    // --- HELPERS ---
    RobotData* get_robot_by_id(int id) {
        for (auto& r : fleet_) if (r.id == id) return &r;
        return nullptr;
    }

    RGB get_robot_color(int id) {
        static const std::vector<RGB> palette = {
            {255, 0, 0}, {0, 0, 255}, {0, 255, 0}, {255, 165, 0},
            {128, 0, 128}, {0, 255, 255}, {255, 0, 255}, {255, 255, 0}
        };
        return palette[std::abs(id) % palette.size()];
    }

    inline float round_to_multiple(float value, float multiple, int decimals) {
        if (multiple == 0.0f) return value; 
        float result = std::round(value / multiple) * multiple;
        float factor = std::pow(10.0f, decimals);
        return std::round(result * factor) / factor;
    }

    void ensure_publisher(int robot_id) {
        if (cloud_publishers_.find(robot_id) == cloud_publishers_.end()) {
            std::string topic = "/fleet/collision_cloud/robot_" + std::to_string(robot_id);
            cloud_publishers_[robot_id] = create_publisher<sensor_msgs::msg::PointCloud2>(topic, 10);
        }
    }

    // --- MATEMÁTICA ---
    Rectangle make_static_rect(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2) {
        double dx = p2.x - p1.x, dy = p2.y - p1.y;
        double len = std::hypot(dx, dy);
        if (len < 0.001) return {};
        Vec2 u = {dx/len, dy/len}, n = {u.y, -u.x};
        Vec2 s = {p1.x - u.x*LONGITUDINAL_PAD, p1.y - u.y*LONGITUDINAL_PAD};
        Vec2 e = {p2.x + u.x*LONGITUDINAL_PAD, p2.y + u.y*LONGITUDINAL_PAD};
        double hw = ROBOT_WIDTH / 2.0;
        return {{e.x+n.x*hw, e.y+n.y*hw}, {e.x-n.x*hw, e.y-n.y*hw}, {s.x-n.x*hw, s.y-n.y*hw}, {s.x+n.x*hw, s.y+n.y*hw}};
    }

    bool check_sat_intersection(const Rectangle& r1, const Rectangle& r2) {
        auto project = [](const Vec2& ax, const Rectangle& p, double& mn, double& mx) {
            mn = std::numeric_limits<double>::infinity(); mx = -std::numeric_limits<double>::infinity();
            for (auto& v : p) { double d = v.dot(ax); mn = std::min(mn, d); mx = std::max(mx, d); }
        };
        for (auto* poly : {&r1, &r2}) {
            for (size_t i = 0; i < poly->size(); ++i) {
                Vec2 edge = (*poly)[(i+1)%poly->size()] - (*poly)[i];
                Vec2 normal = {-edge.y, edge.x};
                double min1, max1, min2, max2;
                project(normal, r1, min1, max1);
                project(normal, r2, min2, max2);
                if (max1 < min2 || max2 < min1) return false;
            }
        }
        return true;
    }

    Rectangle get_intersection_polygon(const Rectangle& subjectPoly, const Rectangle& clipPoly) {
        Rectangle outputList = subjectPoly;
        for (size_t i = 0; i < clipPoly.size(); ++i) {
            Vec2 clipEdgeStart = clipPoly[i];
            Vec2 clipEdgeEnd = clipPoly[(i + 1) % clipPoly.size()];
            Rectangle inputList = outputList;
            outputList.clear();
            if (inputList.empty()) break;
            Vec2 edgeVec = clipEdgeEnd - clipEdgeStart;
            auto is_inside = [&](const Vec2& p) { return (edgeVec.x * (p.y - clipEdgeStart.y) - edgeVec.y * (p.x - clipEdgeStart.x)) >= 0; };
            auto intersection = [&](const Vec2& s, const Vec2& e) {
                double num = (clipEdgeStart.x - s.x) * (clipEdgeStart.y - clipEdgeEnd.y) - (clipEdgeStart.y - s.y) * (clipEdgeStart.x - clipEdgeEnd.x);
                double den = (clipEdgeStart.x - clipEdgeEnd.x) * (s.y - e.y) - (clipEdgeStart.y - clipEdgeEnd.y) * (s.x - e.x);
                double t = num / den;
                return Vec2{s.x + t * (e.x - s.x), s.y + t * (e.y - s.y)};
            };
            Vec2 S = inputList.back();
            for (const auto& E : inputList) {
                if (is_inside(E)) {
                    if (!is_inside(S)) outputList.push_back(intersection(S, E));
                    outputList.push_back(E);
                } else if (is_inside(S)) {
                    outputList.push_back(intersection(S, E));
                }
                S = E;
            }
        }
        return outputList;
    }

    double get_precise_exit_dist(const Rectangle& overlap, const RobotData& r, int seg_idx) {
        if (overlap.empty()) return r.accumulated_dists[seg_idx+1];
        auto p_start = r.path.poses[seg_idx].pose.position;
        auto p_end = r.path.poses[seg_idx+1].pose.position;
        Vec2 start = {p_start.x, p_start.y};
        Vec2 end = {p_end.x, p_end.y};
        Vec2 dir = end - start;
        double len = dir.length();
        if (len < 0.001) return r.accumulated_dists[seg_idx+1];
        Vec2 u = {dir.x / len, dir.y / len};
        double max_proj = -std::numeric_limits<double>::infinity();
        for (const auto& pt : overlap) {
            Vec2 v = pt - start;
            double proj = v.dot(u);
            if (proj > max_proj) max_proj = proj;
        }
        double segment_start_dist = r.accumulated_dists[seg_idx];
        return segment_start_dist + max_proj;
    }

    double get_precise_entry_dist(const Rectangle& overlap, const RobotData& r, int seg_idx) {
        if (overlap.empty()) return r.accumulated_dists[seg_idx];
        auto p_start = r.path.poses[seg_idx].pose.position;
        auto p_end = r.path.poses[seg_idx+1].pose.position;
        Vec2 start = {p_start.x, p_start.y};
        Vec2 end = {p_end.x, p_end.y};
        Vec2 dir = end - start;
        double len = dir.length();
        if (len < 0.001) return r.accumulated_dists[seg_idx];
        Vec2 u = {dir.x / len, dir.y / len};
        double min_proj = std::numeric_limits<double>::infinity();
        for (const auto& pt : overlap) {
            Vec2 v = pt - start;
            double proj = v.dot(u);
            if (proj < min_proj) min_proj = proj;
        }
        if (min_proj < 0) min_proj = 0; 
        double segment_start_dist = r.accumulated_dists[seg_idx];
        return segment_start_dist + min_proj;
    }

    // --- LÓGICA PRINCIPAL ---

    void on_paths(const mobile_manipulation_interfaces::msg::FleetPaths::SharedPtr msg) {
        fleet_.clear();
        collision_data_.clear();
        robot_colors_.clear();
        
        for (size_t i = 0; i < msg->paths.size(); ++i) {
            RobotData data;
            data.id = static_cast<int>(msg->robot_ids[i]);
            data.frame_id = msg->paths[i].header.frame_id;
            data.path = msg->paths[i];
            data.base_speed = (i < msg->robot_speeds.size()) ? msg->robot_speeds[i] : 0.5;
            data.current_dist = 0.0;
            data.current_speed = data.base_speed;
            
            if (data.path.poses.size() < 2) continue;
            robot_colors_[data.id] = get_robot_color(data.id);
            ensure_publisher(data.id);
            
            double current_t = 0.0;
            double current_d = 0.0;
            data.accumulated_dists.push_back(0.0);

            for (size_t j = 0; j < data.path.poses.size() - 1; ++j) {
                auto& p1 = data.path.poses[j].pose.position;
                auto& p2 = data.path.poses[j+1].pose.position;
                data.static_rects.push_back(make_static_rect(p1, p2));
                
                double seg_len = std::hypot(p2.x - p1.x, p2.y - p1.y);
                double seg_time = seg_len / data.base_speed;
                SegmentTime st; st.start_t = current_t; st.end_t = current_t + seg_time;
                data.segment_times.push_back(st);
                
                current_t += seg_time;
                current_d += seg_len;
                data.accumulated_dists.push_back(current_d);
            }
            data.total_length = current_d;
            fleet_.push_back(std::move(data));
        }
        
        check_conflicts();
        solve_conflicts_hybrid();
        publish_collision_clouds();
    }

    void check_conflicts() {
        for (size_t i = 0; i < fleet_.size(); ++i) {
            for (size_t k = i + 1; k < fleet_.size(); ++k) {
                RobotData& r1 = fleet_[i];
                RobotData& r2 = fleet_[k];

                for (size_t s1 = 0; s1 < r1.static_rects.size(); ++s1) {
                    for (size_t s2 = 0; s2 < r2.static_rects.size(); ++s2) {
                        
                        if (check_sat_intersection(r1.static_rects[s1], r2.static_rects[s2])) {
                            double start1 = r1.segment_times[s1].start_t;
                            double end1   = r1.segment_times[s1].end_t;
                            double start2 = r2.segment_times[s2].start_t;
                            double end2   = r2.segment_times[s2].end_t;

                            if (std::max(start1, start2) < std::min(end1, end2)) {
                                int id1 = r1.id; int id2 = r2.id;
                                
                                Rectangle overlap = get_intersection_polygon(r1.static_rects[s1], r2.static_rects[s2]);
                                
                                double precise_entry_r1 = get_precise_entry_dist(overlap, r1, s1);
                                double precise_exit_r1  = get_precise_exit_dist(overlap, r1, s1);
                                double precise_entry_r2 = get_precise_entry_dist(overlap, r2, s2);
                                double precise_exit_r2  = get_precise_exit_dist(overlap, r2, s2);

                                // R1
                                auto& info1 = collision_data_[id1];
                                info1.colliding_with.push_back(id2);
                                if (!overlap.empty()) info1.overlap_areas.push_back(overlap);

                                if (precise_entry_r1 < info1.conflict_entry_dist) info1.conflict_entry_dist = precise_entry_r1;
                                if (precise_exit_r1 > info1.conflict_exit_dist) info1.conflict_exit_dist = precise_exit_r1;
                                if (end2 > info1.max_other_exit_time) info1.max_other_exit_time = end2;
                                
                                if (info1.specific_partner_exit_dists.find(id2) == info1.specific_partner_exit_dists.end() || precise_exit_r2 > info1.specific_partner_exit_dists[id2]) {
                                    info1.specific_partner_exit_dists[id2] = precise_exit_r2;
                                }

                                // R2
                                auto& info2 = collision_data_[id2];
                                info2.colliding_with.push_back(id1);
                                if (!overlap.empty()) info2.overlap_areas.push_back(overlap);

                                if (precise_entry_r2 < info2.conflict_entry_dist) info2.conflict_entry_dist = precise_entry_r2;
                                if (precise_exit_r2 > info2.conflict_exit_dist) info2.conflict_exit_dist = precise_exit_r2;
                                if (end1 > info2.max_other_exit_time) info2.max_other_exit_time = end1;
                                
                                if (info2.specific_partner_exit_dists.find(id1) == info2.specific_partner_exit_dists.end() || precise_exit_r1 > info2.specific_partner_exit_dists[id1]) {
                                    info2.specific_partner_exit_dists[id1] = precise_exit_r1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void solve_conflicts_hybrid() {
        for (auto& [robot_id, info] : collision_data_) {
            if (info.colliding_with.empty()) continue;

            int my_id = robot_id;
            int partner_id = info.colliding_with[0]; 
            
            if (my_id < partner_id) {
                RobotData* myself = get_robot_by_id(my_id);
                
                double collision_start = info.conflict_entry_dist;
                
                // Onde começa a desaceleração
                double trigger_dist = collision_start - APPROACH_DISTANCE;
                if (trigger_dist < 0) trigger_dist = 0;

                // --- CÁLCULO DE TEMPO NECESSÁRIO (DELAY) ---
                
                // 1. Tempo que levaria pra chegar no conflito naturalmente (sem parar)
                double time_arrival_natural = collision_start / myself->base_speed;
                
                // 2. Tempo que o outro sai + Margem
                double time_other_clears = info.max_other_exit_time + SAFETY_TIME_MARGIN;
                
                // 3. Atraso necessário (segundos)
                double delay_needed = time_other_clears - time_arrival_natural;

                bool is_deadlock = false;
                double safe_speed = myself->base_speed;

                if (delay_needed > 0) {
                    // Preciso "queimar" delay_needed segundos enquanto percorro APPROACH_DISTANCE
                    // Tempo normal para percorrer APPROACH_DISTANCE:
                    double time_to_cross_approach_normal = APPROACH_DISTANCE / myself->base_speed;
                    
                    // Novo tempo total para cruzar = Normal + Delay
                    double total_time_available = time_to_cross_approach_normal + delay_needed;
                    
                    safe_speed = APPROACH_DISTANCE / total_time_available;
                    
                    // Se a velocidade for muito baixa OU se a distância for curta demais para ser seguro
                    if (safe_speed < MIN_CRAWL_SPEED || APPROACH_DISTANCE < 1.0) {
                        is_deadlock = true; // Para e espera
                        safe_speed = 0.0;
                    }
                }

                myself->mitigation.active = true;
                myself->mitigation.stop_required = is_deadlock;
                myself->mitigation.target_speed = safe_speed;
                myself->mitigation.trigger_dist = trigger_dist;
                myself->mitigation.stop_line = trigger_dist; // Para no inicio da zona se for deadlock
                myself->mitigation.partner_id = partner_id;
                myself->mitigation.partner_exit_dist = info.specific_partner_exit_dists[partner_id];
                
                if (is_deadlock) {
                    RCLCPP_WARN(get_logger(), "DEADLOCK: R%d para em %.2fm (Wait %.2fs).", my_id, trigger_dist, delay_needed);
                } else if (delay_needed > 0) {
                    RCLCPP_WARN(get_logger(), "SLOW: R%d reduz para %.2f m/s (Delay %.2fs).", my_id, safe_speed, delay_needed);
                }
            }
        }
    }

    // --- VISUALIZAÇÃO ---

    void expand_with_security_zone(std::vector<Point2D>& points) {
        if (points.empty()) return;
        std::set<Point2D> unique_points;
        float obstacle_dist = static_cast<float>(CLOUD_RES); 
        float maxSecurityDistance_ = SECURITY_DISTANCE;
        int decimals = 2;

        for (const auto& p : points) unique_points.insert(p);
        for (const auto& point : points) {
            float toma = 0.0f; int opa = 0;
            while (toma <= maxSecurityDistance_) {
                for (int eita = 0; eita <= opa * 2; eita++) {
                    unique_points.insert({round_to_multiple((point.first + toma) - (obstacle_dist * eita), obstacle_dist, decimals), round_to_multiple((point.second + toma), obstacle_dist, decimals)});
                    unique_points.insert({round_to_multiple((point.first + toma), obstacle_dist, decimals), round_to_multiple((point.second + toma) - (obstacle_dist * eita), obstacle_dist, decimals)});
                    unique_points.insert({round_to_multiple((point.first - toma), obstacle_dist, decimals), round_to_multiple((point.second - toma) + (obstacle_dist * eita), obstacle_dist, decimals)});
                    unique_points.insert({round_to_multiple((point.first - toma) + (obstacle_dist * eita), obstacle_dist, decimals), round_to_multiple((point.second - toma), obstacle_dist, decimals)});
                }
                opa++; toma += obstacle_dist;
            }
        }
        points.assign(unique_points.begin(), unique_points.end());
    }

    sensor_msgs::msg::PointCloud2 to_pointcloud2(const std::vector<Point2D>& points, const std::string& frame_id, const RGB& color, float z = 0.05f) {
        sensor_msgs::msg::PointCloud2 cloud;
        cloud.header.frame_id = frame_id;
        cloud.header.stamp = now();
        cloud.height = 1; cloud.width = points.size();
        cloud.is_dense = true; cloud.is_bigendian = false;
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(points.size());
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x"), iter_y(cloud, "y"), iter_z(cloud, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb(cloud, "rgb");
        for (size_t i = 0; i < points.size(); ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb) {
            *iter_x = points[i].first; *iter_y = points[i].second; *iter_z = z;
            iter_rgb[0] = color.r; iter_rgb[1] = color.g; iter_rgb[2] = color.b;
        }
        return cloud;
    }

    void publish_collision_clouds() {
        for (auto& [robot_id, info] : collision_data_) {
            std::vector<Point2D> pts;
            for(const auto& poly : info.overlap_areas) {
                double cx=0, cy=0; 
                for(auto& v:poly){ cx+=v.x; cy+=v.y; } 
                cx/=poly.size(); cy/=poly.size();
                for(double x=cx-0.2; x<=cx+0.2; x+=CLOUD_RES)
                    for(double y=cy-0.2; y<=cy+0.2; y+=CLOUD_RES) pts.emplace_back(x,y);
            }
            if (!pts.empty()) {
                expand_with_security_zone(pts);
                if (cloud_publishers_.count(robot_id)) {
                    const std::string& frame_id = fleet_[0].frame_id;
                    auto& color = robot_colors_[robot_id];
                    auto cloud_msg = to_pointcloud2(pts, frame_id, color, 0.05f);
                    cloud_publishers_[robot_id]->publish(cloud_msg);
                }
            }
        }
    }

    RobotState get_state_at_distance(const RobotData& robot, double dist) {
        RobotState state;
        if (dist >= robot.total_length) {
            if(!robot.path.poses.empty()) {
                state.pose = robot.path.poses.back().pose;
                state.pose.position.z = 0.15;
            }
            return state;
        }
        for (size_t i = 0; i < robot.accumulated_dists.size(); ++i) {
            if (i == 0) continue; 
            double d_end = robot.accumulated_dists[i];
            double d_start = robot.accumulated_dists[i-1];
            if (dist >= d_start && dist <= d_end) {
                double len = d_end - d_start;
                double ratio = (len > 0.001) ? (dist - d_start) / len : 0.0;
                auto& p1 = robot.path.poses[i-1].pose.position; 
                auto& p2 = robot.path.poses[i].pose.position;
                state.pose.position.x = p1.x + (p2.x - p1.x) * ratio;
                state.pose.position.y = p1.y + (p2.y - p1.y) * ratio;
                state.pose.position.z = 0.15;
                state.yaw = std::atan2(p2.y - p1.y, p2.x - p1.x);
                tf2::Quaternion q; q.setRPY(0, 0, state.yaw);
                state.pose.orientation = tf2::toMsg(q);
                return state;
            }
        }
        if(!robot.path.poses.empty()) { state.pose = robot.path.poses[0].pose; state.pose.position.z = 0.15; }
        return state;
    }

    void animate() {
        if (fleet_.empty()) return;
        
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker del;
        del.action = visualization_msgs::msg::Marker::DELETEALL;
        markers.markers.push_back(del);
        int marker_id = 0;
        
        bool all_finished = true;
        for(const auto& r : fleet_) {
            if (r.current_dist < r.total_length) all_finished = false;
        }
        
        if (all_finished) {
            for(auto& r : fleet_) r.current_dist = 0.0;
        }

        for (auto& robot : fleet_) {
            double target_speed = robot.base_speed;
            bool stop_visual = false;
            
            if (robot.mitigation.active) {
                bool partner_cleared = false;
                RobotData* partner = get_robot_by_id(robot.mitigation.partner_id);
                if (partner && partner->current_dist > robot.mitigation.partner_exit_dist + SAFETY_DIST_MARGIN) {
                    partner_cleared = true;
                }

                if (partner_cleared) {
                    target_speed = robot.base_speed; 
                }
                else if (robot.current_dist >= robot.mitigation.trigger_dist) {
                    if (robot.mitigation.stop_required) {
                        target_speed = 0.0;
                        if (robot.current_dist > robot.mitigation.stop_line) robot.current_dist = robot.mitigation.stop_line; 
                        stop_visual = true;
                    } else {
                        target_speed = robot.mitigation.target_speed;
                    }
                }
            }

            robot.current_speed = target_speed;
            if (robot.current_dist < robot.total_length) {
                double step = robot.current_speed * ANIMATION_FREQ;
                robot.current_dist += step;
            }

            auto state = get_state_at_distance(robot, robot.current_dist);
            auto& color = robot_colors_[robot.id];
            std::string id_str = std::to_string(robot.id);
            
            // Marker Robô
            visualization_msgs::msg::Marker bot;
            bot.header.frame_id = robot.frame_id;
            bot.header.stamp = now();
            bot.ns = "bot_" + id_str;
            bot.id = 0;
            bot.type = visualization_msgs::msg::Marker::CUBE;
            bot.action = visualization_msgs::msg::Marker::ADD;
            bot.pose = state.pose;
            bot.scale.x = bot.scale.y = ROBOT_WIDTH; bot.scale.z = 0.3;
            
            if (stop_visual) {
                bot.color.r = 1.0; bot.color.g = 0.0; bot.color.b = 0.0; bot.color.a = 1.0;
            } else if (robot.current_speed < robot.base_speed - 0.01) {
                bot.color.r = 1.0; bot.color.g = 0.5; bot.color.b = 0.0; bot.color.a = 1.0;
            } else {
                bot.color.r = color.r/255.0; bot.color.g = color.g/255.0; bot.color.b = color.b/255.0; bot.color.a = 0.8;
            }
            markers.markers.push_back(bot);
            
            // Texto Info
            visualization_msgs::msg::Marker txt;
            txt.header.frame_id = robot.frame_id;
            txt.header.stamp = now();
            txt.ns = "spd_" + id_str;
            txt.id = 0;
            txt.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            txt.action = visualization_msgs::msg::Marker::ADD;
            txt.pose = state.pose; txt.pose.position.z += 0.5;
            txt.scale.z = 0.2; txt.color.r=1.0; txt.color.g=1.0; txt.color.b=1.0; txt.color.a=1.0;
            std::stringstream ss;
            if (stop_visual) ss << "WAIT (R" << robot.mitigation.partner_id << ")";
            else ss << std::fixed << std::setprecision(2) << robot.current_speed << " m/s";
            txt.text = ss.str();
            markers.markers.push_back(txt);

            // Parede de Parada
            if (robot.mitigation.active && robot.mitigation.stop_required && !stop_visual && robot.current_dist < robot.mitigation.stop_line) {
                visualization_msgs::msg::Marker wall;
                wall.header.frame_id = robot.frame_id;
                wall.header.stamp = now();
                wall.ns = "wall_" + id_str;
                wall.id = 0;
                wall.type = visualization_msgs::msg::Marker::CUBE;
                wall.action = visualization_msgs::msg::Marker::ADD;
                wall.pose = get_state_at_distance(robot, robot.mitigation.stop_line).pose;
                wall.scale.x = 0.1; wall.scale.y = 1.0; wall.scale.z = 1.0;
                wall.color.r = 1.0; wall.color.a = 0.5;
                markers.markers.push_back(wall);
            }

            // --- CAMINHO VISUAL (Retângulos) ---
            for (size_t r = 0; r < robot.static_rects.size(); ++r) {
                visualization_msgs::msg::Marker rect;
                rect.header.frame_id = robot.frame_id;
                rect.header.stamp = now();
                rect.ns = "path_" + id_str;
                rect.id = marker_id++;
                rect.type = visualization_msgs::msg::Marker::LINE_STRIP;
                rect.action = visualization_msgs::msg::Marker::ADD;
                rect.scale.x = 0.02;
                
                // Determina se este segmento é parte da zona de perigo
                bool is_conflict_segment = false;
                if (collision_data_.count(robot.id)) {
                    auto& info = collision_data_[robot.id];
                    double d_start = (r==0) ? 0.0 : robot.accumulated_dists[r];
                    double d_end = robot.accumulated_dists[r+1];
                    // Se o segmento está dentro do intervalo global de conflito
                    if (d_end > info.conflict_entry_dist && d_start < info.conflict_exit_dist) {
                        is_conflict_segment = true;
                    }
                }

                if (is_conflict_segment) {
                    rect.color.r = 1.0; rect.color.g = 0.0; rect.color.b = 0.0; rect.color.a = 0.3; // Vermelho Translúcido
                    rect.scale.x = 0.04;
                } else {
                    rect.color.r = color.r/255.0; rect.color.g = color.g/255.0; rect.color.b = color.b/255.0; rect.color.a = 0.1; 
                }
                
                for (auto& v : robot.static_rects[r]) {
                    geometry_msgs::msg::Point p; p.x=v.x; p.y=v.y; rect.points.push_back(p);
                }
                rect.points.push_back(rect.points[0]);
                markers.markers.push_back(rect);
            }
        }
        pub_markers_->publish(markers);
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FleetManagement>());
    rclcpp::shutdown();
    return 0;
}