#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <cmath>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "slam_interfaces/msg/gtsam_data.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h> 
#include <gtsam/nonlinear/ISAM2.h>

#include <Eigen/Dense>

struct FrameProcessResult
{
    gtsam::Pose3 delta_base;
    gtsam::SharedNoiseModel visual_noise;
    gtsam::Pose3 estimate;
};

struct RobotSlamState
{
    gtsam::ISAM2 isam2;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial_estimates;
    gtsam::Values optimized_estimates;
    
    int keyframe_id = 0;
    bool has_keyframe = false;
    std::mutex gtsam_mutex;

    bool has_gt = false;
    bool first_gt_received = false;
    gtsam::Pose3 initial_gt_pose;
    gtsam::Pose3 latest_gt_pose;
    gtsam::Pose3 previous_gt_pose;
    double total_gt_distance = 0.0;
};

class GtsamOptimizationNode : public rclcpp::Node
{
public:
    GtsamOptimizationNode() : Node("gtsam_optimization_node")
    {
        this->declare_parameter("num_robots", 1);
        this->declare_parameter("use_ground_truth", true);
        
        int num_robots = this->get_parameter("num_robots").as_int();
        use_ground_truth_ = this->get_parameter("use_ground_truth").as_bool();

        RCLCPP_INFO(this->get_logger(), "Iniciando No GTSAM para %d robo(s)...", num_robots);
        if(use_ground_truth_) RCLCPP_INFO(this->get_logger(), "Comparacao com Ground Truth ATIVADA.");

        robot_states_.reserve(num_robots);
        odom_pubs_.reserve(num_robots);
        graph_markers_pubs_.reserve(num_robots);
        path_pubs_.reserve(num_robots);

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        for (int i = 0; i < num_robots; ++i) 
        {
            robot_states_.push_back(std::make_unique<RobotSlamState>());
        }

        subs_.reserve(num_robots);
        gt_subs_.reserve(num_robots);
        cb_groups_.reserve(num_robots);

        for (int i = 0; i < num_robots; ++i)
        {
            auto cb_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            cb_groups_.push_back(cb_group);

            rclcpp::SubscriptionOptions sub_options;
            sub_options.callback_group = cb_group;

            
            std::string factor_topic = "/robot_" + std::to_string(i) + "/slam/camera_factors";
            auto factor_sub = this->create_subscription<slam_interfaces::msg::GtsamData>(
                factor_topic, 10,
                [this, i](const slam_interfaces::msg::GtsamData::SharedPtr msg) {
                    this->factor_callback(i, msg); 
                },
                sub_options
            );
            subs_.push_back(factor_sub);

           
            if (use_ground_truth_)
            {
                std::string gt_topic = "/robot_" + std::to_string(i) + "/ground_truth";
                auto gt_sub = this->create_subscription<nav_msgs::msg::Odometry>(
                    gt_topic, 10,
                    [this, i](const nav_msgs::msg::Odometry::SharedPtr msg) {
                        this->ground_truth_callback(i, msg);
                    },
                    sub_options
                );
                gt_subs_.push_back(gt_sub);
            }

            std::string odom_topic = "/robot_" + std::to_string(i) + "/odom";
            std::string marker_topic = "/robot_" + std::to_string(i) + "/gtsam_graph";
            std::string path_topic = "/robot_" + std::to_string(i) + "/gtsam_path";

            odom_pubs_.push_back(this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10));
            graph_markers_pubs_.push_back(this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic, 10));
            path_pubs_.push_back(this->create_publisher<nav_msgs::msg::Path>(path_topic, 10));
        }
    }

private:
    std::vector<std::unique_ptr<RobotSlamState>> robot_states_;
    std::vector<rclcpp::Subscription<slam_interfaces::msg::GtsamData>::SharedPtr> subs_;
    std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr> gt_subs_;
    
    std::vector<rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr> odom_pubs_;
    std::vector<rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr> graph_markers_pubs_;
    std::vector<rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr> path_pubs_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::vector<rclcpp::CallbackGroup::SharedPtr> cb_groups_;
    bool use_ground_truth_;

    void publish_gtsam_data(int robot_id, const gtsam::Pose3& optimized_pose, const rclcpp::Time& stamp)
    {
        auto& state = *robot_states_[robot_id];
        
        std::string odom_frame = "robot_" + std::to_string(robot_id) + "/odom";
        std::string main_frame_id = "base_link";

        try 
        {
            if (!state.optimized_estimates.exists(gtsam::symbol_shorthand::X(state.keyframe_id))) return;
            gtsam::Matrix6 covariance_gtsam = state.isam2.marginalCovariance(gtsam::symbol_shorthand::X(state.keyframe_id));
            
            nav_msgs::msg::Odometry odom_msg;
            odom_msg.header.stamp = stamp;
            odom_msg.header.frame_id = odom_frame;        
            odom_msg.child_frame_id = main_frame_id;    

            gtsam::Pose3 base_pose = optimized_pose;
            
            odom_msg.pose.pose.position.x = base_pose.x();
            odom_msg.pose.pose.position.y = base_pose.y();
            odom_msg.pose.pose.position.z = base_pose.z();
            
            Eigen::Quaterniond q(base_pose.rotation().matrix());
            odom_msg.pose.pose.orientation.x = q.x();
            odom_msg.pose.pose.orientation.y = q.y();
            odom_msg.pose.pose.orientation.z = q.z();
            odom_msg.pose.pose.orientation.w = q.w();
            
            geometry_msgs::msg::TransformStamped t;
            t.header.stamp = stamp;
            t.header.frame_id = odom_frame;
            t.child_frame_id = main_frame_id;
            t.transform.translation.x = base_pose.x();
            t.transform.translation.y = base_pose.y();
            t.transform.translation.z = base_pose.z();
            t.transform.rotation.x = q.x();
            t.transform.rotation.y = q.y();
            t.transform.rotation.z = q.z();
            t.transform.rotation.w = q.w();

            if (tf_broadcaster_) {
                tf_broadcaster_->sendTransform(t);
            }
           
            for (int i = 0; i < 3; ++i) 
            {
                for (int j = 0; j < 3; ++j) 
                {
                    odom_msg.pose.covariance[i * 6 + j] = covariance_gtsam(i + 3, j + 3);
                    odom_msg.pose.covariance[(i + 3) * 6 + (j + 3)] = covariance_gtsam(i, j);
                    odom_msg.pose.covariance[i * 6 + (j + 3)] = covariance_gtsam(i + 3, j);
                    odom_msg.pose.covariance[(i + 3) * 6 + j] = covariance_gtsam(i, j + 3);
                }
            }

            odom_pubs_[robot_id]->publish(odom_msg);

            visualization_msgs::msg::MarkerArray marker_array;
            nav_msgs::msg::Path path_msg;
            path_msg.header.stamp = stamp;
            path_msg.header.frame_id = odom_frame; 

            visualization_msgs::msg::Marker nodes_marker;
            nodes_marker.header.frame_id = odom_frame; 
            nodes_marker.header.stamp = stamp;
            nodes_marker.ns = "gtsam_nodes";
            nodes_marker.id = 0;
            nodes_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            nodes_marker.action = visualization_msgs::msg::Marker::ADD;
            nodes_marker.scale.x = 0.05;
            nodes_marker.scale.y = 0.05;
            nodes_marker.scale.z = 0.05;
            nodes_marker.color.a = 1.0;
            nodes_marker.color.r = 0.0;
            nodes_marker.color.g = 1.0;
            nodes_marker.color.b = 0.0;

            visualization_msgs::msg::Marker edges_marker;
            edges_marker.header.frame_id = odom_frame; 
            edges_marker.header.stamp = stamp;
            edges_marker.ns = "gtsam_edges";
            edges_marker.id = 1;
            edges_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
            edges_marker.action = visualization_msgs::msg::Marker::ADD;
            edges_marker.scale.x = 0.02;
            edges_marker.color.a = 1.0;
            edges_marker.color.r = 1.0;
            edges_marker.color.g = 0.0;
            edges_marker.color.b = 0.0;

            for (const auto& key_value : state.optimized_estimates) 
            {
                gtsam::Symbol sym(key_value.key);
                if (sym.chr() != 'x') continue;

                gtsam::Pose3 node_base_pose = key_value.value.cast<gtsam::Pose3>();

                geometry_msgs::msg::Point p;
                p.x = node_base_pose.x();
                p.y = node_base_pose.y();
                p.z = node_base_pose.z();
                nodes_marker.points.push_back(p);

                geometry_msgs::msg::PoseStamped path_pose;
                path_pose.header.frame_id = odom_frame; 
                path_pose.pose.position = p;
                path_msg.poses.push_back(path_pose);
            }

            const gtsam::NonlinearFactorGraph& isam_graph = state.isam2.getFactorsUnsafe();

            for (size_t i = 0; i < isam_graph.size(); ++i) 
            {
                auto factor = isam_graph.at(i);
                auto between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
                
                if (between_factor) 
                {
                    gtsam::Key key1 = between_factor->front();
                    gtsam::Key key2 = between_factor->back();

                    if (state.optimized_estimates.exists(key1) && state.optimized_estimates.exists(key2)) 
                    {
                        gtsam::Pose3 pose1_base = state.optimized_estimates.at<gtsam::Pose3>(key1);
                        gtsam::Pose3 pose2_base = state.optimized_estimates.at<gtsam::Pose3>(key2);

                        geometry_msgs::msg::Point p1, p2;
                        p1.x = pose1_base.x(); p1.y = pose1_base.y(); p1.z = pose1_base.z();
                        p2.x = pose2_base.x(); p2.y = pose2_base.y(); p2.z = pose2_base.z();
                        
                        edges_marker.points.push_back(p1);
                        edges_marker.points.push_back(p2);
                    }
                }
            }

            marker_array.markers.push_back(nodes_marker);
            marker_array.markers.push_back(edges_marker);

            graph_markers_pubs_[robot_id]->publish(marker_array);
            path_pubs_[robot_id]->publish(path_msg);

            RCLCPP_INFO(this->get_logger(), "[Robo %d] --- RELATORIO GTSAM ---", robot_id);
            RCLCPP_INFO(this->get_logger(), "[Robo %d] Nos Totais no Grafo: %d", robot_id, (int)state.optimized_estimates.size());
            RCLCPP_INFO(this->get_logger(), "[Robo %d] Arestas (Fatores) Totais: %d", robot_id, (int)state.isam2.getFactorsUnsafe().size());
            RCLCPP_INFO(this->get_logger(), "[Robo %d] Pose odom->%s [X: %.3f | Y: %.3f | Z: %.3f]", robot_id, main_frame_id.c_str(), base_pose.x(), base_pose.y(), base_pose.z());
            RCLCPP_INFO(this->get_logger(), "-----------------------");
        } 
        catch (const gtsam::IndeterminantLinearSystemException& e) {
            RCLCPP_WARN(this->get_logger(), "[Robo %d] GTSAM IndeterminantLinearSystemException: Grafo instavel no momento.", robot_id);
        }
        catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "[Robo %d] Erro na publicacao dos dados do GTSAM: %s", robot_id, e.what());
        }
    }
   
    void ground_truth_callback(int robot_id, const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        auto& state = *robot_states_[robot_id];
        std::lock_guard<std::mutex> lock(state.gtsam_mutex);

        Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                            msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
        gtsam::Point3 t(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        
        gtsam::Pose3 current_gt_pose = gtsam::Pose3(gtsam::Rot3(q), t);

        if (!state.first_gt_received) 
        {
            state.initial_gt_pose = current_gt_pose;
            state.previous_gt_pose = current_gt_pose; 
            state.total_gt_distance = 0.0;            
            state.first_gt_received = true;
        }
        else
        {
            double step_distance = (current_gt_pose.translation() - state.previous_gt_pose.translation()).norm();
            state.total_gt_distance += step_distance;
        }

        state.latest_gt_pose = current_gt_pose;
        state.previous_gt_pose = current_gt_pose; 
        state.has_gt = true;
    }

    void factor_callback(int robot_id, const slam_interfaces::msg::GtsamData::SharedPtr msg)
    {
        FrameProcessResult result;

        gtsam::Rot3 delta_rot = gtsam::Rot3::Quaternion(
            msg->delta_base.pose.orientation.w, msg->delta_base.pose.orientation.x,
            msg->delta_base.pose.orientation.y, msg->delta_base.pose.orientation.z);
        gtsam::Point3 delta_trans(
            msg->delta_base.pose.position.x, msg->delta_base.pose.position.y, msg->delta_base.pose.position.z);
        result.delta_base = gtsam::Pose3(delta_rot, delta_trans);

        Eigen::MatrixXd cov_eigen = Eigen::MatrixXd::Zero(6, 6);
        for (int i = 0; i < 6; ++i) 
        {
            for (int j = 0; j < 6; ++j) 
            {
                cov_eigen(i, j) = msg->delta_base.covariance[i * 6 + j];
            }
        }
        result.visual_noise = gtsam::noiseModel::Gaussian::Covariance(cov_eigen);

        gtsam::Rot3 est_rot = gtsam::Rot3::Quaternion(
            msg->estimate.orientation.w, msg->estimate.orientation.x,
            msg->estimate.orientation.y, msg->estimate.orientation.z);
        gtsam::Point3 est_trans(
            msg->estimate.position.x, msg->estimate.position.y, msg->estimate.position.z);
        result.estimate = gtsam::Pose3(est_rot, est_trans);

        bool is_loop_closure = false;

        process_gtsam(robot_id, result, is_loop_closure);
    }

    void process_gtsam(int robot_id, const FrameProcessResult& result, bool is_loop)
    {
        auto& state = *robot_states_[robot_id]; 
        std::lock_guard<std::mutex> lock(state.gtsam_mutex);
        
        if (!state.has_keyframe) 
        {
            state.has_keyframe = true;
            state.keyframe_id = 0;
            state.graph.add(gtsam::PriorFactor<gtsam::Pose3>(
                gtsam::symbol_shorthand::X(0), gtsam::Pose3(), gtsam::noiseModel::Isotropic::Variance(6, 1e-6)));
            state.initial_estimates.insert(gtsam::symbol_shorthand::X(0), gtsam::Pose3());
            return; 
        }

        if (!is_loop)
        {
            state.keyframe_id++;
            int kf_id = state.keyframe_id;

            state.graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::symbol_shorthand::X(kf_id - 1), 
                gtsam::symbol_shorthand::X(kf_id), 
                result.delta_base, 
                result.visual_noise
            ));

            if (!state.initial_estimates.exists(gtsam::symbol_shorthand::X(kf_id))) 
            {
                state.initial_estimates.insert(gtsam::symbol_shorthand::X(kf_id), result.estimate);
            }
        }
        else
        {
            int hypothetical_from_id = 0; 

            auto robust_loop_noise = gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(1.345), result.visual_noise);

            state.graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                gtsam::symbol_shorthand::X(hypothetical_from_id), 
                gtsam::symbol_shorthand::X(state.keyframe_id), 
                result.delta_base,
                robust_loop_noise 
            ));
            
            RCLCPP_INFO(this->get_logger(), "[Robo %d] Loop Closure recebido e adicionado ao Grafo!", robot_id);
        }

        state.isam2.update(state.graph, state.initial_estimates);
        state.optimized_estimates = state.isam2.calculateEstimate();
        
        state.graph.resize(0);
        state.initial_estimates.clear();

        if (state.optimized_estimates.exists(gtsam::symbol_shorthand::X(state.keyframe_id))) 
        {
            gtsam::Pose3 corrected_pose = state.optimized_estimates.at<gtsam::Pose3>(gtsam::symbol_shorthand::X(state.keyframe_id));
            
            RCLCPP_INFO(this->get_logger(), "[Robo %d] --- GTSAM OTIMIZADO (KF %d) ---", robot_id, state.keyframe_id);
            RCLCPP_INFO(this->get_logger(), "[Robo %d] Pose X: %.3f | Y: %.3f | Z: %.3f", 
                        robot_id, corrected_pose.x(), corrected_pose.y(), corrected_pose.z());

            if (use_ground_truth_ && state.has_gt) 
            {
                gtsam::Pose3 relative_gt = state.initial_gt_pose.inverse() * state.latest_gt_pose;
                
                double trans_error = (corrected_pose.translation() - relative_gt.translation()).norm();
                double trans_error_pct = (state.total_gt_distance > 0.001) ? (trans_error / state.total_gt_distance) * 100.0 : 0.0;

                gtsam::Rot3 rot_diff = corrected_pose.rotation().between(relative_gt.rotation());
                double rot_error_rad = gtsam::Rot3::Logmap(rot_diff).norm();
                double rot_error_deg = rot_error_rad * (180.0 / M_PI);
                
                RCLCPP_INFO(this->get_logger(), "[Robo %d] --- COMPARACAO GROUND TRUTH ---", robot_id);
                RCLCPP_INFO(this->get_logger(), "[Robo %d] Erro Absoluto Translacao        : %.4f m (%.2f%%)", robot_id, trans_error, trans_error_pct);
                RCLCPP_INFO(this->get_logger(), "[Robo %d] Erro Absoluto Rotacao           : %.2f°", robot_id, rot_error_deg);
            }

            publish_gtsam_data(robot_id, corrected_pose, this->now());
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<GtsamOptimizationNode>();
    int num_robots = node->get_parameter("num_robots").as_int();
    
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 
        num_robots > 0 ? num_robots : 1
    );
    
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}