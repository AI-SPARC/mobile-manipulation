#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <deque>
#include <limits>
#include <csignal>
#include <functional>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/string.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <Eigen/Dense>


static void quatToEulerDeg(double qw, double qx, double qy, double qz,
                            double &roll, double &pitch, double &yaw)
{
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (qw * qy - qz * qx);
    if (std::abs(sinp) >= 1.0)
        pitch = std::copysign(M_PI / 2.0, sinp);
    else
        pitch = std::asin(sinp);

    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    yaw = std::atan2(siny_cosp, cosy_cosp);

    roll  *= 180.0 / M_PI;
    pitch *= 180.0 / M_PI;
    yaw   *= 180.0 / M_PI;
}

static std::string rotMatToString(const gtsam::Rot3 &R)
{
    gtsam::Matrix3 m = R.matrix();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    for (int i = 0; i < 3; ++i) {
        ss << "    [ ";
        for (int j = 0; j < 3; ++j) {
            ss << std::setw(10) << m(i, j);
            if (j < 2) ss << ", ";
        }
        ss << " ]";
        if (i < 2) ss << "\n";
    }
    return ss.str();
}

static std::string stampToString(const builtin_interfaces::msg::Time &t)
{
    std::ostringstream ss;
    ss << t.sec << "." << std::setw(9) << std::setfill('0') << t.nanosec
       << " s  [" << t.sec << " sec, " << t.nanosec << " nsec]";
    return ss.str();
}

class GroundTruth : public rclcpp::Node
{
public:
    GroundTruth()
    : Node("ground_truth_metrics_node")
    {
        gt_path_pub_   = this->create_publisher<nav_msgs::msg::Path>("ground_truth/path", 10);
        odom_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("odom/path", 10);
        report_pub_    = this->create_publisher<std_msgs::msg::String>("slam_metrics/report", 10);

        gt_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ground_truth", 100,
            std::bind(&GroundTruth::gt_callback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&GroundTruth::odom_callback, this, std::placeholders::_1));

        this->declare_parameter<std::string>("output_dir", "/tmp");
        output_dir_ = this->get_parameter("output_dir").as_string();

        RCLCPP_INFO(this->get_logger(),
            "GroundTruth Metrics Node inicializado.\n"
            "  GT  tópico : /ground_truth\n"
            "  Odom tópico: /odom\n"
            "  Relatório  : /slam_metrics/report\n"
            "  Modo       : DUPLA COMPARAÇÃO (estimação pura + latência)\n"
            "  TUM export : %s/gt_tum.txt  e  %s/slam_tum.txt\n"
            "  (salvo automaticamente ao encerrar com Ctrl+C)",
            output_dir_.c_str(), output_dir_.c_str());
    }

    
    ~GroundTruth()
    {
        save_trajectories_tum();
    }

   
    void save_trajectories_tum()
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        if (gt_path_.poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Nenhuma pose acumulada — TUM não gerado.");
            return;
        }

        const std::string gt_path_file   = output_dir_ + "/gt_tum.txt";
        const std::string slam_path_file  = output_dir_ + "/slam_tum.txt";

        std::ofstream f_gt(gt_path_file);
        if (!f_gt.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Não foi possível abrir %s", gt_path_file.c_str());
        } else {
            f_gt << "# TUM RGB-D ground truth format\n";
            f_gt << "# timestamp tx ty tz qx qy qz qw\n";
            f_gt << std::fixed << std::setprecision(9);
            for (const auto &ps : gt_path_.poses) {
                double t = ps.header.stamp.sec + ps.header.stamp.nanosec * 1e-9;
                const auto &p = ps.pose.position;
                const auto &q = ps.pose.orientation;
                f_gt << t    << " "
                     << p.x  << " " << p.y << " " << p.z << " "
                     << q.x  << " " << q.y << " " << q.z << " " << q.w << "\n";
            }
            f_gt.close();
        }

        std::ofstream f_sl(slam_path_file);
        if (!f_sl.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Não foi possível abrir %s", slam_path_file.c_str());
        } else {
            f_sl << "# TUM RGB-D SLAM odometry format\n";
            f_sl << "# timestamp tx ty tz qx qy qz qw\n";
            f_sl << std::fixed << std::setprecision(9);
            for (const auto &ps : odom_path_.poses) {
                double t = ps.header.stamp.sec + ps.header.stamp.nanosec * 1e-9;
                const auto &p = ps.pose.position;
                const auto &q = ps.pose.orientation;
                f_sl << t    << " "
                     << p.x  << " " << p.y << " " << p.z << " "
                     << q.x  << " " << q.y << " " << q.z << " " << q.w << "\n";
            }
            f_sl.close();
        }

        RCLCPP_INFO(this->get_logger(),
            "\n╔══════════════════════════════════════════════════════════════╗\n"
            "║                  TRAJETÓRIAS SALVAS (TUM)                    ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
            "  GT   → %s  (%zu poses)\n"
            "  SLAM → %s  (%zu poses)\n"
            "\n"
            "  Comandos para análise:\n"
            "\n"
            "  # ATE com alinhamento SE(3) — padrão TUM RGB-D benchmark:\n"
            "  evo_ape tum %s %s --align --plot\n"
            "\n"
            "  # ATE sem alinhamento — mostra offset de origem também:\n"
            "  evo_ape tum %s %s --plot\n"
            "\n"
            "  # RPE por step:\n"
            "  evo_rpe tum %s %s --align --plot\n"
            "\n"
            "  # Visualização das trajetórias sobrepostas:\n"
            "  evo_traj tum %s %s --ref %s --align --plot\n"
            "\n"
            "  # Salvar resultados em arquivo:\n"
            "  evo_ape tum %s %s --align --save_results /tmp/evo_result.zip\n",
            gt_path_file.c_str(),   gt_path_.poses.size(),
            slam_path_file.c_str(), odom_path_.poses.size(),
            gt_path_file.c_str(), slam_path_file.c_str(),
            gt_path_file.c_str(), slam_path_file.c_str(),
            gt_path_file.c_str(), slam_path_file.c_str(),
            gt_path_file.c_str(), slam_path_file.c_str(), gt_path_file.c_str(),
            gt_path_file.c_str(), slam_path_file.c_str());
    }

private:
    std::mutex buffer_mutex_;
    std::mutex metrics_mutex_;

    std::string output_dir_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr   gt_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr   odom_path_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr report_pub_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gt_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

    nav_msgs::msg::Path gt_path_;
    nav_msgs::msg::Path odom_path_;

    std::deque<nav_msgs::msg::Odometry::ConstSharedPtr> gt_buffer_;

    bool first_sync_      = true;
    gtsam::Pose3 prev_gt_pose_;
    gtsam::Pose3 prev_odom_pose_;

    int    num_samples_       = 0;
    double sum_sq_ate_        = 0.0;
    double sum_sq_ate_rot_    = 0.0;
    double sum_sq_rpe_trans_  = 0.0;
    double sum_sq_rpe_rot_    = 0.0;
    double total_gt_distance_ = 0.0;
    double total_gt_rotation_ = 0.0;
    double max_ate_trans_     = 0.0;
    double max_ate_rot_       = 0.0;
    double max_rpe_trans_     = 0.0;
    double max_rpe_rot_       = 0.0;

   
    int    num_rt_samples_  = 0;
    double sum_sq_rt_trans_ = 0.0;
    double sum_sq_rt_rot_   = 0.0;
    double max_rt_trans_    = 0.0;
    double max_rt_rot_      = 0.0;

    double first_pair_time_ = -1.0;


    gtsam::Pose3 getPose(const nav_msgs::msg::Odometry::ConstSharedPtr &msg)
    {
        Eigen::Quaterniond q(
            msg->pose.pose.orientation.w,
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z);
        if (q.w() < 0.0) q.coeffs() = -q.coeffs();
        q.normalize();
        gtsam::Point3 t(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z);
        return gtsam::Pose3(gtsam::Rot3(q), t);
    }

    nav_msgs::msg::Odometry::ConstSharedPtr findClosestGT(
        const rclcpp::Time &target,
        double max_diff_s = 0.10)
    {
        nav_msgs::msg::Odometry::ConstSharedPtr best = nullptr;
        double min_diff = std::numeric_limits<double>::max();
        for (const auto &entry : gt_buffer_) {
            double diff = std::abs((target - rclcpp::Time(entry->header.stamp)).seconds());
            if (diff < min_diff) {
                min_diff = diff;
                best     = entry;
            }
        }
        if (min_diff > max_diff_s) return nullptr;
        return best;
    }


    void gt_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        gt_buffer_.push_back(msg);
        if (gt_buffer_.size() > 4000)
            gt_buffer_.pop_front();
    }

    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg)
    {
        rclcpp::Time t_received = this->now();

        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (gt_buffer_.empty()) return;

        rclcpp::Time t_kf    = odom_msg->header.stamp;
        auto gt_at_kf        = findClosestGT(t_kf, 0.05);
        if (!gt_at_kf) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "GT@t_kf: par não encontrado (diff > 50ms). Descartado.");
            return;
        }

        auto   gt_at_now      = findClosestGT(t_received, 0.10);
        double sync_diff_kf   = std::abs((t_kf - rclcpp::Time(gt_at_kf->header.stamp)).seconds());
        double latency_s      = (t_received - t_kf).seconds();

        compute_metrics(gt_at_kf, gt_at_now, odom_msg, sync_diff_kf, latency_s);

        for (auto it = gt_buffer_.begin(); it != gt_buffer_.end(); ++it) {
            if ((*it) == gt_at_kf) {
                gt_buffer_.erase(gt_buffer_.begin(), it);
                break;
            }
        }
    }


    void compute_metrics(
        const nav_msgs::msg::Odometry::ConstSharedPtr &gt_kf_msg,
        const nav_msgs::msg::Odometry::ConstSharedPtr &gt_now_msg,
        const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg,
        double sync_diff_kf_s,
        double latency_s)
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        gtsam::Pose3 gt_kf_pose = getPose(gt_kf_msg);
        gtsam::Pose3 odom_pose  = getPose(odom_msg);

        double gt_t = gt_kf_msg->header.stamp.sec + gt_kf_msg->header.stamp.nanosec * 1e-9;
        if (first_pair_time_ < 0.0) first_pair_time_ = gt_t;
        double elapsed = gt_t - first_pair_time_;

        double gt_r, gt_p, gt_y;
        quatToEulerDeg(gt_kf_msg->pose.pose.orientation.w,
                       gt_kf_msg->pose.pose.orientation.x,
                       gt_kf_msg->pose.pose.orientation.y,
                       gt_kf_msg->pose.pose.orientation.z,
                       gt_r, gt_p, gt_y);

        double od_r, od_p, od_y;
        quatToEulerDeg(odom_msg->pose.pose.orientation.w,
                       odom_msg->pose.pose.orientation.x,
                       odom_msg->pose.pose.orientation.y,
                       odom_msg->pose.pose.orientation.z,
                       od_r, od_p, od_y);

        double vx = odom_msg->twist.twist.linear.x;
        double vy = odom_msg->twist.twist.linear.y;
        double vz = odom_msg->twist.twist.linear.z;
        double wx = odom_msg->twist.twist.angular.x;
        double wy = odom_msg->twist.twist.angular.y;
        double wz = odom_msg->twist.twist.angular.z;
        double v_norm = std::sqrt(vx*vx + vy*vy + vz*vz);

        double ate_trans = (gt_kf_pose.translation() - odom_pose.translation()).norm();
        double ate_dx    =  gt_kf_pose.translation().x() - odom_pose.translation().x();
        double ate_dy    =  gt_kf_pose.translation().y() - odom_pose.translation().y();
        double ate_dz    =  gt_kf_pose.translation().z() - odom_pose.translation().z();
        sum_sq_ate_ += ate_trans * ate_trans;
        max_ate_trans_ = std::max(max_ate_trans_, ate_trans);

        gtsam::Pose3 abs_err_pose = gt_kf_pose.between(odom_pose);
        double ate_rot     = gtsam::Rot3::Logmap(abs_err_pose.rotation()).norm();
        double ate_rot_deg = ate_rot * 180.0 / M_PI;
        sum_sq_ate_rot_ += ate_rot * ate_rot;
        max_ate_rot_ = std::max(max_ate_rot_, ate_rot);

        double rpe_trans = 0.0, rpe_rot = 0.0;
        double delta_gt_dist = 0.0, delta_gt_rot = 0.0;

        if (!first_sync_) {
            gtsam::Pose3 delta_gt   = prev_gt_pose_.between(gt_kf_pose);
            gtsam::Pose3 delta_odom = prev_odom_pose_.between(odom_pose);

            delta_gt_dist = delta_gt.translation().norm();
            delta_gt_rot  = gtsam::Rot3::Logmap(delta_gt.rotation()).norm();
            total_gt_distance_ += delta_gt_dist;
            total_gt_rotation_ += delta_gt_rot;

            gtsam::Pose3 err_rel = delta_gt.between(delta_odom);
            rpe_trans = err_rel.translation().norm();
            rpe_rot   = gtsam::Rot3::Logmap(err_rel.rotation()).norm();

            sum_sq_rpe_trans_ += rpe_trans * rpe_trans;
            sum_sq_rpe_rot_   += rpe_rot   * rpe_rot;
            max_rpe_trans_ = std::max(max_rpe_trans_, rpe_trans);
            max_rpe_rot_   = std::max(max_rpe_rot_,   rpe_rot);
        }

        prev_gt_pose_   = gt_kf_pose;
        prev_odom_pose_ = odom_pose;
        first_sync_     = false;
        num_samples_++;

        double ate_rmse     = std::sqrt(sum_sq_ate_     / num_samples_);
        double ate_rot_rmse = std::sqrt(sum_sq_ate_rot_ / num_samples_);
        double rpe_t_rmse   = (num_samples_ > 1) ? std::sqrt(sum_sq_rpe_trans_ / (num_samples_-1)) : 0.0;
        double rpe_r_rmse   = (num_samples_ > 1) ? std::sqrt(sum_sq_rpe_rot_   / (num_samples_-1)) : 0.0;

        double drift_pct_t  = (total_gt_distance_ > 1e-5) ? (ate_trans / total_gt_distance_) * 100.0 : 0.0;
        double drift_pct_r  = (total_gt_rotation_ > 1e-5) ? (ate_rot   / total_gt_rotation_) * 100.0 : 0.0;
        double ate_rmse_pct = (total_gt_distance_ > 1e-5) ? (ate_rmse  / total_gt_distance_) * 100.0 : 0.0;

        std::string rot_mat_str = rotMatToString(abs_err_pose.rotation());

        double rt_trans = -1.0, rt_rot = -1.0;
        double rt_trans_pct = -1.0, rt_rot_pct = -1.0;
        double rt_rmse_trans = 0.0, rt_rmse_rot = 0.0;
        double lat_component = -1.0;
        double gt_now_r = 0, gt_now_p = 0, gt_now_y = 0;
        double gt_now_x = 0, gt_now_y_pos = 0, gt_now_z = 0;
        std::string rot_mat_rt_str;

        if (gt_now_msg) {
            gtsam::Pose3 gt_now_pose = getPose(gt_now_msg);
            rt_trans = (gt_now_pose.translation() - odom_pose.translation()).norm();
            gtsam::Pose3 rt_err_pose = gt_now_pose.between(odom_pose);
            rt_rot   = gtsam::Rot3::Logmap(rt_err_pose.rotation()).norm();

            sum_sq_rt_trans_ += rt_trans * rt_trans;
            sum_sq_rt_rot_   += rt_rot   * rt_rot;
            max_rt_trans_ = std::max(max_rt_trans_, rt_trans);
            max_rt_rot_   = std::max(max_rt_rot_,   rt_rot);
            num_rt_samples_++;

            rt_rmse_trans = std::sqrt(sum_sq_rt_trans_ / num_rt_samples_);
            rt_rmse_rot   = std::sqrt(sum_sq_rt_rot_   / num_rt_samples_);
            rt_trans_pct  = (total_gt_distance_ > 1e-5) ? (rt_trans / total_gt_distance_) * 100.0 : 0.0;
            rt_rot_pct    = (total_gt_rotation_ > 1e-5) ? (rt_rot   / total_gt_rotation_) * 100.0 : 0.0;
            lat_component = rt_trans - ate_trans;

            quatToEulerDeg(gt_now_msg->pose.pose.orientation.w,
                           gt_now_msg->pose.pose.orientation.x,
                           gt_now_msg->pose.pose.orientation.y,
                           gt_now_msg->pose.pose.orientation.z,
                           gt_now_r, gt_now_p, gt_now_y);
            gt_now_x     = gt_now_msg->pose.pose.position.x;
            gt_now_y_pos = gt_now_msg->pose.pose.position.y;
            gt_now_z     = gt_now_msg->pose.pose.position.z;
            rot_mat_rt_str = rotMatToString(rt_err_pose.rotation());
        }

    
        std::ostringstream oss;
        oss << std::fixed;

        oss << "\n╔══════════════════════════════════════════════════════════════╗\n";
        oss << "║              SLAM GROUND TRUTH METRICS  ─ Sample #"
            << std::setw(5) << num_samples_ << "       ║\n";
        oss << "╚══════════════════════════════════════════════════════════════╝\n";

        oss << "\n┌─ TIMESTAMPS ──────────────────────────────────────────────────\n";
        oss << "│ KF   stamp (t_kf) : " << stampToString(odom_msg->header.stamp) << "\n";
        oss << "│ GT@kf stamp       : " << stampToString(gt_kf_msg->header.stamp) << "\n";
        if (gt_now_msg)
        oss << "│ GT@now stamp      : " << stampToString(gt_now_msg->header.stamp) << "\n";
        oss << "│ Sync diff (kf)    : " << std::setprecision(4) << sync_diff_kf_s * 1000.0 << " ms\n";
        oss << "│ Latência pipeline : " << std::setprecision(1) << latency_s * 1000.0 << " ms"
            << "  (" << std::setprecision(3) << latency_s << " s)\n";
        oss << "│ Tempo decorrido   : " << std::setprecision(3) << elapsed << " s\n";
        oss << "│ GT   frame        : " << gt_kf_msg->header.frame_id << "\n";
        oss << "│ Odom frame        : " << odom_msg->header.frame_id  << "\n";

        oss << "├─ GROUND TRUTH  @  t_kf ───────────────────────────────────────\n";
        oss << std::setprecision(6);
        oss << "│ X : " << std::setw(12) << gt_kf_msg->pose.pose.position.x << " m\n";
        oss << "│ Y : " << std::setw(12) << gt_kf_msg->pose.pose.position.y << " m\n";
        oss << "│ Z : " << std::setw(12) << gt_kf_msg->pose.pose.position.z << " m\n";
        oss << "│ Quaternion : w=" << std::setw(10) << gt_kf_msg->pose.pose.orientation.w
            << "  x=" << std::setw(10) << gt_kf_msg->pose.pose.orientation.x
            << "  y=" << std::setw(10) << gt_kf_msg->pose.pose.orientation.y
            << "  z=" << std::setw(10) << gt_kf_msg->pose.pose.orientation.z << "\n";
        oss << std::setprecision(4);
        oss << "│ Euler (deg) : roll=" << std::setw(9) << gt_r
            << "°  pitch=" << std::setw(9) << gt_p
            << "°  yaw="   << std::setw(9) << gt_y << "°\n";

        if (gt_now_msg) {
        oss << "├─ GROUND TRUTH  @  t_now  (onde o robô está AGORA) ───────────\n";
        oss << std::setprecision(6);
        oss << "│ X : " << std::setw(12) << gt_now_x     << " m\n";
        oss << "│ Y : " << std::setw(12) << gt_now_y_pos << " m\n";
        oss << "│ Z : " << std::setw(12) << gt_now_z     << " m\n";
        oss << "│ Quaternion : w=" << std::setw(10) << gt_now_msg->pose.pose.orientation.w
            << "  x=" << std::setw(10) << gt_now_msg->pose.pose.orientation.x
            << "  y=" << std::setw(10) << gt_now_msg->pose.pose.orientation.y
            << "  z=" << std::setw(10) << gt_now_msg->pose.pose.orientation.z << "\n";
        oss << std::setprecision(4);
        oss << "│ Euler (deg) : roll=" << std::setw(9) << gt_now_r
            << "°  pitch=" << std::setw(9) << gt_now_p
            << "°  yaw="   << std::setw(9) << gt_now_y << "°\n";
        }

        oss << "├─ SLAM (ODOM) POSE  @  t_kf ───────────────────────────────────\n";
        oss << std::setprecision(6);
        oss << "│ X : " << std::setw(12) << odom_msg->pose.pose.position.x << " m\n";
        oss << "│ Y : " << std::setw(12) << odom_msg->pose.pose.position.y << " m\n";
        oss << "│ Z : " << std::setw(12) << odom_msg->pose.pose.position.z << " m\n";
        oss << "│ Quaternion : w=" << std::setw(10) << odom_msg->pose.pose.orientation.w
            << "  x=" << std::setw(10) << odom_msg->pose.pose.orientation.x
            << "  y=" << std::setw(10) << odom_msg->pose.pose.orientation.y
            << "  z=" << std::setw(10) << odom_msg->pose.pose.orientation.z << "\n";
        oss << std::setprecision(4);
        oss << "│ Euler (deg) : roll=" << std::setw(9) << od_r
            << "°  pitch=" << std::setw(9) << od_p
            << "°  yaw="   << std::setw(9) << od_y << "°\n";
        oss << "│ Velocidade  : |v|=" << std::setw(7) << v_norm << " m/s"
            << "  (vx=" << vx << " vy=" << vy << " vz=" << vz << ")\n";
        oss << "│ Vel. Angular:  wx=" << std::setw(8) << wx
            << "  wy=" << std::setw(8) << wy
            << "  wz=" << std::setw(8) << wz << " rad/s\n";

        oss << "├─ TRAJETÓRIA REAL (GT acumulada) ──────────────────────────────\n";
        oss << std::setprecision(4);
        oss << "│ Distância total : " << std::setw(10) << total_gt_distance_ << " m\n";
        oss << "│ Rotação total   : " << std::setw(10) << total_gt_rotation_ << " rad"
            << "  (" << total_gt_rotation_ * 180.0/M_PI << "°)\n";
        oss << "│ Δdist este step : " << std::setw(10) << delta_gt_dist << " m\n";
        oss << "│ Δrot  este step : " << std::setw(10) << delta_gt_rot  << " rad\n";

        oss << "╞═══ [A] ERRO DE ESTIMAÇÃO PURA  (GT@t_kf  vs  SLAM@t_kf) ═════╡\n";
        oss << "│  Pergunta: o SLAM estimou bem a pose do frame capturado?\n";
        oss << std::setprecision(5);
        oss << "│ ΔX : " << std::setw(12) << ate_dx << " m\n";
        oss << "│ ΔY : " << std::setw(12) << ate_dy << " m\n";
        oss << "│ ΔZ : " << std::setw(12) << ate_dz << " m\n";
        oss << "│ |ΔT| atual  : " << std::setw(10) << ate_trans << " m"
            << "  →  " << std::setprecision(3) << drift_pct_t << "% da dist.\n";
        oss << "│ |ΔR| atual  : " << std::setprecision(5) << std::setw(10) << ate_rot << " rad"
            << "  (" << std::setprecision(3) << ate_rot_deg << "°)"
            << "  →  " << std::setprecision(3) << drift_pct_r << "% da rot.\n";
        oss << "│ Matriz rot erro:\n" << rot_mat_str << "\n";
        oss << "│ ATE transl RMSE : " << std::setprecision(5) << std::setw(10) << ate_rmse
            << " m  (" << std::setprecision(3) << ate_rmse_pct << "% da dist.)\n";
        oss << "│ ATE transl MAX  : " << std::setprecision(5) << std::setw(10) << max_ate_trans_ << " m\n";
        oss << "│ ATE rot   RMSE  : " << std::setprecision(5) << std::setw(10) << ate_rot_rmse << " rad"
            << "  (" << ate_rot_rmse * 180.0/M_PI << "°)\n";
        oss << "│ ATE rot   MAX   : " << std::setprecision(5) << std::setw(10) << max_ate_rot_ << " rad"
            << "  (" << max_ate_rot_ * 180.0/M_PI << "°)\n";
        oss << "│ RPE transl RMSE : " << std::setprecision(5) << std::setw(10) << rpe_t_rmse << " m/step\n";
        oss << "│ RPE transl MAX  : " << std::setprecision(5) << std::setw(10) << max_rpe_trans_ << " m/step\n";
        oss << "│ RPE transl atual: " << std::setprecision(5) << std::setw(10) << rpe_trans << " m/step\n";
        oss << "│ RPE rot   RMSE  : " << std::setprecision(5) << std::setw(10) << rpe_r_rmse << " rad/step"
            << "  (" << rpe_r_rmse * 180.0/M_PI << "°/step)\n";
        oss << "│ RPE rot   MAX   : " << std::setprecision(5) << std::setw(10) << max_rpe_rot_ << " rad/step\n";
        oss << "│ RPE rot   atual : " << std::setprecision(5) << std::setw(10) << rpe_rot << " rad/step\n";

        oss << "╞═══ [B] ERRO EM TEMPO REAL      (GT@t_now vs  SLAM@t_kf) ═════╡\n";
        oss << "│  Pergunta: qual o erro que o consumidor recebe agora?\n";
        if (gt_now_msg) {
        oss << std::setprecision(5);
        oss << "│ |ΔT| RT atual   : " << std::setw(10) << rt_trans << " m"
            << "  →  " << std::setprecision(3) << rt_trans_pct << "% da dist.\n";
        oss << "│ |ΔR| RT atual   : " << std::setprecision(5) << std::setw(10) << rt_rot << " rad"
            << "  (" << rt_rot * 180.0/M_PI << "°)"
            << "  →  " << std::setprecision(3) << rt_rot_pct << "% da rot.\n";
        oss << "│ Matriz rot erro:\n" << rot_mat_rt_str << "\n";
        oss << "│ RT transl RMSE  : " << std::setprecision(5) << std::setw(10) << rt_rmse_trans << " m\n";
        oss << "│ RT transl MAX   : " << std::setprecision(5) << std::setw(10) << max_rt_trans_ << " m\n";
        oss << "│ RT rot   RMSE   : " << std::setprecision(5) << std::setw(10) << rt_rmse_rot << " rad"
            << "  (" << rt_rmse_rot * 180.0/M_PI << "°)\n";
        oss << "│ RT rot   MAX    : " << std::setprecision(5) << std::setw(10) << max_rt_rot_ << " rad"
            << "  (" << max_rt_rot_ * 180.0/M_PI << "°)\n";
        } else {
        oss << "│ (GT@t_now indisponível — buffer não cobre esse instante)\n";
        }

        oss << "╞═══ [C] DECOMPOSIÇÃO DO ERRO TOTAL ════════════════════════════╡\n";
        oss << std::setprecision(1);
        oss << "│ Latência do pipeline    : " << std::setw(8) << latency_s * 1000.0 << " ms\n";
        if (gt_now_msg && rt_trans >= 0.0) {
        oss << std::setprecision(5);
        oss << "│ Erro de estimação [A]   : " << std::setw(10) << ate_trans << " m\n";
        oss << "│ Erro tempo real   [B]   : " << std::setw(10) << rt_trans  << " m\n";
        oss << "│ Componente latência[B-A]: " << std::setw(10) << lat_component << " m"
            << "  (" << std::setprecision(1) << latency_s*1000.0 << " ms × "
            << std::setprecision(3) << v_norm << " m/s ≈ "
            << std::setprecision(4) << latency_s * v_norm << " m esperado)\n";
        }
        oss << "╞═══ [TUM] EXPORT ══════════════════════════════════════════════╡\n";
        oss << "│ Poses acumuladas: GT=" << gt_path_.poses.size()
            << "  SLAM=" << odom_path_.poses.size() << "\n";
        oss << "│ Salvo em " << output_dir_ << "/  ao encerrar (Ctrl+C)\n";
        oss << "└───────────────────────────────────────────────────────────────\n";
        oss << "────────────────────────────────────────────────────────────────\n";

        std::string report = oss.str();
        auto msg_out = std_msgs::msg::String();
        msg_out.data = report;
        report_pub_->publish(msg_out);
        RCLCPP_INFO(this->get_logger(), "%s", report.c_str());

        update_paths(gt_kf_msg, odom_msg);
    }

    void update_paths(const nav_msgs::msg::Odometry::ConstSharedPtr &gt_msg,
                      const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg)
    {
        if (gt_path_.poses.empty()) {
            gt_path_.header.frame_id   = gt_msg->header.frame_id;
            odom_path_.header.frame_id = odom_msg->header.frame_id;
        }

        geometry_msgs::msg::PoseStamped gp;
        gp.header = gt_msg->header;
        gp.pose   = gt_msg->pose.pose;
        gt_path_.poses.push_back(gp);
        gt_path_.header.stamp = gt_msg->header.stamp;
        gt_path_pub_->publish(gt_path_);

        geometry_msgs::msg::PoseStamped op;
        op.header = odom_msg->header;
        op.pose   = odom_msg->pose.pose;
        odom_path_.poses.push_back(op);
        odom_path_.header.stamp = odom_msg->header.stamp;
        odom_path_pub_->publish(odom_path_);
    }
};


static std::shared_ptr<GroundTruth> g_node;

void sigint_handler(int)
{
    if (g_node) {
        RCLCPP_INFO(g_node->get_logger(), "\nSIGINT recebido — salvando trajetórias TUM...");
        g_node->save_trajectories_tum();
    }
    rclcpp::shutdown();
}


int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    g_node = std::make_shared<GroundTruth>();

    std::signal(SIGINT, sigint_handler);

    rclcpp::spin(g_node);

    g_node->save_trajectories_tum();
    g_node.reset();
    return 0;
}