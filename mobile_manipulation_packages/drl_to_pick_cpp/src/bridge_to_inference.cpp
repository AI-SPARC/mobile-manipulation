#include "drl_to_pick_cpp/BridgeToInference.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

#include <msgpack.hpp>
#include <pcl_conversions/pcl_conversions.h> 

#include <chrono>
#include <cstring>
#include <algorithm>

namespace drl_to_pick_cpp
{

BridgeToInference::BridgeToInference(const rclcpp::NodeOptions & options)
: Node("bridge_to_inference", options)
{
  declare_parameter("server_host", "localhost");
  declare_parameter("server_port", 5000);
  declare_parameter("score_threshold", 0.15);
  declare_parameter("max_grasps", 50);
  declare_parameter("target_frame", "world");

  server_host_ = get_parameter("server_host").as_string();
  server_port_ = get_parameter("server_port").as_int();
  score_threshold_ = static_cast<float>(get_parameter("score_threshold").as_double());
  max_grasps_ = get_parameter("max_grasps").as_int();
  target_frame_ = get_parameter("target_frame").as_string();

  pub_grasps_ = create_publisher<geometry_msgs::msg::PoseArray>("/grasp_poses", 10);

  sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    "/mapped_object",
    10,
    std::bind(&BridgeToInference::cloud_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "BRIDGE TO INFERENCE (C++)");
  RCLCPP_INFO(get_logger(), "Server: %s:%d", server_host_.c_str(), server_port_);
  RCLCPP_INFO(get_logger(), "Score threshold: %.2f", score_threshold_);
  RCLCPP_INFO(get_logger(), "Listening on topic: mapped_object");
}

std::vector<geometry_msgs::msg::Pose> BridgeToInference::get_latest_grasps()
{
  std::lock_guard<std::mutex> lock(grasp_mutex_);
  return latest_grasps_;
}

void BridgeToInference::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  // RCLCPP_INFO(get_logger(), "==================================================");
  // RCLCPP_INFO(get_logger(), "Recebido PointCloud2. Convertendo...");

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *cloud);

  if (cloud->empty())
  {
    RCLCPP_WARN(get_logger(), "Point cloud vazia recebida no callback!");
    
    {
      std::lock_guard<std::mutex> lock(grasp_mutex_);
      latest_grasps_.clear();
    }
    return;
  }

  
  std::vector<geometry_msgs::msg::Pose> new_grasps = get_grasps_from_server(cloud);

  
  {
    std::lock_guard<std::mutex> lock(grasp_mutex_);
    latest_grasps_ = new_grasps;
  }

  
    // if (!new_grasps.empty())
    // {
    //   publish_grasps(new_grasps);
    // }
    // else
    // {
    //   RCLCPP_WARN(get_logger(), "Nenhum grasp retornado pelo servidor.");
    // }
}

geometry_msgs::msg::Pose BridgeToInference::matrix_to_pose(const Eigen::Matrix4f & matrix)
{
  geometry_msgs::msg::Pose pose;

  pose.position.x = matrix(0, 3);
  pose.position.y = matrix(1, 3);
  pose.position.z = matrix(2, 3);

  Eigen::Matrix3f R = matrix.block<3, 3>(0, 0);
  Eigen::Quaternionf q(R);
  q.normalize();

  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();

  return pose;
}

std::vector<geometry_msgs::msg::Pose> BridgeToInference::get_grasps_from_server(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud)
{
  // RCLCPP_INFO(get_logger(), "Enviando %zu pontos para servidor...", cloud->size());
  auto t0 = std::chrono::steady_clock::now();

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0)
  {
    RCLCPP_ERROR(get_logger(), "Falha ao criar socket");
    return {};
  }

  struct timeval timeout;
  timeout.tv_sec = 30;
  timeout.tv_usec = 0;
  setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

  struct hostent * server = gethostbyname(server_host_.c_str());
  if (server == nullptr)
  {
    RCLCPP_ERROR(get_logger(), "Host não encontrado: %s", server_host_.c_str());
    close(sock);
    return {};
  }

  struct sockaddr_in serv_addr;
  std::memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  std::memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);
  serv_addr.sin_port = htons(server_port_);

  if (connect(sock, reinterpret_cast<struct sockaddr *>(&serv_addr), sizeof(serv_addr)) < 0)
  {
    RCLCPP_ERROR(get_logger(), "Conexão recusada - servidor rodando em %s:%d?",
      server_host_.c_str(), server_port_);
    close(sock);
    return {};
  }

  
  msgpack::sbuffer sbuf;
  msgpack::packer<msgpack::sbuffer> packer(sbuf);

  packer.pack_map(2);

  packer.pack("num_points");
  packer.pack(static_cast<uint32_t>(cloud->size()));

  packer.pack("points");
  packer.pack_array(cloud->size() * 3);
  for (const auto & p : cloud->points)
  {
    packer.pack(p.x);
    packer.pack(p.y);
    packer.pack(p.z);
  }

  
  uint64_t data_size = sbuf.size();
  uint64_t data_size_be = htobe64(data_size);
  if (send(sock, &data_size_be, 8, 0) != 8)
  {
    RCLCPP_ERROR(get_logger(), "Falha ao enviar tamanho");
    close(sock);
    return {};
  }

 
  size_t total_sent = 0;
  while (total_sent < sbuf.size())
  {
    ssize_t sent = send(sock, sbuf.data() + total_sent, sbuf.size() - total_sent, 0);
    if (sent <= 0)
    {
      RCLCPP_ERROR(get_logger(), "Falha ao enviar dados");
      close(sock);
      return {};
    }
    total_sent += sent;
  }

  
  uint64_t resp_size_be;
  if (recv(sock, &resp_size_be, 8, MSG_WAITALL) != 8)
  {
    RCLCPP_ERROR(get_logger(), "Servidor fechou conexão ou erro no recv header");
    close(sock);
    return {};
  }
  uint64_t resp_size = be64toh(resp_size_be);

  
  std::vector<char> resp_data(resp_size);
  size_t total_recv = 0;
  while (total_recv < resp_size)
  {
    ssize_t received = recv(sock, resp_data.data() + total_recv, resp_size - total_recv, 0);
    if (received <= 0)
    {
      RCLCPP_ERROR(get_logger(), "Falha ao receber dados (payload)");
      close(sock);
      return {};
    }
    total_recv += received;
  }

  close(sock);

  
  std::vector<geometry_msgs::msg::Pose> grasps;

  try
  {
    msgpack::object_handle oh = msgpack::unpack(resp_data.data(), resp_data.size());
    msgpack::object obj = oh.get();

    std::map<std::string, msgpack::object> response_map;
    obj.convert(response_map);

    std::vector<float> scores;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();

    if (response_map.count("scores"))
    {
      response_map["scores"].convert(scores);
    }

    if (response_map.count("centroid"))
    {
      std::vector<float> centroid_vec;
      response_map["centroid"].convert(centroid_vec);
      if (centroid_vec.size() == 3)
      {
        centroid = Eigen::Vector3f(centroid_vec[0], centroid_vec[1], centroid_vec[2]);
      }
    }

    if (response_map.count("grasps") && !scores.empty())
    {
      std::vector<float> grasps_flat;
      response_map["grasps"].convert(grasps_flat);

      size_t num_grasps = grasps_flat.size() / 16;

      
      std::vector<std::pair<float, geometry_msgs::msg::Pose>> score_pose_pairs;
      score_pose_pairs.reserve(num_grasps);

      for (size_t i = 0; i < num_grasps; ++i)
      {
        if (scores[i] < score_threshold_)
        {
          continue;
        }

        Eigen::Matrix4f mat;
        for (int r = 0; r < 4; ++r)
        {
          for (int c = 0; c < 4; ++c)
          {
            mat(r, c) = grasps_flat[i * 16 + r * 4 + c];
          }
        }

        
        mat(0, 3) += centroid.x();
        mat(1, 3) += centroid.y();
        mat(2, 3) += centroid.z();

        score_pose_pairs.emplace_back(scores[i], matrix_to_pose(mat));
      }

      
      std::sort(score_pose_pairs.begin(), score_pose_pairs.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });

      
      size_t count = std::min(score_pose_pairs.size(), static_cast<size_t>(max_grasps_));

      grasps.reserve(count);
      for (size_t i = 0; i < count; ++i)
      {
        grasps.push_back(score_pose_pairs[i].second);

        // if (i < 5)
        // {
        //   const auto & pose = score_pose_pairs[i].second;
        //   RCLCPP_INFO(get_logger(), "  #%zu: score=%.3f, pos=[%.3f, %.3f, %.3f]",
        //     i + 1, score_pose_pairs[i].first,
        //     pose.position.x, pose.position.y, pose.position.z);
        // }
      }

      // RCLCPP_INFO(get_logger(), "Grasps válidos: %zu (threshold=%.2f)", count, score_threshold_);
    }
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(get_logger(), "Erro ao deserializar: %s", e.what());
    return {};
  }

  auto t1 = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(t1 - t0).count();
  RCLCPP_INFO(get_logger(), "Recebido %zu grasps em %.2fs", grasps.size(), dt);

  return grasps;
}

void BridgeToInference::publish_grasps(const std::vector<geometry_msgs::msg::Pose> & grasps)
{
  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header.stamp = get_clock()->now();
  pose_array.header.frame_id = target_frame_;
  pose_array.poses = grasps;

  pub_grasps_->publish(pose_array);
  RCLCPP_INFO(get_logger(), "Publicados %zu grasps em /grasp_poses", grasps.size());
}

}  // namespace drl_to_pick_cpp

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(drl_to_pick_cpp::BridgeToInference)