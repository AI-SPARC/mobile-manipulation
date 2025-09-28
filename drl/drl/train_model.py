import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
from object_manipulation_interfaces.srv import EvaluateReward
import sensor_msgs_py.point_cloud2 as pc2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from threading import Lock


# --- Rede neural (política) ---
class PolicyNet(nn.Module):
    def __init__(self, input_dim=1000, output_dim=7):
        super().__init__()
        layers = []
        hidden_dims = [512, 512, 256, 256, 128, 128]
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- Função de voxelização e centralização ---
def pointcloud_to_voxel_vector(msg, cube_size=10, vector_size=1000, max_dist=0.2):
    points = np.array(
        [(x, y, z) for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)],
        dtype=np.float32,
    )

    if points.shape[0] == 0:
        return np.zeros(vector_size, dtype=np.float32), np.zeros(3, dtype=np.float32)

    # Agora não centraliza para a nuvem
    max_abs = np.max(np.abs(points))
    scale = max_dist / max_abs if max_abs != 0 else 1.0
    points_scaled = points * scale

    normalized = ((points_scaled + max_dist) / (2 * max_dist) * cube_size)
    normalized = np.floor(normalized).astype(int)
    normalized = np.clip(normalized, 0, cube_size - 1)

    grid = np.zeros((cube_size, cube_size, cube_size), dtype=np.float32)
    for x, y, z in normalized:
        grid[x, y, z] = 1.0

    flat = grid.flatten()
    if len(flat) < vector_size:
        flat = np.pad(flat, (0, vector_size - len(flat)), "constant")
    else:
        flat = flat[:vector_size]

    # Para referência absoluta, retorna (0,0,0) porque não queremos deslocar a nuvem
    return flat, np.zeros(3, dtype=np.float32)



# --- Nó ROS2 ---
class RLPolicyNode(Node):
    def __init__(self):
        super().__init__("rl_policy_node_service")

        self.model = PolicyNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.cli = self.create_client(EvaluateReward, "compute_reward")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service não disponível, aguardando...")

        self.create_subscription(PointCloud2, "/points", self.pc_callback, 10)

        self.max_distance = 0.1
        self.processing = False
        self.lock = Lock()

    def pc_callback(self, msg: PointCloud2):
        with self.lock:
            if self.processing:
                return
            self.processing = True

        obs_vector, _ = pointcloud_to_voxel_vector(msg, cube_size=10, vector_size=1000, max_dist=self.max_distance)
        obs = torch.tensor(obs_vector, dtype=torch.float32)

        action = self.model(obs)
        dist = torch.distributions.Normal(action, torch.ones_like(action) * 0.1)
        sampled_action = dist.rsample()
        logprob = dist.log_prob(sampled_action).sum()

        # --- posição no frame world ---
        pos_world = sampled_action[:3].detach().numpy()  # agora é diretamente no world
        pos_world = np.clip(pos_world, -self.max_distance, self.max_distance)

        pose_msg = Pose()
        pose_msg.position.x = float(pos_world[0])
        pose_msg.position.y = float(pos_world[1])
        pose_msg.position.z = float(pos_world[2])
        pose_msg.orientation.x = float(sampled_action[3].item())
        pose_msg.orientation.y = float(sampled_action[4].item())
        pose_msg.orientation.z = float(sampled_action[5].item())
        pose_msg.orientation.w = float(sampled_action[6].item())

        self.get_logger().info(f"Pose proposta (world): {pose_msg.position.x}, {pose_msg.position.y}, {pose_msg.position.z}")

        req = EvaluateReward.Request()
        req.pose = sampled_action.detach().numpy().tolist()

        future = self.cli.call_async(req)
        future.add_done_callback(lambda f, lp=logprob: self.service_response_callback(f, lp))


    def service_response_callback(self, future, logprob):
        try:
            response = future.result()
            reward = response.reward
            self.update_policy(logprob, reward)
        except Exception as e:
            self.get_logger().error(f"Falha no serviço: {e}")
        finally:
            self.processing = False

    def update_policy(self, logprob, reward):
        # garante que logprob mantém grad
        if not isinstance(logprob, torch.Tensor):
            logprob = torch.tensor(logprob, dtype=torch.float32)

        reward_tensor = torch.tensor(float(reward), dtype=torch.float32)
        loss = -(logprob * reward_tensor)
        if loss.dim() > 0:
            loss = loss.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.get_logger().info(f"Pose atualizada, reward={reward:.6f}, loss={loss.item():.6f}")


def main():
    rclpy.init()
    node = RLPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
