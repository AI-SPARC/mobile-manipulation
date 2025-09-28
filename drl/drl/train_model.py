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
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)



# --- Função de voxelização e centralização ---
def pointcloud_to_vector(msg, vector_size=1000):
    # Extrai pontos (x,y,z) da nuvem
    points = np.array(
        [(x, y, z) for x, y, z in pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )],
        dtype=np.float32,
    )

    if points.shape[0] == 0:
        return np.zeros(vector_size, dtype=np.float32), np.zeros(3, dtype=np.float32)

    # Ordena por x, depois y, depois z
    points_sorted = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]

    # Centro do cubo (média dos extremos)
    p_min = points_sorted.min(axis=0)
    p_max = points_sorted.max(axis=0)
    center = (p_min + p_max) / 2.0

    # Achata em vetor 1D
    flat = points_sorted.flatten()

    # Ajusta tamanho para a rede (corta ou preenche com zero)
    if len(flat) < vector_size:
        flat = np.pad(flat, (0, vector_size - len(flat)), "constant")
    else:
        flat = flat[:vector_size]

    return flat, center


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --- Nó ROS2 ---
class RLPolicyNode(Node):
    def __init__(self):
        super().__init__("rl_policy_node_service")

        self.model = PolicyNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-1)

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

        obs_vector, center = pointcloud_to_vector(msg, vector_size=1000)
        obs = torch.tensor(obs_vector, dtype=torch.float32, device=device).unsqueeze(0)

        action = self.model(obs)
        dist = torch.distributions.Normal(action, torch.ones_like(action, device=device) * 1e-8)
        sampled_action = dist.rsample()
        logprob = dist.log_prob(sampled_action).sum()

        # Posição e orientação
        pos_world = sampled_action[:3].detach().cpu().numpy()
        orient = sampled_action[3:7].detach().cpu().numpy() if len(sampled_action) >= 7 else np.array([0,0,0,1])

        # Distância ao centro (posição apenas)
        dist_to_center = np.linalg.norm(pos_world - center)

        pose_msg = Pose()
        pose_msg.position.x = float(pos_world[0])
        pose_msg.position.y = float(pos_world[1])
        pose_msg.position.z = float(pos_world[2])
        pose_msg.orientation.x = float(orient[0])
        pose_msg.orientation.y = float(orient[1])
        pose_msg.orientation.z = float(orient[2])
        pose_msg.orientation.w = float(orient[3])

        self.get_logger().info(
            f"Pose proposta (world): pos={pos_world}, ori={orient}, distância ao centro={dist_to_center:.4f}"
        )
        
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
