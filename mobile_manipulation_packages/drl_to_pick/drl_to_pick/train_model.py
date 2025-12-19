#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
import os
import sys
import threading
import struct
import matplotlib.pyplot as plt

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

# ==========================================
# 1. CONFIGURAÇÕES & HIPERPARÂMETROS
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning Rates
LR_ACTOR = 1e-4   
LR_CRITIC = 1e-3  

BATCH_SIZE = 32
EPOCHS = 200

# Limites do Espaço de Trabalho
WORKSPACE_LIMIT = 0.3 

# Arquivo de dados
DATASET_FILE = "/home/momesso/isaacsim/xarm_final_v5/results_safe.yaml"

# Variáveis globais para plotagem
GLOBAL_HISTORY = {
    'epoch': [],
    'critic_loss_avg': [],
    'actor_obj_avg': []
}

# ==========================================
# 2. REDES NEURAIS (Actor-Critic)
# ==========================================
class CubeEncoder(nn.Module):
    def __init__(self):
        super(CubeEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2) 
        self.flat_size = 32 * 12 * 12 * 12 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.encoder = CubeEncoder()
        self.fc1 = nn.Linear(self.encoder.flat_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out_pos = nn.Linear(128, 3) 
        self.out_quat = nn.Linear(128, 4) 

    def forward(self, grid):
        features = self.encoder(grid)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        pos = torch.tanh(self.out_pos(x)) * WORKSPACE_LIMIT 
        quat = F.normalize(self.out_quat(x), p=2, dim=1)
        return torch.cat([pos, quat], dim=1)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.encoder = CubeEncoder()
        self.pose_fc = nn.Linear(7, 32) 
        self.fc1 = nn.Linear(self.encoder.flat_size + 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, grid, pose):
        grid_feat = self.encoder(grid)
        pose_feat = F.relu(self.pose_fc(pose))
        combined = torch.cat([grid_feat, pose_feat], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.out(x) 

# ==========================================
# 3. DATASET
# ==========================================
class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, static_grid):
        self.data = []
        self.static_grid = static_grid
        if not os.path.exists(file_path):
            print(f"ERRO: Arquivo não encontrado: {file_path}")
            sys.exit(1)
        print(f"Carregando dataset: {file_path}")
        with open(file_path, 'r') as stream:
            try:
                doc_list = yaml.safe_load(stream)
                if not isinstance(doc_list, list): doc_list = [doc_list]
                success_count = 0
                for entry in doc_list:
                    if not isinstance(entry, dict): continue
                    p = entry.get('pose_pos', [0,0,0])
                    r = entry.get('pose_rot', [0,0,0,1])
                    pose_np = np.array(p + r, dtype=np.float32)
                    success = entry.get('contact_success', False)
                    max_force = float(entry.get('max_force', 0.0))
                    
                    if success and max_force >= 200.0:
                        reward = max_force / 200.0 
                        success_count += 1
                    else:
                        reward = -10.0 
                    self.data.append((pose_np, reward))
                print(f"Carregados {len(self.data)} dados. Sucessos Reais (>200N): {success_count}")
            except Exception as e:
                print(f"Erro no YAML: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pose, score = self.data[idx]
        return (torch.FloatTensor(self.static_grid), 
                torch.FloatTensor(pose), 
                torch.FloatTensor([score]))

# ==========================================
# 4. NÓ ROS 2 (CORRIGIDO)
# ==========================================
class NeuroGraspNode(Node):
    def __init__(self):
        super().__init__('neuro_grasp_trainer')
        
        self.pose_pub_ = self.create_publisher(PoseArray, '/actor_poses', 10)
        self.pc_pub_ = self.create_publisher(PointCloud2, '/input_grid', 10)
        self.timer = self.create_timer(0.01, self.train_step)
        
        self.static_grid = np.zeros((1, 25, 25, 25), dtype=np.float32)
        self.static_grid[0, 10:15, 10:15, 10:15] = 1.0 
        
        self.setup_training()
        
    def setup_training(self):
        try:
            self.dataset = GraspDataset(DATASET_FILE, self.static_grid)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        except Exception as e:
            self.get_logger().error(f"Dataset fail: {e}")
            sys.exit(1)

        self.actor = Actor().to(DEVICE)
        self.critic = Critic().to(DEVICE)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.critic_criterion = nn.MSELoss()
        
        self.epoch = 1 
        self.iter_loader = iter(self.dataloader)

        self.epoch_critic_losses = []
        self.epoch_actor_losses = []

    def train_step(self):
        if self.epoch > EPOCHS:
            if self.epoch == EPOCHS + 1:
                 self.get_logger().info(">>> TREINO FINALIZADO <<<")
                 self.epoch += 1
            self.publish_inference()
            return

        try:
            grids, real_poses, real_rewards = next(self.iter_loader)
        except StopIteration:
            self.finish_epoch()
            self.iter_loader = iter(self.dataloader)
            grids, real_poses, real_rewards = next(self.iter_loader)

        grids = grids.to(DEVICE)
        real_poses = real_poses.to(DEVICE)
        real_rewards = real_rewards.to(DEVICE)

        # 1. Crítico
        self.opt_critic.zero_grad()
        predicted_values = self.critic(grids, real_poses)
        loss_critic = self.critic_criterion(predicted_values, real_rewards)
        loss_critic.backward()
        self.opt_critic.step()

        # 2. Ator
        for p in self.critic.parameters(): p.requires_grad = False
        self.opt_actor.zero_grad()
        generated_poses = self.actor(grids)
        actor_value = self.critic(grids, generated_poses)
        loss_actor = -actor_value.mean() 
        loss_actor.backward()
        self.opt_actor.step()
        for p in self.critic.parameters(): p.requires_grad = True

        self.epoch_critic_losses.append(loss_critic.item())
        self.epoch_actor_losses.append(loss_actor.item())

    def finish_epoch(self):
        avg_c = np.mean(self.epoch_critic_losses) if self.epoch_critic_losses else 0
        avg_a = np.mean(self.epoch_actor_losses) if self.epoch_actor_losses else 0
        
        GLOBAL_HISTORY['epoch'].append(self.epoch)
        GLOBAL_HISTORY['critic_loss_avg'].append(avg_c)
        GLOBAL_HISTORY['actor_obj_avg'].append(avg_a)

        self.get_logger().info(f"Epoch [{self.epoch}/{EPOCHS}] | Critic MSE: {avg_c:.4f} | Actor Loss: {avg_a:.4f}")
        
        self.epoch_critic_losses = []
        self.epoch_actor_losses = []
        self.epoch += 1

        # Publica inferência a cada 5 épocas
        if self.epoch % 5 == 0: self.publish_inference()

    def publish_inference(self):
        with torch.no_grad():
            grid_sample = torch.tensor(self.static_grid).unsqueeze(0).to(DEVICE)
            msg = PoseArray()
            msg.header = Header(frame_id="world")
            msg.header.stamp = self.get_clock().now().to_msg()
            
            poses = self.actor(grid_sample.repeat(5, 1, 1, 1, 1)).cpu().numpy()
            
            for p in poses:
                pp = Pose()
                # --- CORREÇÃO AQUI: Cast explícito para float() ---
                # Numpy types crasham o ROS2 (Assertion PyFloat_Check failed)
                pp.position.x = float(p[0])
                pp.position.y = float(p[1])
                pp.position.z = float(p[2])
                pp.orientation.x = float(p[3])
                pp.orientation.y = float(p[4])
                pp.orientation.z = float(p[5])
                pp.orientation.w = float(p[6])
                msg.poses.append(pp)
            
            self.pose_pub_.publish(msg)
            self.publish_pc2()

    def publish_pc2(self):
        points = []
        indices = np.argwhere(self.static_grid[0] > 0.5)
        scale = 0.02
        for idx in indices:
            points.append(struct.pack('fff', idx[0]*scale, idx[1]*scale, idx[2]*scale))
        msg = PointCloud2()
        msg.header = Header(frame_id="world")
        msg.height = 1; msg.width = len(points)
        msg.fields = [PointField(name='x', offset=0, datatype=7, count=1), 
                      PointField(name='y', offset=4, datatype=7, count=1),
                      PointField(name='z', offset=8, datatype=7, count=1)]
        msg.point_step = 12; msg.row_step = 12 * len(points)
        msg.data = b''.join(points)
        self.pc_pub_.publish(msg)

# ==========================================
# 5. MAIN
# ==========================================
def main(args=None):
    rclpy.init(args=args)
    node = NeuroGraspNode()
    
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Treinamento NeuroGrasp', fontsize=14)
    
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Época')

    ax1.set_title('Critic Loss (MSE)', color='darkred')
    ax2.set_title('Actor Loss (-Reward)', color='darkblue')

    print("Iniciando Interface Gráfica...")

    try:
        while rclpy.ok():
            if len(GLOBAL_HISTORY['epoch']) > 0:
                epochs = GLOBAL_HISTORY['epoch']
                c_avg = GLOBAL_HISTORY['critic_loss_avg']
                a_avg = GLOBAL_HISTORY['actor_obj_avg']
                
                ax1.clear(); ax2.clear()
                
                # Re-aplica estilos pois o clear() limpa tudo
                ax1.grid(True, linestyle='--', alpha=0.7)
                ax1.set_title('Critic Loss (MSE)', color='darkred')
                ax1.plot(epochs, c_avg, 'r-o', label='Critic')
                
                ax2.grid(True, linestyle='--', alpha=0.7)
                ax2.set_title('Actor Loss (-Reward)', color='darkblue')
                ax2.plot(epochs, a_avg, 'b-s', label='Actor')

                plt.draw()
                plt.pause(0.5) 
            else:
                plt.pause(1.0)
                
    except KeyboardInterrupt:
        print("\nFinalizando...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close()

if __name__ == "__main__":
    main()