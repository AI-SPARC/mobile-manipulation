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
import glob
import time
import math
import warnings

import rclpy
from rclpy.node import Node

# ==========================================
# 0. CONFIGURAÇÃO DE OTIMIZAÇÃO
# ==========================================
torch.set_float32_matmul_precision('high')

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "neuro_grasp_model_optimized.pth")

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.allow_tf32 = False 
    GPU_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU"

print(f"\n>>> HARDWARE: {GPU_NAME}")
print(f">>> MODO: TREINO COM RETOMADA (RESUME)")

# ==========================================
# 1.1 FÍSICA E GRID
# ==========================================
CUBE_SIDE = 0.06
VOXEL_RES = 0.001  
GRID_SIZE = int(np.ceil(CUBE_SIDE / VOXEL_RES)) 
WORKSPACE_LIMIT = CUBE_SIDE / 2.0 
print(f">>> GRID: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE}")

LR_ACTOR = 2e-5    
LR_CRITIC = 2e-4   
BATCH_SIZE = 4     
EPOCHS = 200       
DATASET_ROOT = "/home/momesso/isaacsim/toma"
NUM_WORKERS = 4   

# Limite máximo de recompensa (para evitar explosão numérica)
MAX_REWARD_CAP = 500.0 

GLOBAL_HISTORY = {'epoch': [], 'c_loss': [], 'a_loss': []}

# ==========================================
# 2. UTILITÁRIOS
# ==========================================
def npy_to_voxel_grid(npy_path):
    grid = np.zeros((1, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    if not os.path.exists(npy_path): return grid, None

    try:
        raw_points = np.load(npy_path)
        mask = (np.abs(raw_points[:, 0]) <= WORKSPACE_LIMIT) & \
               (np.abs(raw_points[:, 1]) <= WORKSPACE_LIMIT) & \
               (np.abs(raw_points[:, 2]) <= WORKSPACE_LIMIT)
        points = raw_points[mask]
        
        if len(points) > 0:
            norm_pts = (points + WORKSPACE_LIMIT) / (2 * WORKSPACE_LIMIT)
            scaled_pts = norm_pts * (GRID_SIZE - 0.001)
            indices = np.floor(scaled_pts).astype(int)
            indices = np.clip(indices, 0, GRID_SIZE - 1)
            grid[0, indices[:,0], indices[:,1], indices[:,2]] = 1.0
            return grid, points
    except:
        pass
    return grid, None

# ==========================================
# 3. REDES NEURAIS
# ==========================================
class CubeEncoder(nn.Module):
    def __init__(self):
        super(CubeEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2) 
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self.flat_size = x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.encoder = CubeEncoder()
        self.fc1 = nn.Linear(self.encoder.flat_size, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.out_pos = nn.Linear(512, 3) 
        self.out_quat = nn.Linear(512, 4) 

    def forward(self, grid):
        x = F.relu(self.fc1(self.encoder(grid)))
        x = F.relu(self.fc2(x))
        pos = torch.tanh(self.out_pos(x)) * WORKSPACE_LIMIT 
        quat = F.normalize(self.out_quat(x), p=2, dim=1)
        return torch.cat([pos, quat], dim=1)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.encoder = CubeEncoder()
        self.pose_fc = nn.Linear(7, 64) 
        self.fc1 = nn.Linear(self.encoder.flat_size + 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, grid, pose):
        grid_feat = self.encoder(grid)
        pose_feat = F.relu(self.pose_fc(pose))
        combined = torch.cat([grid_feat, pose_feat], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.out(x) 

# ==========================================
# 4. DATASET (COM LIMITADOR)
# ==========================================
class MultiGraspDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.data = []
        if not os.path.exists(root_dir): sys.exit(1)
            
        run_dirs = glob.glob(os.path.join(root_dir, "run_*"))
        print(f">> Carregando metadados de {len(run_dirs)} pastas...")
        
        for r_dir in run_dirs:
            yaml_path = os.path.join(r_dir, "results_safe.yaml")
            npy_path = os.path.join(r_dir, "object_pointcloud.npy")
            if not os.path.exists(yaml_path) or not os.path.exists(npy_path): continue
            
            grid, _ = npy_to_voxel_grid(npy_path)
            
            try:
                with open(yaml_path, 'r') as stream:
                    doc_list = yaml.safe_load(stream)
                    if not doc_list: continue
                    if not isinstance(doc_list, list): doc_list = [doc_list]
                    for entry in doc_list:
                        if not isinstance(entry, dict): continue
                        p = entry.get('pose_pos', [0,0,0])
                        r = entry.get('pose_rot', [0,0,0,1])
                        pose_np = np.array(p + r, dtype=np.float32)
                        success = entry.get('contact_success', False)
                        max_force = float(entry.get('max_force', 0.0))
                        
                        if success and max_force >= 10.0:
                            # ------------------------------------------------
                            # RECOMPENSA SEGURA
                            # ------------------------------------------------
                            raw_reward = max_force / 50.0
                            reward = min(raw_reward, MAX_REWARD_CAP)
                        else:
                            reward = -5.0 
                        self.data.append((grid, pose_np, reward))
            except: pass
        print(f">> Dataset pronto: {len(self.data)} amostras.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grid, pose, score = self.data[idx]
        return (torch.from_numpy(grid), torch.from_numpy(pose), torch.FloatTensor([score]))

# ==========================================
# 5. TREINAMENTO
# ==========================================
class NeuroGraspNode(Node):
    def __init__(self):
        super().__init__('neuro_grasp_trainer')
        
        self.dataset = MultiGraspDataset(DATASET_ROOT)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            drop_last=True,
            num_workers=NUM_WORKERS,    
            pin_memory=True,
            persistent_workers=True
        )
        
        # Inicializa Modelos
        self.actor = Actor().to(DEVICE)
        self.critic = Critic().to(DEVICE)
        
        # Inicializa Otimizadores
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.sched_actor = optim.lr_scheduler.CosineAnnealingLR(self.opt_actor, T_max=EPOCHS, eta_min=1e-6)
        self.sched_critic = optim.lr_scheduler.CosineAnnealingLR(self.opt_critic, T_max=EPOCHS, eta_min=1e-6)
        
        self.critic_criterion = nn.MSELoss()
        
        self.epoch = 1
        
        # ------------------------------------------------------
        # LÓGICA DE RECUPERAÇÃO (RESUME)
        # ------------------------------------------------------
        if os.path.exists(MODEL_PATH):
            try:
                print(f"\n>>> ENCONTRADO CHECKPOINT: {MODEL_PATH}")
                checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
                
                # Carrega Pesos
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                
                # Recupera Época
                self.epoch = checkpoint['epoch'] + 1
                
                # Tenta carregar otimizadores (se existirem no arquivo antigo)
                if 'opt_actor_state_dict' in checkpoint:
                    self.opt_actor.load_state_dict(checkpoint['opt_actor_state_dict'])
                    self.opt_critic.load_state_dict(checkpoint['opt_critic_state_dict'])
                    print(">>> Estado dos Otimizadores recuperado com sucesso.")
                else:
                    print(">>> AVISO: Checkpoint antigo sem otimizadores. Reiniciando Adam.")

                print(f">>> MODELOS CARREGADOS! Retomando da Época {self.epoch}")
                
            except Exception as e:
                print(f">>> ERRO AO CARREGAR CHECKPOINT: {e}")
                print(">>> Iniciando treinamento do zero (Backup corrompido ou incompatível).")
        else:
            print(">>> NENHUM CHECKPOINT ENCONTRADO. Iniciando do zero.")
        # ------------------------------------------------------

        self.iter_loader = iter(self.dataloader)
        self.total_batches = len(self.dataloader)
        self.batch_count = 0 
        
        self.epoch_critic_losses = []
        self.epoch_actor_losses = []
        self.epoch_start_time = time.time()
        
        self.create_timer(0.001, self.train_loop)

    def save_checkpoint(self):
        checkpoint = {
            'epoch': self.epoch,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            # Salvando Otimizadores agora para garantir resume perfeito no futuro
            'opt_actor_state_dict': self.opt_actor.state_dict(),
            'opt_critic_state_dict': self.opt_critic.state_dict()
        }
        # Salva o atual
        torch.save(checkpoint, MODEL_PATH)
        
        # Salva Backup a cada 25 épocas
        if self.epoch % 25 == 0:
            bkp_path = MODEL_PATH.replace(".pth", f"_ep{self.epoch}.pth")
            torch.save(checkpoint, bkp_path)

    def train_loop(self):
        if self.epoch > EPOCHS:
            self.get_logger().info("TREINAMENTO FINALIZADO.")
            time.sleep(5)
            sys.exit(0)

        try:
            grids, real_poses, real_rewards = next(self.iter_loader)
        except StopIteration:
            self.finish_epoch()
            self.iter_loader = iter(self.dataloader)
            grids, real_poses, real_rewards = next(self.iter_loader)

        self.batch_count += 1
        grids = grids.to(DEVICE, non_blocking=True)
        real_poses = real_poses.to(DEVICE, non_blocking=True)
        real_rewards = real_rewards.to(DEVICE, non_blocking=True)

        # --- CRITIC ---
        self.opt_critic.zero_grad(set_to_none=True)
        predicted_rewards = self.critic(grids, real_poses)
        loss_c = self.critic_criterion(predicted_rewards, real_rewards)
        loss_c.backward()
        
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        self.opt_critic.step()

        # --- ACTOR ---
        for p in self.critic.parameters(): p.requires_grad = False
        self.opt_actor.zero_grad(set_to_none=True)
        generated_poses = self.actor(grids)
        actor_value = self.critic(grids, generated_poses)
        loss_a = -actor_value.mean()
        loss_a.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        
        self.opt_actor.step()
        for p in self.critic.parameters(): p.requires_grad = True

        val_loss_c = loss_c.item()
        val_loss_a = loss_a.item()
        self.epoch_critic_losses.append(val_loss_c)
        self.epoch_actor_losses.append(val_loss_a)

        if self.batch_count % 5 == 0:
            print(f"Ep {self.epoch} [{self.batch_count}/{self.total_batches}] | "
                  f"Critic L: {val_loss_c:.5f} | Actor L: {val_loss_a:.5f}", end='\r')

        del grids, real_poses, real_rewards, predicted_rewards, loss_c, loss_a

    def finish_epoch(self):
        duration = time.time() - self.epoch_start_time
        avg_c = np.mean(self.epoch_critic_losses)
        avg_a = np.mean(self.epoch_actor_losses)
        
        self.sched_actor.step()
        self.sched_critic.step()
        
        self.save_checkpoint()
        
        GLOBAL_HISTORY['epoch'].append(self.epoch)
        GLOBAL_HISTORY['c_loss'].append(avg_c)
        GLOBAL_HISTORY['a_loss'].append(avg_a)
        
        print(f"\n>>> FIM EP {self.epoch} | Tempo: {duration:.1f}s | "
              f"Média C: {avg_c:.4f} | Média A: {avg_a:.4f}")
        
        self.epoch_critic_losses = []
        self.epoch_actor_losses = []
        self.batch_count = 0
        self.epoch += 1
        self.epoch_start_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = NeuroGraspNode()
    t_ros = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t_ros.start()
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
    
    print(">>> INICIANDO TREINO (MODO RESUME)...")
    
    try:
        while rclpy.ok():
            time.sleep(2.0) 
            if len(GLOBAL_HISTORY['epoch']) > 0:
                ax1.clear(); ax2.clear()
                ax1.set_title("Critic Loss"); ax1.plot(GLOBAL_HISTORY['epoch'], GLOBAL_HISTORY['c_loss'], 'r')
                ax2.set_title("Actor Loss"); ax2.plot(GLOBAL_HISTORY['epoch'], GLOBAL_HISTORY['a_loss'], 'b')
                plt.draw(); plt.pause(0.1)
            
            if node.epoch > EPOCHS:
                break
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close()

if __name__ == "__main__":
    main()