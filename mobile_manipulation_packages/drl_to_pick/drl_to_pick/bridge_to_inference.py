#!/usr/bin/env python3
"""
GraspNet ROS2 Node - Franka Panda

Nó ROS2 que:
1. Recebe PointCloud2 do tópico /object_pointcloud
2. Envia para o servidor GraspNet TCP
3. Publica poses de grasp em /grasp_poses (PoseArray)

Convenção GraspNet:
- Approach = +X (R[:, 0])
- Finger opening = Y (R[:, 1])
"""

import rclpy
from rclpy.node import Node
import socket
import pickle
import numpy as np
import math
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose
import sensor_msgs_py.point_cloud2 as pc2
import time


class GraspProviderNode(Node):
    def __init__(self):
        super().__init__('grasp_provider_node')
        
        self.subscription = self.create_subscription(
            PointCloud2, '/object_pointcloud', self.listener_callback, 10)
        
        self.pub_grasps = self.create_publisher(PoseArray, '/grasp_poses', 10)
        
        self.declare_parameter('server_host', 'localhost')
        self.declare_parameter('server_port', 5000)
        self.declare_parameter('score_threshold', 0.15)
        self.declare_parameter('max_grasps', 50)
        
        self.server_host = self.get_parameter('server_host').value
        self.server_port = self.get_parameter('server_port').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.max_grasps = self.get_parameter('max_grasps').value
        
        self.declare_parameter('enable_crop', False)
        self.declare_parameter('crop_x_min', 0.2)
        self.declare_parameter('crop_x_max', 1.0)
        self.declare_parameter('crop_y_min', -0.5)
        self.declare_parameter('crop_y_max', 0.5)
        self.declare_parameter('crop_z_min', -0.5)
        self.declare_parameter('crop_z_max', 1.5)
        
        self.enable_crop = self.get_parameter('enable_crop').value
        self.crop_bounds = {
            'x': (self.get_parameter('crop_x_min').value, self.get_parameter('crop_x_max').value),
            'y': (self.get_parameter('crop_y_min').value, self.get_parameter('crop_y_max').value),
            'z': (self.get_parameter('crop_z_min').value, self.get_parameter('crop_z_max').value),
        }
        
        self.last_centroid = None
        
        self.get_logger().info('='*60)
        self.get_logger().info('🤖 GRASP PROVIDER NODE - FRANKA PANDA')
        self.get_logger().info('='*60)
        self.get_logger().info(f'   Server: {self.server_host}:{self.server_port}')
        self.get_logger().info(f'   Score threshold: {self.score_threshold}')
        self.get_logger().info(f'   Max grasps: {self.max_grasps}')
        self.get_logger().info(f'   Crop enabled: {self.enable_crop}')
        self.get_logger().info('   Aguardando point cloud...')

    def get_grasps_from_server(self, points: np.ndarray) -> dict:
        """
        Envia point cloud para o servidor e recebe grasps.
        
        Args:
            points: (N, 3) point cloud já pré-processada
            
        Returns:
            dict com 'scores', 'pred_grasps_cam', 'widths', 'depths'
        """
        try:
            self.get_logger().info(f'   📤 Enviando {points.shape[0]} pontos para servidor...')
            t0 = time.time()
            
            pc_batch = np.expand_dims(points.astype(np.float32), axis=0)
            
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(30.0)
            client.connect((self.server_host, self.server_port))
            
            data = pickle.dumps(pc_batch)
            client.sendall(len(data).to_bytes(8, 'big'))
            client.sendall(data)
            
            raw_msglen = client.recv(8)
            if not raw_msglen:
                self.get_logger().error('   ❌ Servidor fechou conexão')
                return None
            msglen = int.from_bytes(raw_msglen, 'big')
            
            chunks = []
            bytes_recd = 0
            while bytes_recd < msglen:
                chunk = client.recv(min(msglen - bytes_recd, 4096000))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_recd += len(chunk)
            
            client.close()
            
            result = pickle.loads(b"".join(chunks))
            dt = time.time() - t0
            
            n_grasps = len(result.get('scores', []))
            self.get_logger().info(f'Recebido {n_grasps} grasps em {dt:.2f}s')

            self.get_logger().info(f'Centróide salvo: {self.last_centroid}')
                        
            return result
            
        except socket.timeout:
            self.get_logger().error('   ❌ Timeout na conexão com servidor')
            return None
        except ConnectionRefusedError:
            self.get_logger().error(f'   ❌ Conexão recusada - servidor rodando em {self.server_host}:{self.server_port}?')
            return None
        except Exception as e:
            self.get_logger().error(f'   ❌ Erro: {e}')
            return None

    def crop_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """Aplica crop na point cloud"""
        if not self.enable_crop:
            return points
        
        mask = (
            (points[:, 0] >= self.crop_bounds['x'][0]) & 
            (points[:, 0] <= self.crop_bounds['x'][1]) &
            (points[:, 1] >= self.crop_bounds['y'][0]) & 
            (points[:, 1] <= self.crop_bounds['y'][1]) &
            (points[:, 2] >= self.crop_bounds['z'][0]) & 
            (points[:, 2] <= self.crop_bounds['z'][1])
        )
        
        cropped = points[mask]
        self.get_logger().info(f'   Crop: {points.shape[0]} → {cropped.shape[0]} pontos')
        
        return cropped

    def preprocess_pointcloud(self, points: np.ndarray) -> tuple:
        """
        Pré-processa point cloud para o servidor.
        
        O servidor espera:
        - Point cloud centralizada na origem
        - Z offset de 0.5m (simula distância da câmera)
        
        Returns:
            (points_processed, centroid) - centroid para destransformar depois
        """
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        points_centered[:, 2] += 0.5
        
        return points_centered, centroid

    def transform_grasps_to_world(self, grasps_cam: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """
        Transforma grasps do frame do servidor para o frame world.
        
        O servidor retorna grasps com:
        - Z offset de 0.5m
        - Centralizado na origem
        
        Precisamos:
        - Remover Z offset
        - Adicionar centroid original
        """
        grasps_world = []
        
        for g_cam in grasps_cam:
            g_world = g_cam.copy()
            
            g_world[2, 3] -= 0.5  
            g_world[:3, 3] += centroid  
            
            grasps_world.append(g_world)
        
        self.get_logger().info(f'Grasp antes da transformação: {grasps_cam[0][:3, 3]}')
        self.get_logger().info(f'Grasp depois da transformação: {grasps_world[0][:3, 3]}')
        return np.array(grasps_world)

    def matrix_to_quaternion(self, R: np.ndarray) -> tuple:


        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        # Normaliza
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        return (x/norm, y/norm, z/norm, w/norm)

    def publish_grasps(self, header, grasps: np.ndarray, scores: np.ndarray, widths: np.ndarray = None):
        """
        Publica grasps como PoseArray.
        
        A orientação publicada segue a convenção GraspNet:
        - Approach = +X local
        - Finger opening = Y local
        """
        pose_array = PoseArray()
        pose_array.header = header
        
        mask = scores >= self.score_threshold
        valid_indices = np.where(mask)[0]
        sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
        
        sorted_indices = sorted_indices[:self.max_grasps]
        
        self.get_logger().info(f'   📊 Grasps válidos: {len(sorted_indices)} (threshold={self.score_threshold})')
        
        for i, idx in enumerate(sorted_indices):
            g = grasps[idx]
            
            pose = Pose()
            
            pose.position.x = float(g[0, 3])
            pose.position.y = float(g[1, 3])
            pose.position.z = float(g[2, 3])
            
            R = g[:3, :3]
            qx, qy, qz, qw = self.matrix_to_quaternion(R)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            
            pose_array.poses.append(pose)
            
            # Log top 5
            if i < 5:
                width_str = f", width={widths[idx]*100:.1f}cm" if widths is not None else ""
                self.get_logger().info(
                    f'      #{i+1}: score={scores[idx]:.3f}, '
                    f'pos=[{g[0,3]:.3f}, {g[1,3]:.3f}, {g[2,3]:.3f}]{width_str}'
                )
        
        self.pub_grasps.publish(pose_array)
        self.get_logger().info(f'   ✅ Publicados {len(pose_array.poses)} grasps em /grasp_poses')

    def listener_callback(self, msg: PointCloud2):
        """Callback principal quando recebe point cloud"""
        self.get_logger().info('')
        self.get_logger().info('='*50)
        self.get_logger().info('📷 Point cloud recebida!')
        
        points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc_list = list(points_gen)
        
        if not pc_list:
            self.get_logger().warn('   ⚠️ Point cloud vazia!')
            return
        
        pc_np = np.array(pc_list, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        pc_np = pc_np.view(np.float32).reshape(-1, 3)
        
        self.get_logger().info(f'   Pontos: {pc_np.shape[0]}')
        self.get_logger().info(f'   Bounds: X[{pc_np[:,0].min():.3f}, {pc_np[:,0].max():.3f}]')
        self.get_logger().info(f'           Y[{pc_np[:,1].min():.3f}, {pc_np[:,1].max():.3f}]')
        self.get_logger().info(f'           Z[{pc_np[:,2].min():.3f}, {pc_np[:,2].max():.3f}]')
        
        pc_cropped = self.crop_pointcloud(pc_np)
        
        if pc_cropped.shape[0] < 100:
            self.get_logger().warn('   ⚠️ Poucos pontos após crop!')
            return
        
        pc_processed, centroid = self.preprocess_pointcloud(pc_cropped)
        self.last_centroid = centroid
        
        self.get_logger().info(f'   Centróide: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]')
        
        result = self.get_grasps_from_server(pc_processed)
        
        if result is None:
            self.get_logger().warn('   ⚠️ Falha na comunicação com servidor')
            return
        
        scores = result.get('scores', np.array([]))
        grasps_cam = result.get('pred_grasps_cam', np.array([]))
        widths = result.get('widths', None)
        
        if len(scores) == 0:
            self.get_logger().warn('   ⚠️ Nenhum grasp encontrado')
            return
        
        self.get_logger().info(f'   Max score: {scores.max():.3f}')
        
        grasps_world = self.transform_grasps_to_world(grasps_cam, centroid)
        
        self.publish_grasps(msg.header, grasps_world, scores, widths)


def main(args=None):
    rclpy.init(args=args)
    node = GraspProviderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()