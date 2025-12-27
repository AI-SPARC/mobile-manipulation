#!/usr/bin/env python3
"""
GraspNet ROS2 Node - Franka Panda

Nó ROS2 que:
1. Recebe PointCloud2 do tópico /depth_pcl
2. Transforma para o frame world usando TF2
3. Filtra NaN e faz crop da região de interesse
4. Envia para o servidor GraspNet TCP
5. Publica poses de grasp em /grasp_poses (PoseArray) no frame world

Convenção GraspNet:
- Approach = +X (R[:, 0])
- Finger opening = Y (R[:, 1])
"""

import rclpy
from rclpy.node import Node
import socket
import pickle
import numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose
import sensor_msgs_py.point_cloud2 as pc2
import time


import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class GraspProviderNode(Node):
    def __init__(self):
        super().__init__('grasp_provider_node')
        
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        
        self.subscription = self.create_subscription(
            PointCloud2, '/filtered_points', self.listener_callback, 10)
        
 
        self.pub_grasps = self.create_publisher(PoseArray, '/grasp_poses', 10)
        
        
        self.declare_parameter('server_host', 'localhost')
        self.declare_parameter('server_port', 5000)
        self.declare_parameter('score_threshold', 0.15)
        self.declare_parameter('max_grasps', 50)
        
        self.server_host = self.get_parameter('server_host').value
        self.server_port = self.get_parameter('server_port').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.max_grasps = self.get_parameter('max_grasps').value
        
        self.declare_parameter('target_frame', 'world')
        self.target_frame = self.get_parameter('target_frame').value
        
       
        self.declare_parameter('enable_crop', True)
        self.declare_parameter('crop_x_min', -1.0)
        self.declare_parameter('crop_x_max', 1.0)
        self.declare_parameter('crop_y_min', -1.0)
        self.declare_parameter('crop_y_max', 1.0)
        self.declare_parameter('crop_z_min', 0.0)   
        self.declare_parameter('crop_z_max', 0.5)    
        
        self.enable_crop = self.get_parameter('enable_crop').value
        self.crop_bounds = {
            'x': (self.get_parameter('crop_x_min').value, self.get_parameter('crop_x_max').value),
            'y': (self.get_parameter('crop_y_min').value, self.get_parameter('crop_y_max').value),
            'z': (self.get_parameter('crop_z_min').value, self.get_parameter('crop_z_max').value),
        }
        
        #
        self.declare_parameter('max_points', 100000)
        self.max_points = self.get_parameter('max_points').value
        
        self.get_logger().info('='*60)
        self.get_logger().info('BRIDGE TO INFERENCE')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Server: {self.server_host}:{self.server_port}')
        self.get_logger().info(f'Target frame: {self.target_frame}')
        self.get_logger().info(f'Score threshold: {self.score_threshold}')
        self.get_logger().info(f'Max grasps: {self.max_grasps}')
        self.get_logger().info(f'Crop enabled: {self.enable_crop}')
        if self.enable_crop:
            self.get_logger().info(f'Crop X: [{self.crop_bounds["x"][0]}, {self.crop_bounds["x"][1]}]')
            self.get_logger().info(f'Crop Y: [{self.crop_bounds["y"][0]}, {self.crop_bounds["y"][1]}]')
            self.get_logger().info(f'Crop Z: [{self.crop_bounds["z"][0]}, {self.crop_bounds["z"][1]}]')
        self.get_logger().info(f'Max points: {self.max_points}')
        self.get_logger().info('Aguardando point cloud...')

    def get_transform(self, source_frame: str, target_frame: str) -> np.ndarray:
        """
        Obtém transformação TF2 como matriz 4x4.
        
        Returns:
            Matriz 4x4 de transformação ou None se falhar
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            t = transform.transform.translation
            translation = np.array([t.x, t.y, t.z])
            
            q = transform.transform.rotation
            R = self.quaternion_to_matrix(q.x, q.y, q.z, q.w)
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = translation
            
            return T
            
        except TransformException as e:
            self.get_logger().warn(f'TF2 falhou: {e}')
            return None

    def quaternion_to_matrix(self, x, y, z, w) -> np.ndarray:
        """Converte quaternion para matriz de rotação 3x3."""
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        return R

    def transform_pointcloud(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Transforma point cloud usando matriz 4x4.
        
        Args:
            points: (N, 3) point cloud
            T: (4, 4) matriz de transformação
            
        Returns:
            Point cloud transformada (N, 3)
        """
        
        ones = np.ones((len(points), 1))
        points_h = np.hstack([points, ones])  
        
        points_transformed = (T @ points_h.T).T
        
       
        return points_transformed[:, :3]

    def get_grasps_from_server(self, points: np.ndarray) -> dict:
        """Envia point cloud para o servidor e recebe grasps."""
        try:
            self.get_logger().info(f'Enviando {points.shape[0]} pontos para servidor...')
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
                self.get_logger().error('Servidor fechou conexão')
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
            
            return result
            
        except socket.timeout:
            self.get_logger().error('Timeout na conexão com servidor')
            return None
        except ConnectionRefusedError:
            self.get_logger().error(f'Conexão recusada - servidor rodando em {self.server_host}:{self.server_port}?')
            return None
        except Exception as e:
            self.get_logger().error(f'Erro: {e}')
            return None

    def filter_and_crop_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """Filtra NaN/Inf e aplica crop na point cloud."""
        original_count = len(points)
        
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        
        nan_removed = original_count - len(points)
        if nan_removed > 0:
            self.get_logger().info(f'Removidos {nan_removed} pontos NaN/Inf')
        
        if len(points) == 0:
            return points
        
       
        dist_mask = np.linalg.norm(points, axis=1) < 10.0
        points = points[dist_mask]
        
        
        if self.enable_crop and len(points) > 0:
            crop_mask = (
                (points[:, 0] >= self.crop_bounds['x'][0]) & 
                (points[:, 0] <= self.crop_bounds['x'][1]) &
                (points[:, 1] >= self.crop_bounds['y'][0]) & 
                (points[:, 1] <= self.crop_bounds['y'][1]) &
                (points[:, 2] >= self.crop_bounds['z'][0]) & 
                (points[:, 2] <= self.crop_bounds['z'][1])
            )
            points = points[crop_mask]
            self.get_logger().info(f'Após crop (world frame): {len(points)} pontos')
        
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            self.get_logger().info(f'Downsample para {self.max_points} pontos')
        
        return points

    def transform_grasps_to_world(self, grasps_cam: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """
        Transforma grasps para o frame world.
        
        O servidor retorna grasps no frame centralizado (origem = centróide do objeto).
        Precisamos adicionar o centróide de volta para obter coordenadas no frame world.
        
        Args:
            grasps_cam: Grasps no frame centralizado (do servidor)
            centroid: Centróide original da point cloud (retornado pelo servidor)
            
        Returns:
            Grasps no frame world
        """
        grasps_world = []
        
        for g_cam in grasps_cam:
            g_world = g_cam.copy()
            
            g_world[:3, 3] += centroid
            
            grasps_world.append(g_world)
        
        return np.array(grasps_world)

    def matrix_to_quaternion(self, R: np.ndarray) -> tuple:
        """Converte matriz de rotação 3x3 para quaternion (x, y, z, w)."""
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
        
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        return (x/norm, y/norm, z/norm, w/norm)

    def publish_grasps(self, grasps: np.ndarray, scores: np.ndarray, widths: np.ndarray = None):
        """Publica grasps como PoseArray no frame world."""
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.target_frame  
        
        mask = scores >= self.score_threshold
        valid_indices = np.where(mask)[0]
        sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
        

        sorted_indices = sorted_indices[:self.max_grasps]
        
        self.get_logger().info(f'Grasps válidos: {len(sorted_indices)} (threshold={self.score_threshold})')
        
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
            
            if i < 5:
                width_str = f", width={widths[idx]*100:.1f}cm" if widths is not None else ""
                self.get_logger().info(
                    f'      #{i+1}: score={scores[idx]:.3f}, '
                    f'pos=[{g[0,3]:.3f}, {g[1,3]:.3f}, {g[2,3]:.3f}]{width_str}'
                )
        
        self.pub_grasps.publish(pose_array)
        self.get_logger().info(f'Publicados {len(pose_array.poses)} grasps em /grasp_poses (frame: {self.target_frame})')

    def listener_callback(self, msg: PointCloud2):
        """Callback principal quando recebe point cloud"""
        self.get_logger().info('')
        self.get_logger().info('='*50)
        self.get_logger().info('Point cloud recebida!')
        
        source_frame = msg.header.frame_id
        self.get_logger().info(f'   Frame de origem: {source_frame}')
        
        T_cam_to_world = self.get_transform(source_frame, self.target_frame)
        
        if T_cam_to_world is None:
            self.get_logger().error(f'Não foi possível obter TF: {source_frame} -> {self.target_frame}')
            return
        
        self.get_logger().info(f'TF obtido: {source_frame} -> {self.target_frame}')
        
        points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc_list = list(points_gen)
        
        if not pc_list:
            self.get_logger().warn('Point cloud vazia!')
            return
        
        pc_np = np.array(pc_list, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        pc_np = pc_np.view(np.float32).reshape(-1, 3)
        
        self.get_logger().info(f'Pontos raw (camera frame): {pc_np.shape[0]}')
        
        pc_world = self.transform_pointcloud(pc_np, T_cam_to_world)
        self.get_logger().info(f'Pontos transformados para world frame')
        
        pc_filtered = self.filter_and_crop_pointcloud(pc_world)
        
        if len(pc_filtered) < 100:
            self.get_logger().warn(f'Poucos pontos após filtro ({len(pc_filtered)})')
            self.get_logger().warn('Ajuste os parâmetros de crop para o frame world!')
            return
        
        self.get_logger().info(f'Bounds (world frame):')
        self.get_logger().info(f'X[{pc_filtered[:,0].min():.3f}, {pc_filtered[:,0].max():.3f}]')
        self.get_logger().info(f'Y[{pc_filtered[:,1].min():.3f}, {pc_filtered[:,1].max():.3f}]')
        self.get_logger().info(f'Z[{pc_filtered[:,2].min():.3f}, {pc_filtered[:,2].max():.3f}]')
        
        result = self.get_grasps_from_server(pc_filtered)
        
        if result is None:
            self.get_logger().warn('Falha na comunicação com servidor')
            return
        
        scores = result.get('scores', np.array([]))
        grasps_cam = result.get('pred_grasps_cam', np.array([]))
        widths = result.get('widths', None)
        centroid = result.get('centroid', np.zeros(3)) 
        
        if len(scores) == 0:
            self.get_logger().warn('Nenhum grasp encontrado')
            return
        
        self.get_logger().info(f'Centróide (do servidor): [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]')
        self.get_logger().info(f'Max score: {scores.max():.3f}')
        
        grasps_world = self.transform_grasps_to_world(grasps_cam, centroid)
        
        self.publish_grasps(grasps_world, scores, widths)


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