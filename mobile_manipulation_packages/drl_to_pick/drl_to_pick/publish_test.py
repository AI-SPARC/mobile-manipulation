import rclpy
from rclpy.node import Node
import numpy as np
import os

from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg

class NpyToPointCloudPublisher(Node):

    def __init__(self):
        super().__init__('npy_pc_publisher')
        
        
        self.declare_parameter('npy_path', '/home/momesso/isaac-sim/toma/run_0/object_pointcloud.npy')
        self.publisher_ = self.create_publisher(PointCloud2, '/object_pointcloud', 10)
        
        
        self.timer = self.create_timer(2.0, self.timer_callback)
        
        self.get_logger().info('Nó de leitura de NPY iniciado...')

    def create_pointcloud2(self, points):
        """
        Converte um array numpy (N, 3) para sensor_msgs/PointCloud2
        """
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'  
        
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg.point_step = 12  
        msg.row_step = msg.point_step * points.shape[0]
        msg.data = points.astype(np.float32).tobytes()
        
        return msg

    def timer_callback(self):
        npy_path = self.get_parameter('npy_path').get_parameter_value().string_value
        
        if not os.path.exists(npy_path):
            self.get_logger().error(f'Arquivo não encontrado: {npy_path}')
            return

        try:
            pc_np = np.load(npy_path)
            
            if pc_np.shape[1] > 3:
                pc_np = pc_np[:, :3]

            pc_msg = self.create_pointcloud2(pc_np)
            
            self.publisher_.publish(pc_msg)
            self.get_logger().info(f'Publicada nuvem de pontos com {len(pc_np)} pontos de {npy_path}')
            
        except Exception as e:
            self.get_logger().error(f'Erro ao carregar ou publicar NPY: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = NpyToPointCloudPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()