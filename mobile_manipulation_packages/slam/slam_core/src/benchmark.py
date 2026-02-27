import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import message_filters
import math

class OdomBenchmarkNode(Node):
    def __init__(self):
        super().__init__('odom_benchmark_node')
        
        # Cria os subscribers usando message_filters (não disparam callbacks sozinhos)
        self.gt_sub = message_filters.Subscriber(self, Odometry, 'ground_truth')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        
        # Sincronizador de tempo aproximado
        # queue_size: tamanho da fila de mensagens a manter
        # slop: tolerância de tempo em segundos entre os timestamps (ex: 50ms)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.gt_sub, self.odom_sub], 
            queue_size=10, 
            slop=0.05
        )
        
        # Registra a função de callback que será chamada quando houver o 'match'
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info("Nó de Benchmark de Odometria iniciado. Aguardando mensagens sincronizadas...")

    def sync_callback(self, gt_msg, odom_msg):
        # Extrai as posições de ambas as mensagens
        gt_pos = gt_msg.pose.pose.position
        odom_pos = odom_msg.pose.pose.position
        
        # Calcula a distância euclidiana (erro de translação real)
        dx = gt_pos.x - odom_pos.x
        dy = gt_pos.y - odom_pos.y
        dz = gt_pos.z - odom_pos.z
        distance_error = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calcula a norma do ground truth (distância da origem) para usar como base da porcentagem
        gt_norm = math.sqrt(gt_pos.x**2 + gt_pos.y**2 + gt_pos.z**2)
        
        # Calcula a diferença percentual em relação à posição do ground truth
        if gt_norm > 1e-6:  # Evita divisão por zero
            percent_diff = (distance_error / gt_norm) * 100.0
        else:
            percent_diff = 0.0 
            
        # Loga os resultados
        self.get_logger().info(
            f"Tempo: {gt_msg.header.stamp.sec}.{gt_msg.header.stamp.nanosec:09d} | "
            f"Erro Euclidiano: {distance_error:.4f}m | "
            f"Diferença Relativa: {percent_diff:.2f}%"
        )

def main(args=None):
    rclpy.init(args=args)
    node = OdomBenchmarkNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()