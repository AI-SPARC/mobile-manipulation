#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import socket
import json
import struct
import numpy as np

class GroundingDinoClient(Node):
    def __init__(self):
        super().__init__('grounding_dino_node')
        self.declare_parameter('prompt', 'fire extinguisher . door')
        self.bridge = CvBridge()
        
        # Subscriber da Câmera
        self.subscription = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        
        # Publisher das Detecções (Dados para o Robô)
        self.publisher_ = self.create_publisher(
            Detection2DArray, '/grounding_dino/detections', 10)
            
        # Publisher de Debug (Imagem Desenhada para Você Ver)
        self.debug_pub = self.create_publisher(
            Image, '/grounding_dino/debug_image', 10)
            
        self.get_logger().info("Cliente DINO (ROS 2) iniciado! Conectando ao servidor Python 3.10...")

    def query_dino_server(self, cv_image, prompt):
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(('localhost', 5555))

            # Codifica imagem para enviar via JSON/Socket
            _, img_encoded = cv2.imencode('.jpg', cv_image)
            img_hex = img_encoded.tobytes().hex()

            payload = json.dumps({"prompt": prompt, "image_hex": img_hex}).encode('utf-8')
            
            # Envia tamanho + dados
            client.sendall(struct.pack('>I', len(payload)) + payload)

            # Recebe resposta
            response_size = struct.unpack('>I', client.recv(4))[0]
            response_data = b""
            while len(response_data) < response_size:
                packet = client.recv(4096)
                if not packet: break
                response_data += packet
            
            client.close()
            return json.loads(response_data.decode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Erro de conexão com servidor DINO: {e}")
            return []

    def draw_detections(self, cv_image, results):
        """Desenha as caixas e textos na imagem"""
        annotated_img = cv_image.copy()
        
        for res in results:
            # O servidor manda [cx, cy, w, h]
            cx, cy, w, h = res['bbox']
            label = res['label']
            score = res['score']
            
            # Converter centro para cantos (top-left e bottom-right)
            x_min = int(cx - w / 2)
            y_min = int(cy - h / 2)
            x_max = int(cx + w / 2)
            y_max = int(cy + h / 2)
            
            # Cor Verde (BGR)
            color = (0, 255, 0)
            
            # 1. Desenha o Retângulo
            cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), color, 2)
            
            # 2. Desenha o Texto (Com fundo preto para ler melhor)
            text = f"{label}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(annotated_img, (x_min, y_min - 20), (x_min + text_w, y_min), color, -1)
            cv2.putText(annotated_img, text, (x_min, y_min - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
        return annotated_img

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Erro ao converter imagem: {e}")
            return

        prompt = self.get_parameter('prompt').value
        
        # 1. Pega resultados do Servidor
        results = self.query_dino_server(cv_image, prompt)

        # 2. Publica detections (Vision Msgs)
        if results:
            det_array = Detection2DArray()
            det_array.header = msg.header
            for res in results:
                det = Detection2D()
                det.header = msg.header
                cx, cy, w, h = res['bbox']
                det.bbox.center.position.x = float(cx)
                det.bbox.center.position.y = float(cy)
                det.bbox.size_x = float(w)
                det.bbox.size_y = float(h)
                
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = res['label']
                hyp.hypothesis.score = res['score']
                det.results.append(hyp)
                det_array.detections.append(det)
            
            self.publisher_.publish(det_array)

        # 3. Desenha e Publica Imagem de Debug
        # Mesmo se não tiver resultados, publicamos a imagem original para saber que a câmera funciona
        debug_image = self.draw_detections(cv_image, results)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
        debug_msg.header = msg.header # Importante manter o timestamp
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GroundingDinoClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()