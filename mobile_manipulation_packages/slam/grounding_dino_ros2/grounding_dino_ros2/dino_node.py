#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Mensagens
from sensor_msgs.msg import Image, CameraInfo
# IMPORTANTE: Adicionei ObjectHypothesis aqui
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose, ObjectHypothesis
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3

# Bibliotecas de Processamento
from cv_bridge import CvBridge
import cv2
import socket
import json
import struct
import numpy as np

class GroundingDinoClient(Node):
    def __init__(self):
        super().__init__('grounding_dino_node')
        
        self.declare_parameter('prompt', 'fire extinguisher . chair . backpack')
        self.declare_parameter('mad_threshold', 1.5) 
        
        self.bridge = CvBridge()
        self.latest_depth = None
        self.camera_info = None
        
        # QoS Permissivo
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.sub_rgb = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, qos_sensor)
        
        self.sub_depth = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, qos_sensor)
            
        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.info_callback, qos_sensor)

        # Publishers
        self.det3d_pub = self.create_publisher(Detection3DArray, '/grounding_dino/detections_3d', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/grounding_dino/markers', 10)
        self.debug_pub = self.create_publisher(Image, '/grounding_dino/debug_image', 10)

        self.get_logger().info("Cliente DINO 3D (MAD Filter) Iniciado!")

    def depth_callback(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if cv_depth.dtype == np.uint16:
                self.latest_depth = cv_depth.astype(np.float32) / 1000.0
            else:
                self.latest_depth = cv_depth
        except Exception as e:
            self.get_logger().error(f"Erro depth: {e}")

    def info_callback(self, msg):
        self.camera_info = msg

    def get_robust_3d_position(self, bbox_pixels, img_w, img_h):
        # CHECK 1: Profundidade existe?
        if self.latest_depth is None:
            # self.get_logger().warn("[3D Fail] Sem imagem de profundidade.")
            return None, None
        
        depth_h, depth_w = self.latest_depth.shape
        
        # Escala automática (caso RGB e Depth tenham resoluções diferentes)
        scale_x = depth_w / img_w
        scale_y = depth_h / img_h
        
        cx, cy, w_box, h_box = bbox_pixels
        
        # Converte bbox para coordenadas da Depth
        cx_d = cx * scale_x
        cy_d = cy * scale_y
        w_d = w_box * scale_x
        h_d = h_box * scale_y
        
        x1 = int(max(0, cx_d - w_d/2))
        y1 = int(max(0, cy_d - h_d/2))
        x2 = int(min(depth_w, cx_d + w_d/2))
        y2 = int(min(depth_h, cy_d + h_d/2))
        
        if x2 <= x1 or y2 <= y1: return None, None

        depth_crop = self.latest_depth[y1:y2, x1:x2]
        valid_depths = depth_crop[(depth_crop > 0) & (~np.isnan(depth_crop))]

        valid_depths = valid_depths[valid_depths <= 5.0]

        if len(valid_depths) < 5: return None, None

        # MAD Filter
        median_z = np.median(valid_depths)
        mad = np.median(np.abs(valid_depths - median_z))
        sigma = max(mad, 0.02) 
        threshold = self.get_parameter('mad_threshold').value * sigma
        inliers = valid_depths[np.abs(valid_depths - median_z) < threshold]
        
        if len(inliers) == 0: return None, None

        Z = float(np.mean(inliers))

     
        
        Z = float(np.mean(inliers))
        
        if Z > 5.0:
            # self.get_logger().info(f"Objeto descartado: muito longe ({Z:.2f}m)")
            return None, None

        # Projeção Pinhole
        if self.camera_info:
            fx = self.camera_info.k[0] * (img_w / depth_w)
            fy = self.camera_info.k[4] * (img_h / depth_h)
            cx_cam = self.camera_info.k[2] * (img_w / depth_w)
            cy_cam = self.camera_info.k[5] * (img_h / depth_h)
        else:
            fx = fy = img_w / (2 * np.tan(np.deg2rad(60) / 2))
            cx_cam, cy_cam = img_w / 2, img_h / 2

        X = (cx - cx_cam) * Z / fx
        Y = (cy - cy_cam) * Z / fy
        
        width_meters = (w_box * Z) / fx
        height_meters = (h_box * Z) / fy
        depth_size = 4 * np.std(inliers) if len(inliers) > 1 else 0.1
        
        # self.get_logger().info(f"SUCESSO! Z={Z:.2f}m")
        return (X, Y, Z), (width_meters, height_meters, depth_size)

    def query_server(self, cv_image, prompt):
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(1.0) 
            client.connect(('localhost', 5555))
            
            _, img_encoded = cv2.imencode('.jpg', cv_image)
            img_hex = img_encoded.tobytes().hex()
            payload = json.dumps({"prompt": prompt, "image_hex": img_hex}).encode('utf-8')
            
            client.sendall(struct.pack('>I', len(payload)) + payload)
            
            header = client.recv(4)
            if not header: return []
            size = struct.unpack('>I', header)[0]
            
            data = b""
            while len(data) < size:
                packet = client.recv(4096)
                if not packet: break
                data += packet
            
            client.close()
            return json.loads(data.decode('utf-8'))
        except Exception:
            return []

    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = cv_image.shape
        prompt = self.get_parameter('prompt').value
        
        results = self.query_server(cv_image, prompt)
        
        # Prepara mensagens
        det3d_array = Detection3DArray()
        det3d_array.header = msg.header
        
        marker_array = MarkerArray()
        debug_img = cv_image.copy()
        
        for i, res in enumerate(results):
            label = res['label']
            bbox_norm = res['bbox'] # [cx, cy, w, h] (0-1)
            
            # Converte Normalizado -> Pixels
            cx_px = bbox_norm[0] * w
            cy_px = bbox_norm[1] * h
            w_px = bbox_norm[2] * w
            h_px = bbox_norm[3] * h
            
            # Tenta calcular 3D
            pos_3d, size_3d = self.get_robust_3d_position((cx_px, cy_px, w_px, h_px), w, h)
            
            x1 = int(cx_px - w_px/2)
            y1 = int(cy_px - h_px/2)
            x2 = int(cx_px + w_px/2)
            y2 = int(cy_px + h_px/2)
            
            if pos_3d:
                # SUCESSO 3D
                color = (0, 255, 0)
                x, y, z = pos_3d
                sw, sh, sd = size_3d
                text = f"{label} {z:.2f}m"

                # --- 1. CRIAÇÃO CORRETA DA MENSAGEM ROS (A CORREÇÃO ESTÁ AQUI) ---
                det = Detection3D()
                det.header = msg.header
                
                # A. Cria a hipótese pura
                hyp = ObjectHypothesis()
                hyp.class_id = label
                hyp.score = float(res['score'])
                
                # B. Coloca dentro do wrapper com pose
                hyp_with_pose = ObjectHypothesisWithPose()
                hyp_with_pose.hypothesis = hyp
                
                det.results.append(hyp_with_pose)
                
                # C. Preenche a BBox 3D
                det.bbox.center.position.x = x
                det.bbox.center.position.y = y
                det.bbox.center.position.z = z
                det.bbox.size.x = sw
                det.bbox.size.y = sh
                det.bbox.size.z = sd 
                
                det3d_array.detections.append(det)
                
                # --- 2. MARKERS PARA O RVIZ ---
                marker = Marker()
                marker.header = msg.header
                marker.ns = "dino_box"
                marker.id = i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z
                marker.scale.x = sw
                marker.scale.y = sh
                marker.scale.z = sd
                marker.color.a = 0.4
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.lifetime.sec = 1
                marker_array.markers.append(marker)

            else:
                # FALHA 3D
                color = (0, 0, 255)
                text = f"{label} (2D)"
            
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
        if det3d_array.detections:
            self.det3d_pub.publish(det3d_array)
            self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = GroundingDinoClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()