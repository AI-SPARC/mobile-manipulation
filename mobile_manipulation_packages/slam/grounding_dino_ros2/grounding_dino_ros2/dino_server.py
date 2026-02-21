#!/usr/bin/env python3
import socket
import struct
import json
import cv2
import numpy as np
import os
import sys
import torch
from PIL import Image

# --- CONFIGURAÇÃO DE CAMINHOS ---
HOME = os.path.expanduser("~")
BASE_PATH = os.path.join(HOME, "pibic/src/mobile_manipulation_packages/slam/GroundingDINO")

if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

try:
    import groundingdino.datasets.transforms as T
    from groundingdino.util.inference import load_model, predict
except ImportError:
    print(f"ERRO: Não foi possível importar 'groundingdino'. Verifique o caminho: {BASE_PATH}")
    sys.exit(1)

CONFIG_PATH = os.path.join(BASE_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(BASE_PATH, "weights/groundingdino_swint_ogc.pth")

print("Carregando Modelo...", flush=True)
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
model = model.to("cuda")
print("Pronto! Aguardando na porta 5555...", flush=True)

def get_prediction(cv_image, prompt):
    image_pil = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=0.75,
        text_threshold=0.25
    )

    results = []
    # h, w, _ = cv_image.shape  <-- NÃO PRECISA MAIS DISSO AQUI
    
    if boxes.shape[0] > 0:
        # AQUI ESTAVA O ERRO: Não convertemos para pixels aqui.
        # Enviamos "cru" (0.0 a 1.0) para o ROS decidir.
        for i, box in enumerate(boxes):
            results.append({
                "label": phrases[i],
                "score": float(logits[i]),
                "bbox": box.tolist() # [cx, cy, w, h] NORMALIZADO (0-1)
            })
    return results

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', 5555))
    server.listen(1)

    while True:
        try:
            conn, addr = server.accept()
            header = conn.recv(4)
            if not header:
                conn.close()
                continue
            data_size = struct.unpack('>I', header)[0]
            
            data = b""
            while len(data) < data_size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet
            
            request = json.loads(data.decode('utf-8'))
            img_bytes = bytes.fromhex(request['image_hex'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            detections = get_prediction(cv_image, request['prompt'])

            response = json.dumps(detections).encode('utf-8')
            conn.sendall(struct.pack('>I', len(response)) + response)
            conn.close()

        except Exception as e:
            print(f"Erro: {e}")
            if 'conn' in locals(): conn.close()
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    start_server()