#!/usr/bin/env python3
import socket
import struct
import json
import cv2
import numpy as np
import os
import torch
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict

# --- CONFIGURAÇÃO (Seus caminhos originais) ---
HOME = os.path.expanduser("~")
BASE_PATH = os.path.join(HOME, "pibic/src/mobile_manipulation_packages/GroundingDINO")
CONFIG_PATH = os.path.join(BASE_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(BASE_PATH, "weights/groundingdino_swint_ogc.pth")

print("Carregando Modelo no Python 3.10...", flush=True)
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
model = model.to("cuda")
print("Modelo Carregado! Aguardando conexões na porta 5555...", flush=True)

def get_prediction(cv_image, prompt):
    # Processamento do DINO
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
        box_threshold=0.35,
        text_threshold=0.25
    )

    # Prepara resposta JSON
    results = []
    h, w, _ = cv_image.shape
    if boxes.shape[0] > 0:
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        for i, box in enumerate(boxes_pixel):
            results.append({
                "label": phrases[i],
                "score": float(logits[i]),
                "bbox": box.tolist() # [cx, cy, w, h]
            })
    return results

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 5555)) # Porta interna
    server.listen(1)

    while True:
        conn, addr = server.accept()
        try:
            # 1. Recebe tamanho da mensagem (4 bytes)
            data_size = struct.unpack('>I', conn.recv(4))[0]
            
            # 2. Recebe a mensagem (Prompt + Imagem codificada)
            data = b""
            while len(data) < data_size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet
            
            # 3. Decodifica
            request = json.loads(data.decode('utf-8'))
            prompt = request['prompt']
            
            # A imagem vem como lista de bytes, reconstrói numpy
            img_bytes = bytes.fromhex(request['image_hex'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 4. Inferência
            detections = get_prediction(cv_image, prompt)

            # 5. Envia Resposta
            response = json.dumps(detections).encode('utf-8')
            conn.sendall(struct.pack('>I', len(response)) + response)

        except Exception as e:
            print(f"Erro no servidor: {e}")
        finally:
            conn.close()

if __name__ == '__main__':
    start_server()