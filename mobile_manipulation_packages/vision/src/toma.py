import numpy as np
import open3d as o3d
import os


base_dir = "/home/momesso/isaac-sim/toma"
output_folder = os.path.join(base_dir, "pcds") 


os.makedirs(output_folder, exist_ok=True)

print(f"Salvando arquivos em: {output_folder}")


count = 0
for i in range(5, 90):
    source_folder = f"run_{i}"
    npy_path = os.path.join(base_dir, source_folder, "object_pointcloud.npy")
    
    if os.path.exists(npy_path):
        try:
            
            points = np.load(npy_path)
            
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            
            file_name = f"toma_{i}.pcd" 
            save_path = os.path.join(output_folder, file_name)
            
           
            o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)
            print(f"Salvo: {file_name}")
            count += 1
            
        except Exception as e:
            print(f"Erro em {source_folder}: {e}")

if count == 0:
    print("\nAVISO: Nenhum arquivo foi convertido! Verifique se as pastas run_X existem em /home/momesso/")
else:
    print(f"\nSucesso! {count} arquivos salvos na pasta 'pcds'.")