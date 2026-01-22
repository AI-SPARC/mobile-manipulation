import numpy as np
import open3d as o3d # pip install open3d

# Carrega seu npy antigo
points = np.load("/home/momesso/object_pointcloud.npy") 

# Cria o objeto e salva como PCD
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("/home/momesso/toma5.pcd", pcd)

print("Salvo como nuvem.pcd!")