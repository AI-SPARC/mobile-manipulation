import omni.physx as _physx
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf
import omni.usd
import carb
import numpy as np
import random
import os

# --- Tenta carregar Debug Draw ---
try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    import omni.isaac.core.utils.extensions as extensions
    extensions.enable_extension("omni.isaac.debug_draw")
    from omni.isaac.debug_draw import _debug_draw

# ================= CONFIGURAÇÕES CRÍTICAS =================
# COPIEI DO SEU LOG. Verifique se no Stage é /World ou /world
TARGET_PRIM_PATH = "/world/merda" 

NUM_SAMPLES = 8000
SCAN_RADIUS = 0.5  # Reduzi pois uma furadeira é pequena. Se for grande, aumente.
TARGET_JITTER = 0.02
SAVE_FILE = True
OUTPUT_FILE = "/home/momesso/lixo.pcd"
# ==========================================================

def get_physx_interface():
    return _physx.get_physx_scene_query_interface()

def force_high_precision_collider(stage, prim_path):
    """
    Remove RigidBody dinâmico e aplica colisor Triangle Mesh (Perfeito).
    Isso evita o erro 'Resetting approximation shape to convexHull'.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        carb.log_error(f"Objeto não encontrado: {prim_path}")
        return False

    carb.log_warn(f"Configurando colisor de ALTA PRECISÃO em: {prim_path}")

    # 1. Desativar RigidBody Dinâmico (Se tiver)
    # Se for RigidBody, o PhysX proíbe TriangleMesh. Vamos deletar a API de RigidBody temporariamente
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        # Ou alternativamente, setar como Kinematic:
        # rb = UsdPhysics.RigidBodyAPI(prim)
        # rb.CreateKinematicEnabledAttr(True)

    # 2. Aplicar APIs de Colisão
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    
    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        UsdPhysics.MeshCollisionAPI.Apply(prim)

    # 3. FORÇAR APROXIMAÇÃO "NONE" (Triangle Mesh)
    # Isso garante que o raio bata na malha exata, não numa caixa em volta dela
    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    mesh_api.CreateApproximationAttr().Set("none")
    
    return True

def save_pcd(points, filepath):
    if not points: return
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(points)}
DATA ascii
"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(header)
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        carb.log_info(f"Arquivo PCD salvo: {filepath}")
    except Exception as e:
        carb.log_error(f"Erro ao salvar: {e}")

def generate_spherical_scan():
    stage = omni.usd.get_context().get_stage()
    physx_query = get_physx_interface()
    draw = _debug_draw.acquire_debug_draw_interface()
    
    # 1. Preparar o Objeto (CORREÇÃO DO ERRO DO LOG)
    if not force_high_precision_collider(stage, TARGET_PRIM_PATH):
        return

    # Esperar um frame para o PhysX atualizar o colisor (hack necessário as vezes)
    # Em script síncrono é difícil, mas vamos tentar prosseguir
    
    prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
    usd_cache = UsdGeom.XformCache(0)
    transform = usd_cache.GetLocalToWorldTransform(prim)
    object_center = transform.ExtractTranslation()
    center_np = np.array([object_center[0], object_center[1], object_center[2]])

    points = []
    if draw: draw.clear_points()
    
    carb.log_info(f"Escaneando... (Raio: {SCAN_RADIUS})")

    attempts = 0
    max_attempts = NUM_SAMPLES * 10
    
    while len(points) < NUM_SAMPLES and attempts < max_attempts:
        attempts += 1
        
        # Geometria Esférica
        u = random.uniform(-1, 1)
        theta = random.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - u**2) * np.cos(theta)
        y = np.sqrt(1 - u**2) * np.sin(theta)
        z = u
        
        ray_origin_np = center_np + (np.array([x, y, z]) * SCAN_RADIUS)
        jitter = np.random.uniform(-TARGET_JITTER, TARGET_JITTER, size=3)
        ray_target_np = center_np + jitter
        
        direction = ray_target_np - ray_origin_np
        dist = np.linalg.norm(direction)
        if dist < 1e-4: continue
        direction = direction / dist
        
        # Raycast
        origin_carb = carb.Float3(ray_origin_np[0], ray_origin_np[1], ray_origin_np[2])
        dir_carb = carb.Float3(direction[0], direction[1], direction[2])
        
        hit = physx_query.raycast_closest(origin_carb, dir_carb, dist * 2.0)
        
        if hit["hit"]:
            hit_path = hit["rigidBody"] if hit["rigidBody"] else hit["collider"]
            hit_path_str = str(hit_path)
            
            # Verificação relaxada (case insensitive parcial)
            # Verifica se '_35_power_drill' está no caminho atingido
            target_name = TARGET_PRIM_PATH.split("/")[-1] # pega só o nome final
            
            if target_name in hit_path_str:
                p = hit["position"]
                points.append((p[0], p[1], p[2]))

    carb.log_info(f"Scan finalizado: {len(points)} pontos.")
    
    if len(points) == 0:
        carb.log_error("ZERO pontos! Verifique se o caminho do objeto está 100% correto (Maiúsculas/Minúsculas).")
        carb.log_error(f"Caminho alvo: {TARGET_PRIM_PATH}")
    else:
        if draw:
            draw.draw_points(points, [(0, 1, 0, 1)] * len(points), [2] * len(points))
        if SAVE_FILE:
            save_pcd(points, OUTPUT_FILE)

generate_spherical_scan()