import omni.usd
from pxr import Gf, UsdGeom, Usd, UsdPhysics
import random
import omni.kit.commands
import numpy as np
import carb
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.nucleus as nucleus_utils

# --- CONFIGURAÇÃO ---
target_prim_path = "/world/WillowBench_02"
folder_path = "/world/Training_Objects"

num_objects = 20
min_size = 0.06
max_size = 0.08
spawn_height_offset = 0.1

CARDBOX_ASSETS = [
    "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
]

scatter_area_x = 0.3 
scatter_area_y = 0.3 
table_height_estimate = 0.2

def fix_collision_recursive(prim):
    """Percorre a hierarquia e força convexHull em todas as malhas de colisão."""
    
    if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
        
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        else:
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            
        
        mesh_api.CreateApproximationAttr().Set("convexHull")
        
    
    for child in prim.GetChildren():
        fix_collision_recursive(child)

def spawn_physics_cardboxes():
    stage = omni.usd.get_context().get_stage()
    target_prim = stage.GetPrimAtPath(target_prim_path)

    if not target_prim.IsValid():
        carb.log_error(f"Objeto alvo '{target_prim_path}' não encontrado!")
        print("ERRO: Alvo não encontrado.")
        return

    
    if stage.GetPrimAtPath(folder_path):
        omni.kit.commands.execute("DeletePrims", paths=[folder_path])
    omni.kit.commands.execute("CreatePrim", prim_path=folder_path, prim_type="Xform")

    
    xform = UsdGeom.Xformable(target_prim)
    world_transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = world_transform.ExtractTranslation()
    
    center_x = translation[0]
    center_y = translation[1]
    base_z = translation[2] + table_height_estimate

    min_x = center_x - (scatter_area_x / 2.0)
    max_x = center_x + (scatter_area_x / 2.0)
    min_y = center_y - (scatter_area_y / 2.0)
    max_y = center_y + (scatter_area_y / 2.0)

  
    print(f"Gerando {num_objects} caixas com física RECURSIVA...")
    assets_root_path = nucleus_utils.get_assets_root_path()
    if assets_root_path is None:
        print("ERRO: Nucleus não encontrado.")
        return

    for i in range(num_objects):
        usd_path = assets_root_path + random.choice(CARDBOX_ASSETS)
        prim_name = f"Cardbox_{i}"
        full_prim_path = f"{folder_path}/{prim_name}"
        
        rand_x = random.uniform(min_x, max_x)
        rand_y = random.uniform(min_y, max_y)
        rand_z = base_z + spawn_height_offset + (i * 0.02) + random.uniform(0, 0.02)
        s = random.uniform(min_size, max_size)
        
        add_reference_to_stage(usd_path=usd_path, prim_path=full_prim_path)
        
        prim = stage.GetPrimAtPath(full_prim_path)
        if prim.IsValid():
            
            xform_api = UsdGeom.XformCommonAPI(prim)
            xform_api.SetTranslate(Gf.Vec3d(rand_x, rand_y, rand_z))
            xform_api.SetScale(Gf.Vec3f(s, s, s))
            rand_rot = random.uniform(0, 360)
            xform_api.SetRotate(Gf.Vec3f(0, 0, rand_rot))

            
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(prim)
            
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(0.05)

           
            fix_collision_recursive(prim)

    print(f"Caixas criadas.")

spawn_physics_cardboxes()