import asyncio
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, Sdf
import random
from omni.isaac.core.utils.semantics import add_update_semantics
from pxr import PhysxSchema, Sdf

def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            UsdGeom.Xformable(prim).AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)


def add_colliders(root_prim):
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            physx_collision_api.CreateContactOffsetAttr(0.003)
            physx_collision_api.CreateRestOffsetAttr(0.001)
        if desc_prim.IsA(UsdGeom.Mesh):
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")


def has_colliders(root_prim):
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False


def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None):
    if has_colliders(prim):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
        if angular_damping is not None:
            physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)
    else:
        print(f"Prim '{prim.GetPath()}' não tem colisores. Pulando corpo rígido.")





def add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False):
    add_colliders(prim)
    add_rigid_body_dynamics(prim, disable_gravity=disable_gravity)


class ContinuousCubeSpawner:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.app = omni.kit.app.get_app()
        self.cube_counter = 0
        self.spawn_height = 1.0
        self._running = False
        self._task = None
        self.delete_y_threshold = -2.0
        # self.delete_z_threshold = 0.2  
        self.parent_path = "/world/Cubes"

        if not self.stage.GetPrimAtPath(self.parent_path):
            self.stage.DefinePrim(self.parent_path, "Xform")

        self.timeline_event_sub = self.timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

        

    async def spawn_single_cube(self):

     

        prim_path = f"{self.parent_path}/Cube_{self.cube_counter:02d}"
        spawn_pos = Gf.Vec3f(random.uniform(0.4, 0.65),
                            random.uniform(2.0, 2.4),
                            self.spawn_height)
        spawn_scale = Gf.Vec3f(0.02, 0.02, 0.02)
        prim = self.stage.DefinePrim(prim_path, "Cube")
        set_transform_attributes(prim, location=spawn_pos, scale=spawn_scale)
        add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False)

        idx_str = f"{self.cube_counter + 1:02d}"
        label_token = f"box_{idx_str}"

       
        try:
            add_update_semantics(prim, label_token, "class")
            print(f"SemanticsAPI aplicado: {label_token}")
        except Exception as e:
            print("Erro ao aplicar SemanticsAPI:", e)

    

        self.cube_counter += 1
        print(f"Cubo criado: {prim_path} semantic={label_token}")

    def _delete_fallen_cubes(self):
        # remove cubos abaixo do limite
        for prim in list(self.stage.GetPrimAtPath(self.parent_path).GetChildren()):
            if prim.IsValid() and prim.IsA(UsdGeom.Cube):
                try:
                    pos = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
                    if pos[1] < self.delete_y_threshold:
                        self.stage.RemovePrim(prim.GetPath())
                        print(f"Cubo removido: {prim.GetPath()}")
                except Exception:
                    pass

    async def _spawn_loop(self):
        while self._running:
            await asyncio.sleep(2.0)
            if self.timeline.is_playing():
                await self.spawn_single_cube()
                self._delete_fallen_cubes()
                await self.app.next_update_async()

    def _on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            if not self._running:
                self._running = True
                self._task = asyncio.ensure_future(self._spawn_loop())
                print("Spawner iniciado.")
        elif event.type in (int(omni.timeline.TimelineEventType.STOP),
                            int(omni.timeline.TimelineEventType.PAUSE)):
            if self._running:
                self._running = False
                print("Spawner parado.")


spawner_manager = ContinuousCubeSpawner()


# import asyncio
# import omni.kit.app
# import omni.usd
# from pxr import Usd, UsdGeom, Sdf, Gf, UsdPhysics, PhysxSchema
# import random

# def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
#     if location is not None:
#         if not prim.HasAttribute("xformOp:translate"):
#             UsdGeom.Xformable(prim).AddTranslateOp()
#         prim.GetAttribute("xformOp:translate").Set(location)
#     if orientation is not None:
#         if not prim.HasAttribute("xformOp:orient"):
#             UsdGeom.Xformable(prim).AddOrientOp()
#         prim.GetAttribute("xformOp:orient").Set(orientation)
#     if rotation is not None:
#         if not prim.HasAttribute("xformOp:rotateXYZ"):
#             UsdGeom.Xformable(prim).AddRotateXYZOp()
#         prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
#     if scale is not None:
#         if not prim.HasAttribute("xformOp:scale"):
#             UsdGeom.Xformable(prim).AddScaleOp()
#         prim.GetAttribute("xformOp:scale").Set(scale)


# def add_colliders(root_prim):
#     for desc_prim in Usd.PrimRange(root_prim):
#         if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
#             collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
#             collision_api.CreateCollisionEnabledAttr(True)
#             physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
#             physx_collision_api.CreateContactOffsetAttr(0.003)
#             physx_collision_api.CreateRestOffsetAttr(0.001)
#         if desc_prim.IsA(UsdGeom.Mesh):
#             mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
#             mesh_collision_api.CreateApproximationAttr().Set("convexHull")


# def has_colliders(root_prim):
#     for desc_prim in Usd.PrimRange(root_prim):
#         if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
#             return True
#     return False


# def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None):
#     if has_colliders(prim):
#         rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
#         rigid_body_api.CreateRigidBodyEnabledAttr(True)
#         physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
#         physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
#         if angular_damping is not None:
#             physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)
#     else:
#         print(f"Prim '{prim.GetPath()}' não tem colisores. Pulando corpo rígido.")


# def add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False):
#     add_colliders(prim)
#     add_rigid_body_dynamics(prim, disable_gravity=disable_gravity)


# class ContinuousCubeSpawner:
    
#     def __init__(self):
#         self.stage = omni.usd.get_context().get_stage()
#         self.timeline = omni.timeline.get_timeline_interface()
#         self.app = omni.kit.app.get_app()
#         self.cube_counter = 0
#         self.spawn_height = 1.0
#         self.spawn_x = 0.62
#         self.spawn_y = 2.36 
#         self._running = False
#         self._task = None

#         # Conecta eventos de simulação
#         self.timeline_event_sub = self.timeline.get_timeline_event_stream().create_subscription_to_pop(
#             self._on_timeline_event
#         )

#     async def spawn_single_cube(self):
#         prim_path = f"/world/Cubes/SpawnedCube_{self.cube_counter}"
#         self.cube_counter += 1
#         spawn_pos = Gf.Vec3f(random.uniform(0.4, 0.65), random.uniform(2.0, 2.4), self.spawn_height)
#         spawn_scale = Gf.Vec3f(0.02, 0.02, 0.02)

#         prim = self.stage.DefinePrim(prim_path, "Cube")
#         set_transform_attributes(prim, location=spawn_pos, scale=spawn_scale)
#         add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False)
#         print(f"Cubo criado: {prim_path}")

#     async def _spawn_loop(self):
#         while self._running:
#             await asyncio.sleep(2.0)
#             if self.timeline.is_playing():
#                 await self.spawn_single_cube()
#                 await self.app.next_update_async()

#     def _on_timeline_event(self, event):
#         # 0 = stop, 1 = play, 2 = pause
#         if event.type == int(omni.timeline.TimelineEventType.PLAY):
#             if not self._running:
#                 self._running = True
#                 self._task = asyncio.ensure_future(self._spawn_loop())
#                 print("Spawner iniciado.")
#         elif event.type in (int(omni.timeline.TimelineEventType.STOP),
#                             int(omni.timeline.TimelineEventType.PAUSE)):
#             if self._running:
#                 self._running = False
#                 print("Spawner parado.")


# spawner_manager = ContinuousCubeSpawner()
