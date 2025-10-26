import asyncio
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, UsdShade
import random
from omni.isaac.core.utils.semantics import add_update_semantics


def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
    xformable = UsdGeom.Xformable(prim)
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            xformable.AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            xformable.AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            xformable.AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            xformable.AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)


def get_or_create_global_physics_material(stage):
    """Cria um único PhysicsMaterial e retorna o objeto UsdShade.Material."""
    material_root = "/World/PhysicsMaterials"
    mat_path = f"{material_root}/Material_Cube"

    if not stage.GetPrimAtPath(material_root):
        stage.DefinePrim(material_root, "Scope")

    material_prim = stage.GetPrimAtPath(mat_path)
    if not material_prim.IsValid():
        material = UsdShade.Material.Define(stage, mat_path)
        material_prim = material.GetPrim()

        material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
        material_api.CreateStaticFrictionAttr(1.0)
        material_api.CreateDynamicFrictionAttr(1.0)
        material_api.CreateRestitutionAttr(0.0)

        physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
        physx_api.CreateFrictionCombineModeAttr("multiply")
        physx_api.CreateRestitutionCombineModeAttr("multiply")
        physx_api.CreateDampingCombineModeAttr("max")

        print(" PhysicsMaterial global criado em /World/PhysicsMaterials/Material_Cube")
    else:
        material = UsdShade.Material(stage.GetPrimAtPath(mat_path))
        print(" PhysicsMaterial global já existe")

    return material


def add_colliders(prim):
    if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr(True)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision_api.CreateContactOffsetAttr(0.003)
        physx_collision_api.CreateRestOffsetAttr(0.001)
        if prim.IsA(UsdGeom.Mesh):
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")


def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None, mass=0.001):
    """Adiciona corpo rígido e define massa física."""
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)

    # aplica PhysX config extra
    physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rigid_body_api.CreateDisableGravityAttr(disable_gravity)
    if angular_damping is not None:
        physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)

    # adiciona massa específica
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)


def apply_global_material(stage, prim):
    """Aplica o material físico global a um prim."""
    material = get_or_create_global_physics_material(stage)
    rel = prim.CreateRelationship("physics:material:binding", False)
    rel.SetTargets([material.GetPath()])


class ContinuousCubeSpawner:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.app = omni.kit.app.get_app()
        self.cube_counter = 0
        self.spawn_height = 1.0
        self.delete_y_threshold = -2.0
        self.parent_path = "/World/Cubes"
        self._running = False
        self._task = None

        if not self.stage.GetPrimAtPath(self.parent_path):
            self.stage.DefinePrim(self.parent_path, "Xform")

        self.timeline_event_sub = self.timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

    async def spawn_single_cube(self):
        try:
            prim_path = f"{self.parent_path}/Cube_{self.cube_counter:02d}"
            spawn_pos = Gf.Vec3f(random.uniform(0.4, 0.65),
                                 random.uniform(2.0, 2.4),
                                 self.spawn_height)
            spawn_scale = Gf.Vec3f(0.02, 0.02, 0.02)

            prim = self.stage.DefinePrim(prim_path, "Cube")
            set_transform_attributes(prim, location=spawn_pos, scale=spawn_scale)

            add_colliders(prim)
            add_rigid_body_dynamics(prim, mass=0.001)
            apply_global_material(self.stage, prim)

            label_token = f"box_{self.cube_counter+1:02d}"
            add_update_semantics(prim, label_token, "class")

            print(f" Cubo criado: {prim_path} (massa=0.001 kg, material global aplicado)")
            self.cube_counter += 1

        except Exception as e:
            print(f"[ Erro ao spawnar cubo]: {e}")

    def _delete_fallen_cubes(self):
        parent_prim = self.stage.GetPrimAtPath(self.parent_path)
        for prim in list(parent_prim.GetChildren()):
            try:
                pos = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
                if pos[1] < self.delete_y_threshold:
                    self.stage.RemovePrim(prim.GetPath())
                    print(f" Cubo removido: {prim.GetPath()}")
            except Exception:
                pass

    async def _spawn_loop(self):
        print("Loop de spawn iniciado")
        while self._running:
            try:
                if self.timeline.is_playing():
                    await self.spawn_single_cube()
                    self._delete_fallen_cubes()
                await asyncio.sleep(10.0)
            except Exception as e:
                print(f"[ Loop error]: {e}")
            await self.app.next_update_async()

    def _on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            if not self._running:
                self._running = True
                self._task = asyncio.ensure_future(self._spawn_loop())
                print("Spawner iniciado")
        elif event.type in (int(omni.timeline.TimelineEventType.STOP),
                            int(omni.timeline.TimelineEventType.PAUSE)):
            if self._running:
                self._running = False
                print("Spawner parado")


spawner_manager = ContinuousCubeSpawner()
