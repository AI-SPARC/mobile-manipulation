import omni.usd
import omni.timeline
import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, PhysxSchema
import numpy as np
from omni.physx.scripts import physicsUtils, particleUtils
import time

# ---------------------------------------------------------------------------------
# ======================== PARÂMETROS PARA CONFIGURAR ===============================
# ---------------------------------------------------------------------------------

PRIM_A_SEGUIR = "/denso_robot/link6/Cylinder"
PHYSICS_SCENE_PATH = "/World/physicsScene"
EIXO_FRONTAL_DO_GRIPPER = Gf.Vec3f(0, 0, 1)
INTERVALO_CRIACAO = 0.0001
TEMPO_DE_VIDA = 0.075
RAIO_PARTICULA = 0.009
COR_INICIAL_FAISCA = Gf.Vec3f(1.0, 0.1, 0.0)
COR_FINAL_FAISCA = Gf.Vec3f(1.0, 0.0, 0.0)
FECHAMENTO_DO_CONE = 0.905
VELOCIDADE_MIN = 1.0
VELOCIDADE_MAX = 3.0

# ---------------------------------------------------------------------------------
# ========================== LÓGICA DO SCRIPT (NÃO MEXER) ===========================
# ---------------------------------------------------------------------------------

class SparkEmitter:
    def __init__(self, stage):
        self._stage = stage
        self._is_active = True
        self._is_running = False
        self._time = 0.0
        self._next_spawn_time = 0.0
        self._active_particles = []
        self._rng = np.random.default_rng()
        self._timeline_sub = None
        self._physics_sub = None
        self._particle_system_path = Sdf.Path("/World/particleSystem")

    def setup_scene(self):
        """Prepara a cena de forma não-destrutiva."""
        if not self._stage.GetPrimAtPath("/World"):
            self._stage.SetDefaultPrim(UsdGeom.Xform.Define(self._stage, "/World").GetPrim())
        if not self._stage.GetPrimAtPath("/World/physicsScene"):
            scene = UsdPhysics.Scene.Define(self._stage, "/World/physicsScene")
            scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
        if not self._stage.GetPrimAtPath("/World/particles"):
            UsdGeom.Xform.Define(self._stage, "/World/particles")
        
        if not self._stage.GetPrimAtPath(self._particle_system_path):
            rest_offset = RAIO_PARTICULA / 0.6
            particle_system = particleUtils.add_physx_particle_system(
                self._stage, self._particle_system_path, contact_offset=rest_offset + 0.005, rest_offset=rest_offset,
                particle_contact_offset=rest_offset + 0.005, solid_rest_offset=rest_offset,
                fluid_rest_offset=RAIO_PARTICULA, solver_position_iterations=4,
                simulation_owner=PHYSICS_SCENE_PATH, max_neighborhood=96
            )
            pbd_material_path = "/World/pbdParticleMaterial"
            particleUtils.add_pbd_particle_material(self._stage, pbd_material_path, friction=0.5)
            physicsUtils.add_physics_material_to_prim(self._stage, particle_system.GetPrim(), pbd_material_path)
            
        # REMOVIDO: Toda a lógica complexa de CollisionGroups que estava causando erros.
        print("Cena de física verificada.")

    def setup_callbacks(self):
        """Inscreve os métodos nos eventos da timeline e da física."""
        self._timeline_sub = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop(self._on_timeline_event)
        self._physics_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_physics_step)
        print("Emissor de faíscas iniciado. Pressione Play na timeline.")

    def cleanup(self):
        """Função de limpeza completa para desligar o script com segurança."""
        self._is_active = False
        if self._timeline_sub: self._timeline_sub = None
        if self._physics_sub: self._physics_sub = None
        self._clear_all_particles()
        print("Emissor de faíscas anterior parado e limpo.")
        
    def _clear_all_particles(self):
        """Apaga todas as partículas ativas da cena."""
        for particle_data in self._active_particles:
            prim = particle_data["prim"]
            if prim and prim.IsValid():
                self._stage.RemovePrim(prim.GetPath())
        self._active_particles = []

    def _on_timeline_event(self, e):
        """Lida com os eventos de Play, Pause e Stop da timeline."""
        if not self._is_active: return
        
        if e.type == int(omni.timeline.TimelineEventType.PLAY):
            self._is_running = True
            self._time = 0.0
            self._next_spawn_time = 0.0
        elif e.type == int(omni.timeline.TimelineEventType.PAUSE):
            self._is_running = False
            self._clear_all_particles()
        elif e.type == int(omni.timeline.TimelineEventType.STOP):
            self._is_running = False
            self._clear_all_particles()

    def _on_physics_step(self, e):
        """Loop principal, chamado a cada frame de física se a simulação estiver rodando."""
        if not self._is_active or not self._is_running: return

        dt = e.payload.get("dt", 0.0)
        if dt == 0.0: return
        self._time += dt
        
        self._update_particle_lifetimes()

        control_prim = self._stage.GetPrimAtPath(PRIM_A_SEGUIR)
        if control_prim.IsValid():
            world_transform = UsdGeom.Imageable(control_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_offset_point = Gf.Vec3f(1.95, 0, 0) # Offset de segurança
            world_spawn_point = world_transform.Transform(local_offset_point)
            current_origin = Gf.Vec3f(world_spawn_point)

            gripper_forward_in_world = world_transform.TransformDir(EIXO_FRONTAL_DO_GRIPPER)
            gripper_forward_in_world.Normalize()
            emission_axis = -Gf.Vec3f(gripper_forward_in_world)
            
            if self._time >= self._next_spawn_time:
                self._next_spawn_time = self._time + INTERVALO_CRIACAO
                self._create_particle(current_origin, emission_axis)

    def _update_particle_lifetimes(self):
        """Verifica e apaga partículas que atingiram o tempo de vida."""
        particles_to_remove = []
        for particle_data in self._active_particles:
            if self._time - particle_data["creation_time"] > TEMPO_DE_VIDA:
                prim = particle_data["prim"]
                if prim and prim.IsValid():
                    self._stage.RemovePrim(prim.GetPath())
                particles_to_remove.append(particle_data)
        if particles_to_remove:
            self._active_particles = [p for p in self._active_particles if p not in particles_to_remove]
        
    def _create_particle(self, origin_pos, cone_axis):
        """Cria um prim de partícula único."""
        random_dir = Gf.Vec3f(self._rng.uniform(-1.0, 1.0), self._rng.uniform(-1.0, 1.0), self._rng.uniform(-1.0, 1.0))
        final_dir = (cone_axis * FECHAMENTO_DO_CONE) + (random_dir * (1.0 - FECHAMENTO_DO_CONE))
        final_dir.Normalize()
        speed = self._rng.uniform(VELOCIDADE_MIN, VELOCIDADE_MAX)
        velocity = final_dir * speed
        t = self._rng.random()
        spark_color = COR_INICIAL_FAISCA * (1 - t) + COR_FINAL_FAISCA * t
        
        unique_id = int(time.time() * 10000)
        particle_path = f"/World/particles/particle_{unique_id}"
        
        prim = particleUtils.add_physx_particleset_points(
            self._stage, Sdf.Path(particle_path), [origin_pos], [velocity], 
            [2 * RAIO_PARTICULA], self._particle_system_path, 
            self_collision=True, fluid=False, particle_group=0, 
            # CORREÇÃO DEFINITIVA: Massa extremamente pequena para não exercer força no robô.
            particle_mass=0.00001, 
            density=0.0
        )
        prim.CreateDisplayColorAttr([spark_color])
        self._active_particles.append({"prim": prim.GetPrim(), "creation_time": self._time})

# ============================== EXECUÇÃO PRINCIPAL ====================================
if "spark_emitter" in globals() and globals()["spark_emitter"] is not None:
    print("Limpando instância anterior do emissor.")
    globals()["spark_emitter"].cleanup()
    globals()["spark_emitter"] = None

stage = omni.usd.get_context().get_stage()
if stage:
    spark_emitter = SparkEmitter(stage)
    spark_emitter.setup_scene()
    spark_emitter.setup_callbacks()
    globals()["spark_emitter"] = spark_emitter
else:
    print("Erro: Nenhum stage USD válido encontrado.")