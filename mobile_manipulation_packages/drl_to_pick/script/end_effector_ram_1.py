import asyncio
import os
import numpy as np
import yaml
import random
import math
import carb.settings
import gc
import time  # Necessário para garantir aleatoriedade real a cada execução

# --- OMNIVERSE / USD IMPORTS ---
import omni.kit.app
import omni.usd
from omni.physx import get_physx_scene_query_interface 
from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema, Sdf, Vt

# --- ISAAC SIM IMPORTS ---
from omni.isaac.core import World
from omni.isaac.sensor import ContactSensor
from isaacsim.replicator.grasping.grasping_manager import GraspingManager
from omni.physx.scripts import utils as physxUtils
from omni.isaac.core.utils.xforms import get_world_pose

# ==========================================
# 0. CONFIGURAÇÃO INICIAL E ALEATORIEDADE
# ==========================================
# Força semente baseada no relógio do sistema para evitar repetição ao reiniciar o script
random.seed(time.time())
np.random.seed(int(time.time()))

def nuke_physics_logs():
    settings = carb.settings.get_settings()
    channels = [
        "/log/channels/omni.physx.plugin/level",
        "/log/channels/omni.physx/level",
        "/log/channels/physics/level",
        "/log/channels/usd/level"
    ]
    for ch in channels: settings.set_string(ch, "fatal")
    settings.set_bool("/app/showNotification", False)

# --- CAMINHOS ---
STAGE_PATH = "/home/momesso/pibic/src/isaacsim_moveit/maps/test_gripper.usd"
ORIGINAL_CONFIG_PATH = "/home/momesso/config.yaml"
OUTPUT_DIR_ROOT = os.path.join(os.getcwd(), "toma_final") 

# --- CONTROLE GERAL ---
TOTAL_ITERATIONS = 500 
GLOBAL_MAX_SIZE = 0.06 
THICKNESS_MIN = 0.01
THICKNESS_MAX = 0.035

# --- PARÂMETROS DE SIMULAÇÃO ---
RENDER_SKIP = 500 
DENSITY_SAMPLES = 25000 
FORCE_THRESHOLD = 0.01  

# ==========================================
# 1. SISTEMA DE NÃO-REPETIÇÃO (BARALHO)
# ==========================================
ALL_SHAPES_LIST = [
    # --- PRIMITIVOS BÁSICOS ---
    "SIMPLE_CUBE", "FLAT_PLATE", "CYLINDER", "TRIANGLE_WEDGE", 
    "HEX_ROD", "OCTAGON_ROD", "PENTAGON_ROD", "DIAMOND_PRISM", 
    "PYRAMID_TRUNCATED", "CONE_TRUNCATED",

    # --- PERFIS ESTRUTURAIS ---
    "L_SHAPE", "T_SHAPE", "U_BRACKET", "I_BEAM", "C_CHANNEL", 
    "Z_SHAPE", "E_SHAPE", "H_BEAM", "CROSS_PROFILE", "ANGLE_BRACKET_V",
    "STRUT_CHANNEL", "JOIST_HANGER", "CORNER_BRACE_3WAY", "RAIL_PROFILE", 
    "SLOTTED_RAIL",

    # --- TUBULAÇÃO E CONEXÕES ---
    "T_PIPE_JOINT", "ELBOW_PIPE", "CROSS_PIPE_4WAY", "FLANGE_ADAPTER", 
    "COUPLING_SLEEVE", "BUSHING_FLANGED", "MANIFOLD_3PORT", "REDUCER_COUPLING",
    "END_CAP_HEX", "Y_PIPE_DIVIDER", "U_RETURN_BEND", "NIPPLE_PIPE",

    # --- FIXADORES ---
    "HEX_NUT", "SQUARE_NUT", "WING_NUT", "CASTLE_NUT", "ACORN_NUT",
    "THICK_WASHER", "SPLIT_WASHER", 
    "BOLT_HEX_SHORT", "BOLT_HEX_LONG", "BOLT_ALLEN", "BOLT_CARRIAGE",
    "EYE_BOLT", "U_BOLT_PLATE", "SET_SCREW", "THUMB_SCREW",
    "RIVET_MOCK", "COTTER_PIN_MOCK", "DOWEL_PIN", "CLEVIS_PIN", "SHAFT_COLLAR",

    # --- TRANSMISSÃO MECÂNICA ---
    "SPUR_GEAR_6T", "SPUR_GEAR_8T", "SPUR_GEAR_12T", "SPUR_GEAR_24T",
    "RACK_GEAR", "WORM_GEAR_MOCK", "SPROCKET_THIN",
    "PULLEY_V_BELT", "PULLEY_TIMING", "CAM_LOBE", "CRANKSHAFT_SEGMENT",
    "UNIVERSAL_JOINT_CENTER", "CONNECTING_ROD", "PISTON_HEAD", "BEARING_BALL_MOCK",

    # --- SUPORTES E PEÇAS USINADAS ---
    "CLEVIS_MOUNT", "PILLOW_BLOCK", "HINGE_ASSEMBLY", "BEARING_HOUSING",
    "MOTOR_MOUNT_PLATE", "STEP_BLOCK", "V_BLOCK", "PARALLEL_BAR",
    "GUSSET_PLATE", "TRIANGLE_BRACKET", "SHELF_BRACKET", "DIN_RAIL_SEGMENT",
    "HEAT_SINK_FINS", "HEAT_SINK_RADIAL", "FAN_BLADE_MOCK", "IMPELLER_MOCK",
    "VALVE_HANDWHEEL", "STAR_KNOB",

    # --- FERRAMENTAS E HARDWARE ---
    "HAMMER_HEAD", "WRENCH_OPEN_END", "WRENCH_BOX_END", "HEX_KEY_L", 
    "SOCKET_MOCK", "DRIVER_BIT_PHILLIPS", "DRIVER_BIT_FLAT",
    "PADLOCK_BODY", "CARABINER_D", "CHAIN_LINK",

    # --- CARACTERES INDUSTRIAIS ---
    "CHAR_A", "CHAR_B", "CHAR_C", "CHAR_D", "CHAR_E", "CHAR_F", "CHAR_G", "CHAR_H",
    "CHAR_I", "CHAR_J", "CHAR_K", "CHAR_L", "CHAR_M", "CHAR_N", "CHAR_O", "CHAR_P",
    "CHAR_Q", "CHAR_R", "CHAR_S", "CHAR_T", "CHAR_U", "CHAR_V", "CHAR_W", "CHAR_X",
    "CHAR_Y", "CHAR_Z",
    "DIGIT_0", "DIGIT_1", "DIGIT_2", "DIGIT_3", "DIGIT_4", 
    "DIGIT_5", "DIGIT_6", "DIGIT_7", "DIGIT_8", "DIGIT_9",

    # --- FORMAS ABSTRATAS ---
    "REINFORCED_CROSS", "DUMBBELL", "PLUS_SIGN", "SQUARE_FRAME", "CORNER_XYZ",
    "OFFSET_T", "THREE_STEP_STAIRS", "SPOOL_SHAPE", "CROSS_3D_SOLID", "DOUBLE_U_BACK"
]

SHAPE_BUFFER = []

def get_next_unique_shape():
    global SHAPE_BUFFER
    if len(SHAPE_BUFFER) == 0:
        print(f">>> [SISTEMA] Embaralhando deck ({len(ALL_SHAPES_LIST)} objetos)...")
        SHAPE_BUFFER = list(ALL_SHAPES_LIST)
        random.shuffle(SHAPE_BUFFER)
    return SHAPE_BUFFER.pop()

# ==========================================
# 2. GERADORES DE GEOMETRIA (ALTA PRECISÃO)
# ==========================================
def apply_mesh_physx(prim, color=None):
    physxUtils.setCollider(prim, approximationShape="convexDecomposition")
    if not prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    
    decomp_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
    decomp_api.CreateErrorPercentageAttr(0.02) 
    decomp_api.CreateMaxConvexHullsAttr(64)
    decomp_api.CreateHullVertexLimitAttr(64)
    decomp_api.CreateShrinkWrapAttr(False)

    if color:
        mesh = UsdGeom.Mesh(prim)
        mesh.CreateDisplayColorAttr([color])

def create_box_mesh(stage, parent_path, name, size, offset, rot_deg=(0,0,0), color=None):
    child_path = f"{parent_path}/{name}"
    mesh = UsdGeom.Mesh.Define(stage, child_path)
    sx, sy, sz = size[0]/2, size[1]/2, size[2]/2
    points = [
        Gf.Vec3f(-sx, -sy, -sz), Gf.Vec3f(sx, -sy, -sz), 
        Gf.Vec3f(sx, sy, -sz), Gf.Vec3f(-sx, sy, -sz),
        Gf.Vec3f(-sx, -sy, sz), Gf.Vec3f(sx, -sy, sz), 
        Gf.Vec3f(sx, sy, sz), Gf.Vec3f(-sx, sy, sz),
    ]
    counts = [4] * 6
    indices = [
        0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 
        1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7
    ]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    if rot_deg != (0,0,0): mesh.AddRotateXYZOp().Set(Gf.Vec3f(*rot_deg))
    mesh.AddTranslateOp().Set(Gf.Vec3f(*offset))
    apply_mesh_physx(mesh.GetPrim(), color)

def create_prism_mesh(stage, parent_path, name, radius, height, sides, offset, rot_deg=(0,0,0), color=None):
    child_path = f"{parent_path}/{name}"
    mesh = UsdGeom.Mesh.Define(stage, child_path)
    points = []
    h_half = height / 2.0
    for i in range(sides):
        angle = 2.0 * math.pi * i / sides
        points.append(Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), -h_half))
    for i in range(sides):
        angle = 2.0 * math.pi * i / sides
        points.append(Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), h_half))
    counts = []
    indices = []
    counts.append(sides)
    indices.extend(reversed(range(0, sides)))
    counts.append(sides)
    indices.extend(range(sides, 2 * sides))
    for i in range(sides):
        next_i = (i + 1) % sides
        counts.append(4)
        indices.extend([i, next_i, next_i + sides, i + sides])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    if rot_deg != (0,0,0): mesh.AddRotateXYZOp().Set(Gf.Vec3f(*rot_deg))
    mesh.AddTranslateOp().Set(Gf.Vec3f(*offset))
    apply_mesh_physx(mesh.GetPrim(), color)

# ==========================================
# 3. CONSTRUÇÃO DO OBJETO
# ==========================================
def setup_physics_scene(stage):
    scene_path = "/World/PhysicsScene"
    if not stage.GetPrimAtPath(scene_path):
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -9.81))

def generate_industrial_object(stage):
    obj_path = "/World/IndustrialPart"
    setup_physics_scene(stage)
    if stage.GetPrimAtPath(obj_path):
        stage.RemovePrim(obj_path)
    
    xform = UsdGeom.Xform.Define(stage, obj_path)
    xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass_api.CreateMassAttr(0.2)
    
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim())
    physx.CreateDisableGravityAttr(True)
    physx.CreateSleepThresholdAttr(0.0)

    # ----------------------------------------------------
    # AJUSTE SOLICITADO: Scale Factor entre 0.5 e 1.0
    # ----------------------------------------------------
    scale_factor = random.uniform(0.5, 1.0) 
    
    L = GLOBAL_MAX_SIZE * scale_factor 
    raw_t = random.uniform(THICKNESS_MIN, THICKNESS_MAX)
    t = min(raw_t, L * 0.4) 

    shape_type = get_next_unique_shape()
    print(f"  -> Gerando: {shape_type} | Box: {L*100:.1f}cm | t: {t*1000:.1f}mm")

    # ----------------------------------------------------
    # GERAÇÃO DE COR ALEATÓRIA (DINÂMICA)
    # ----------------------------------------------------
    r = random.uniform(0.1, 0.9)
    g = random.uniform(0.1, 0.9)
    b = random.uniform(0.1, 0.9)
    # Mistura com cinza para manter aspecto industrial
    mix = 0.5
    r = r * (1 - mix) + 0.5 * mix
    g = g * (1 - mix) + 0.5 * mix
    b = b * (1 - mix) + 0.5 * mix
    base_color = Gf.Vec3f(r, g, b)

    # --- LÓGICA DE FORMAS ---
    if shape_type == "SIMPLE_CUBE":
        create_box_mesh(stage, obj_path, "b", (L, L, L), (0,0,0), (0,0,0), base_color)
    elif shape_type == "FLAT_PLATE":
        create_box_mesh(stage, obj_path, "p", (L, L, t), (0,0,0), (0,0,0), base_color)
    elif shape_type == "CYLINDER":
        create_prism_mesh(stage, obj_path, "c", L/2, L, 64, (0,0,0), (90,0,0), base_color)
    elif shape_type == "TRIANGLE_WEDGE":
        create_prism_mesh(stage, obj_path, "w", L/2, L, 3, (0,0,0), (90, 0, 0), base_color)
    elif shape_type == "HEX_ROD":
        create_prism_mesh(stage, obj_path, "h", L/2, L, 6, (0,0,0), (0, 90, 0), base_color)
    elif shape_type == "OCTAGON_ROD":
        create_prism_mesh(stage, obj_path, "o", L/2, L, 8, (0,0,0), (90,0,0), base_color)
    elif shape_type == "PENTAGON_ROD":
        create_prism_mesh(stage, obj_path, "p", L/2, L, 5, (0,0,0), (90,0,0), base_color)
    elif shape_type == "DIAMOND_PRISM":
        create_prism_mesh(stage, obj_path, "d", L/2, L, 4, (0,0,0), (90,0,45), base_color)
    elif shape_type == "PYRAMID_TRUNCATED":
        create_box_mesh(stage, obj_path, "b1", (L, L, L/2), (0,0,-L/4), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "b2", (L*0.6, L*0.6, L/2), (0,0,L/4), (0,0,0), base_color)
    elif shape_type == "CONE_TRUNCATED":
        create_prism_mesh(stage, obj_path, "base", L/2, L*0.5, 64, (0,0, -L*0.25), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "top", L*0.25, L*0.5, 64, (0,0, L*0.25), (90,0,0), base_color)
    elif shape_type == "L_SHAPE":
        offset_pos = (L - t) / 2
        create_box_mesh(stage, obj_path, "l1", (t, L*0.4, L), (-offset_pos, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "l2", (L, L*0.4, t), (0, 0, -offset_pos), (0,0,0), base_color)
    elif shape_type == "T_SHAPE":
        create_box_mesh(stage, obj_path, "s", (t, t, L-t), (0,0, -t/2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "t", (L, t, t), (0,0, (L-t)/2), (0,0,0), base_color)
    elif shape_type == "U_BRACKET":
        create_box_mesh(stage, obj_path, "b", (L, L*0.4, t), (0,0, -(L-t)/2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "w1", (t, L*0.4, L), (-(L-t)/2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "w2", (t, L*0.4, L), ((L-t)/2, 0, 0), (0,0,0), base_color)
    elif shape_type == "I_BEAM":
        offset = (L - t) / 2
        create_box_mesh(stage, obj_path, "w", (t, L, L), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "t", (L, L, t), (0,0, offset), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "b", (L, L, t), (0,0, -offset), (0,0,0), base_color)
    elif shape_type == "H_BEAM":
        offset = (L - t) / 2
        create_box_mesh(stage, obj_path, "w", (L, L, t), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "l", (t, L, L), (-offset, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "r", (t, L, L), (offset, 0, 0), (0,0,0), base_color)
    elif shape_type == "C_CHANNEL":
        offset = (L - t) / 2
        create_box_mesh(stage, obj_path, "w", (t, L, L), (-offset, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "f1", (L, L, t), (0, 0, offset), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "f2", (L, L, t), (0, 0, -offset), (0,0,0), base_color)
    elif shape_type == "Z_SHAPE":
        offset = (L - t) / 2
        create_box_mesh(stage, obj_path, "t", (L, t, L*0.4), (0, offset, L/4), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "c", (t, L, L*0.4), (0, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "b", (L, t, L*0.4), (0, -offset, -L/4), (0,0,0), base_color)
    elif shape_type == "E_SHAPE":
        offset = (L - t) / 2
        create_box_mesh(stage, obj_path, "bk", (t, L, L*0.4), (-L*0.2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "t", (L*0.6, t, L*0.4), (0, offset, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "m", (L*0.6, t, L*0.4), (0, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "b", (L*0.6, t, L*0.4), (0, -offset, 0), (0,0,0), base_color)
    elif shape_type == "CROSS_PROFILE":
        create_box_mesh(stage, obj_path, "v", (t, L, t), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "h", (L, t, t), (0,0,0), (0,0,0), base_color)
    elif shape_type == "ANGLE_BRACKET_V":
        leg_l = L*0.7
        create_box_mesh(stage, obj_path, "l1", (leg_l, t, L*0.4), (leg_l/3, 0, 0), (0,0,45), base_color)
        create_box_mesh(stage, obj_path, "l2", (leg_l, t, L*0.4), (-leg_l/3, 0, 0), (0,0,-45), base_color)
    elif shape_type == "STRUT_CHANNEL":
        create_box_mesh(stage, obj_path, "b", (L, L, t), (0,0, -L/2+t/2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s1", (t, L, L), (-L/2+t/2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s2", (t, L, L), (L/2-t/2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "l1", (L*0.2, L, t), (-L/2+L*0.1, 0, L/2-t/2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "l2", (L*0.2, L, t), (L/2-L*0.1, 0, L/2-t/2), (0,0,0), base_color)
    elif shape_type == "JOIST_HANGER":
        create_box_mesh(stage, obj_path, "back", (L, t, L*0.8), (0, -L/2+t/2, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "bott", (L, L*0.6, t), (0, -L/2+L*0.3, -L*0.4), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s1", (t, L*0.6, L*0.8), (-L/2+t/2, -L/2+L*0.3, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s2", (t, L*0.6, L*0.8), (L/2-t/2, -L/2+L*0.3, 0), (0,0,0), base_color)
    elif shape_type == "CORNER_BRACE_3WAY":
        create_box_mesh(stage, obj_path, "x", (L*0.5, t, t), (L*0.25, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "y", (t, L*0.5, t), (0, L*0.25, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "z", (t, t, L*0.5), (0, 0, L*0.25), (0,0,0), base_color)
    elif shape_type == "T_PIPE_JOINT":
        r = L * 0.15
        create_prism_mesh(stage, obj_path, "m", r, L, 32, (0,0,0), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "b", r, L*0.5, 32, (0,0, L*0.25), (0,0,0), base_color)
    elif shape_type == "ELBOW_PIPE":
        l_len = L * 0.7 
        r = L * 0.15
        create_prism_mesh(stage, obj_path, "l1", r, l_len, 32, (l_len/2 - r, 0, 0), (0, 90, 0), base_color)
        create_prism_mesh(stage, obj_path, "l2", r, l_len, 32, (0, 0, l_len/2 - r), (0, 0, 0), base_color)
        create_box_mesh(stage, obj_path, "j", (r*2, r*2, r*2), (0,0,0), (0,0,0), base_color)
    elif shape_type == "CROSS_PIPE_4WAY":
        r = L * 0.15
        create_prism_mesh(stage, obj_path, "p1", r, L, 32, (0,0,0), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "p2", r, L, 32, (0,0,0), (0,90,0), base_color)
    elif shape_type == "FLANGE_ADAPTER":
        tr, fr = L * 0.15, L * 0.3
        tube_len = L - t
        create_prism_mesh(stage, obj_path, "t", tr, tube_len, 32, (0,0, t/2), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "f", fr, t, 32, (0,0, -tube_len/2), (0,0,0), base_color)
    elif shape_type == "COUPLING_SLEEVE":
        create_prism_mesh(stage, obj_path, "o", L*0.25, L*0.8, 32, (0,0,0), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "r", L*0.3, L*0.2, 32, (0,0,0), (90,0,0), base_color)
    elif shape_type == "BUSHING_FLANGED":
        create_prism_mesh(stage, obj_path, "b", L*0.15, L*0.8, 32, (0,0, t/2), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "h", L*0.25, t, 32, (0,0, -L*0.4 + t/2), (0,0,0), base_color)
    elif shape_type == "MANIFOLD_3PORT":
        r = L*0.12
        create_prism_mesh(stage, obj_path, "main", r, L, 32, (0,0,0), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "p1", r*0.8, L*0.3, 32, (-L*0.3, 0, L*0.2), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "p2", r*0.8, L*0.3, 32, (0, 0, L*0.2), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "p3", r*0.8, L*0.3, 32, (L*0.3, 0, L*0.2), (0,0,0), base_color)
    elif shape_type == "REDUCER_COUPLING":
        create_prism_mesh(stage, obj_path, "big", L*0.3, L*0.5, 32, (0,0, -L*0.25), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "sml", L*0.15, L*0.5, 32, (0,0, L*0.25), (0,0,0), base_color)
    elif shape_type == "END_CAP_HEX":
        create_prism_mesh(stage, obj_path, "hex", L*0.3, t*2, 6, (0,0, -L*0.2), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "cap", L*0.25, L*0.4, 32, (0,0, t), (0,0,0), base_color)
    elif shape_type == "Y_PIPE_DIVIDER":
        create_prism_mesh(stage, obj_path, "m", L*0.15, L*0.4, 32, (0,0, -L*0.1), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "b1", L*0.12, L*0.3, 32, (L*0.1, 0, L*0.2), (0,30,0), base_color)
        create_prism_mesh(stage, obj_path, "b2", L*0.12, L*0.3, 32, (-L*0.1, 0, L*0.2), (0,-30,0), base_color)
    elif shape_type == "U_RETURN_BEND":
        create_prism_mesh(stage, obj_path, "l1", L*0.15, L*0.7, 32, (L*0.2, 0, 0), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "l2", L*0.15, L*0.7, 32, (-L*0.2, 0, 0), (90,0,0), base_color)
        create_box_mesh(stage, obj_path, "join", (L*0.7, L*0.3, L*0.3), (0, 0, L*0.35), (0,0,0), base_color)
    elif shape_type == "NIPPLE_PIPE":
        create_prism_mesh(stage, obj_path, "body", L*0.15, L*0.6, 32, (0,0,0), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "t1", L*0.16, L*0.2, 32, (0,0, L*0.3), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "t2", L*0.16, L*0.2, 32, (0,0, -L*0.3), (90,0,0), base_color)
    elif shape_type in ["HEX_NUT", "SQUARE_NUT"]:
        sides = 6 if shape_type == "HEX_NUT" else 4
        create_prism_mesh(stage, obj_path, "n", L*0.4, L*0.3, sides, (0,0,0), (0,0,0), base_color)
    elif shape_type == "THICK_WASHER":
        create_prism_mesh(stage, obj_path, "w", L*0.4, t, 32, (0,0,0), (0,0,0), base_color)
    elif shape_type == "SPLIT_WASHER":
        create_prism_mesh(stage, obj_path, "w", L*0.4, t, 32, (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "cut", (L, t*2, t*2), (0, L*0.4, 0), (0,0,0), base_color) 
    elif shape_type == "BOLT_HEX_SHORT":
        create_prism_mesh(stage, obj_path, "h", L*0.3, L*0.2, 6, (0,0, L*0.2), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "s", L*0.15, L*0.4, 32, (0,0, -L*0.1), (0,0,0), base_color)
    elif shape_type == "BOLT_HEX_LONG":
        create_prism_mesh(stage, obj_path, "h", L*0.3, L*0.15, 6, (0,0, L*0.4), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "s", L*0.15, L*0.8, 32, (0,0, -L*0.07), (0,0,0), base_color)
    elif shape_type == "BOLT_ALLEN":
        create_prism_mesh(stage, obj_path, "h", L*0.3, L*0.2, 32, (0,0, L*0.3), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "s", L*0.15, L*0.6, 32, (0,0, -L*0.1), (0,0,0), base_color)
    elif shape_type == "SHAFT_COLLAR":
        create_prism_mesh(stage, obj_path, "c", L*0.3, L*0.15, 32, (0,0,0), (90,0,0), base_color)
        create_box_mesh(stage, obj_path, "s", (t, t, L*0.35), (0, L*0.25, 0), (0,0,0), base_color)
    elif shape_type == "WING_NUT":
        create_prism_mesh(stage, obj_path, "c", L*0.15, L*0.2, 32, (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "w1", (L*0.4, t, L*0.3), (L*0.25, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "w2", (L*0.4, t, L*0.3), (-L*0.25, 0, 0), (0,0,0), base_color)
    elif shape_type == "EYE_BOLT":
        create_prism_mesh(stage, obj_path, "s", L*0.1, L*0.6, 32, (0,0, -L*0.2), (0,0,0), base_color)
        r = L*0.2
        create_prism_mesh(stage, obj_path, "t1", L*0.08, r*2, 16, (0, r, L*0.2), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "t2", L*0.08, r*2, 16, (0, -r, L*0.2), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "t3", L*0.08, r*2, 16, (r, 0, L*0.2), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "t4", L*0.08, r*2, 16, (-r, 0, L*0.2), (0,0,0), base_color)
    elif shape_type == "COTTER_PIN_MOCK":
        create_prism_mesh(stage, obj_path, "head", L*0.15, t, 32, (0,0, L*0.4), (90,0,0), base_color)
        create_box_mesh(stage, obj_path, "l1", (t, t, L*0.8), (t/2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "l2", (t, t, L*0.6), (-t/2, 0, -L*0.1), (0,0,0), base_color)
    elif shape_type == "BOLT_CARRIAGE":
        create_prism_mesh(stage, obj_path, "h", L*0.4, L*0.1, 32, (0,0, L*0.45), (0,0,0), base_color) 
        create_box_mesh(stage, obj_path, "sq", (L*0.2, L*0.2, L*0.1), (0,0, L*0.35), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "s", L*0.1, L*0.7, 32, (0,0, -L*0.05), (0,0,0), base_color)
    elif shape_type == "CASTLE_NUT":
        create_prism_mesh(stage, obj_path, "base", L*0.4, L*0.2, 6, (0,0, -L*0.1), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "top", L*0.3, L*0.15, 32, (0,0, L*0.1), (0,0,0), base_color)
    elif shape_type == "ACORN_NUT":
        create_prism_mesh(stage, obj_path, "base", L*0.4, L*0.25, 6, (0,0, -L*0.1), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "dome", L*0.35, L*0.2, 32, (0,0, L*0.15), (0,0,0), base_color)
    elif shape_type == "U_BOLT_PLATE":
        create_box_mesh(stage, obj_path, "plate", (L*0.6, L*0.2, t), (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "l1", t*2, L*0.5, 16, (L*0.2, 0, L*0.25), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "l2", t*2, L*0.5, 16, (-L*0.2, 0, L*0.25), (0,0,0), base_color)
    elif shape_type == "SET_SCREW":
        create_prism_mesh(stage, obj_path, "body", L*0.15, L*0.3, 32, (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "hex", L*0.08, L*0.32, 6, (0,0,0), (0,0,0), base_color) 
    elif shape_type == "THUMB_SCREW":
        create_prism_mesh(stage, obj_path, "head", L*0.4, t*2, 32, (0,0, L*0.4), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "s", L*0.1, L*0.8, 32, (0,0, 0), (0,0,0), base_color)
    elif shape_type == "RIVET_MOCK":
        create_prism_mesh(stage, obj_path, "head", L*0.25, t, 32, (0,0, L*0.3), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "body", L*0.1, L*0.6, 32, (0,0, 0), (0,0,0), base_color)
    elif shape_type == "CLEVIS_PIN":
        create_prism_mesh(stage, obj_path, "head", L*0.25, t, 32, (0,0, L*0.45), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "body", L*0.12, L*0.9, 32, (0,0, 0), (0,0,0), base_color)
    elif "SPUR_GEAR" in shape_type:
        teeth_map = {"6T":6, "8T":8, "12T":12, "24T":12} 
        n = teeth_map.get(shape_type.split("_")[-1], 8)
        create_prism_mesh(stage, obj_path, "c", L*0.4, t*2, 32, (0,0,0), (0,0,0), base_color)
        for i in range(int(n/2)):
            create_box_mesh(stage, obj_path, f"t{i}", (L, L*0.15, t*2), (0,0,0), (0,0, i*(360/n)), base_color)
    elif shape_type == "RACK_GEAR":
        create_box_mesh(stage, obj_path, "b", (L, L*0.3, t), (0,0,0), (0,0,0), base_color)
        for i in range(5):
            ox = -L/2 + i*(L/5) + L/10
            create_box_mesh(stage, obj_path, f"te{i}", (L/10, L*0.1, t), (ox, L*0.2, 0), (0,0,0), base_color)
    elif shape_type == "CAM_LOBE":
        create_prism_mesh(stage, obj_path, "b", L*0.25, t, 32, (0, -L*0.1, 0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "tp", L*0.15, t, 32, (0, L*0.2, 0), (0,0,0), base_color)
    elif shape_type == "VALVE_HANDWHEEL":
        rw = L * 0.45 
        create_prism_mesh(stage, obj_path, "c1", t, rw*2, 16, (rw, 0, 0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "c2", t, rw*2, 16, (-rw, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "sp", (rw*2, t, t), (0,0,0), (0,0,0), base_color)
    elif shape_type == "CLEVIS_MOUNT":
        bs = L*0.6
        create_box_mesh(stage, obj_path, "b", (bs, bs, t), (0,0,-L*0.2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "e1", (t, bs, L*0.4), (bs/2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "e2", (t, bs, L*0.4), (-bs/2, 0, 0), (0,0,0), base_color)
    elif shape_type == "PILLOW_BLOCK":
        create_box_mesh(stage, obj_path, "b", (L, L*0.4, t), (0,0,-L*0.15), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "h", L*0.25, L*0.4, 32, (0,0, L*0.1), (90,0,0), base_color)
    elif shape_type == "HINGE_ASSEMBLY":
        plate_w = (L/2) - t
        create_box_mesh(stage, obj_path, "p1", (plate_w, t, L*0.5), (t + plate_w/2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "p2", (plate_w, t, L*0.5), (-t - plate_w/2, 0, 0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "pn", t*1.5, L*0.5, 16, (0,0,0), (0,0,0), base_color)
    elif shape_type == "DOWEL_PIN":
        create_prism_mesh(stage, obj_path, "p", L*0.05, L, 16, (0,0,0), (90,0,0), base_color)
    elif shape_type == "SPROCKET_THIN":
        create_prism_mesh(stage, obj_path, "c", L*0.45, t, 32, (0,0,0), (0,0,0), base_color)
        for i in range(4):
            create_box_mesh(stage, obj_path, f"t{i}", (L, L*0.1, t), (0,0,0), (0,0, i*45), base_color)
    elif shape_type == "CONNECTING_ROD":
        create_prism_mesh(stage, obj_path, "end1", L*0.25, t*2, 32, (L*0.35, 0, 0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "end2", L*0.15, t*2, 32, (-L*0.35, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "bar", (L*0.7, L*0.1, t), (0,0,0), (0,0,0), base_color)
    elif shape_type == "PISTON_HEAD":
        create_prism_mesh(stage, obj_path, "head", L*0.3, L*0.4, 32, (0,0, L*0.2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "pin", (L*0.2, L*0.2, L*0.2), (0,0, -L*0.1), (0,0,0), base_color)
    elif shape_type == "PULLEY_V_BELT":
        create_prism_mesh(stage, obj_path, "s1", L*0.4, t, 32, (0,0, t), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "s2", L*0.4, t, 32, (0,0, -t), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "in", L*0.3, t, 32, (0,0, 0), (0,0,0), base_color)
    elif shape_type == "PULLEY_TIMING":
        create_prism_mesh(stage, obj_path, "main", L*0.4, L*0.3, 32, (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "f1", L*0.45, t, 32, (0,0, L*0.15), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "f2", L*0.45, t, 32, (0,0, -L*0.15), (0,0,0), base_color)
    elif shape_type == "UNIVERSAL_JOINT_CENTER":
        create_box_mesh(stage, obj_path, "c", (L*0.3, L*0.3, L*0.3), (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "p1", L*0.1, L, 16, (0,0,0), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "p2", L*0.1, L, 16, (0,0,0), (0,90,0), base_color)
    elif shape_type == "CRANKSHAFT_SEGMENT":
        create_prism_mesh(stage, obj_path, "m", L*0.15, L*0.2, 32, (0,0, -L*0.3), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "o", L*0.15, L*0.2, 32, (L*0.2, 0, L*0.3), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "web", (L*0.4, L*0.1, L*0.6), (L*0.1, 0, 0), (0,0,0), base_color)
    elif shape_type == "BEARING_BALL_MOCK":
        create_prism_mesh(stage, obj_path, "out", L*0.4, L*0.2, 32, (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "in", L*0.2, L*0.2, 32, (0,0,0), (0,0,0), base_color)
    elif shape_type == "WORM_GEAR_MOCK":
        create_prism_mesh(stage, obj_path, "s", L*0.15, L, 16, (0,0,0), (0,0,0), base_color)
        for i in range(5):
            z = -L*0.3 + i*L*0.15
            create_prism_mesh(stage, obj_path, f"th{i}", L*0.2, t, 32, (0,0, z), (15,0,0), base_color)
    elif shape_type == "HEAT_SINK_FINS":
        create_box_mesh(stage, obj_path, "base", (L, L, t), (0,0, -L*0.3), (0,0,0), base_color)
        for i in range(5):
            off = -L/2 + L*0.1 + i*(L*0.2)
            create_box_mesh(stage, obj_path, f"f{i}", (t, L, L*0.5), (off, 0, 0), (0,0,0), base_color)
    elif shape_type == "HEAT_SINK_RADIAL":
        create_prism_mesh(stage, obj_path, "c", L*0.15, L*0.5, 16, (0,0,0), (0,0,0), base_color)
        for i in range(4):
            create_box_mesh(stage, obj_path, f"f{i}", (L, t, L*0.5), (0,0,0), (0,0, i*45), base_color)
    elif shape_type == "FAN_BLADE_MOCK":
        create_prism_mesh(stage, obj_path, "hub", L*0.15, t*2, 16, (0,0,0), (0,0,0), base_color)
        for i in range(4):
            create_box_mesh(stage, obj_path, f"b{i}", (L*0.4, L*0.15, t), (L*0.25, 0, 0), (15,0, i*90), base_color)
    elif shape_type == "IMPELLER_MOCK":
        create_prism_mesh(stage, obj_path, "hub", L*0.15, t*4, 16, (0,0,0), (0,0,0), base_color)
        for i in range(6):
            create_box_mesh(stage, obj_path, f"b{i}", (L*0.3, t, L*0.2), (L*0.2, 0, 0), (0,0, i*60), base_color)
    elif shape_type == "STAR_KNOB":
        create_prism_mesh(stage, obj_path, "c", L*0.2, L*0.2, 32, (0,0,0), (0,0,0), base_color)
        al = L*0.4
        for ang, n in zip([0,45,90,135], ["a1","a2","a3","a4"]):
            create_box_mesh(stage, obj_path, n, (al*2, t, t), (0,0,0), (0,0,ang), base_color)
    elif shape_type == "REINFORCED_CROSS":
        create_box_mesh(stage, obj_path, "x", (L, t, t), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "y", (t, L, t), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "z", (t, t, L), (0,0,0), (0,0,0), base_color)
        cs = t*1.5
        create_box_mesh(stage, obj_path, "c1", (cs, cs, cs), (L/2-cs/2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "c2", (cs, cs, cs), (-L/2+cs/2, 0, 0), (0,0,0), base_color)
    elif shape_type == "DUMBBELL":
        weight_w = t
        shaft_len = L - 2*weight_w
        create_prism_mesh(stage, obj_path, "s", t, shaft_len, 32, (0,0,0), (0,90,0), base_color)
        create_prism_mesh(stage, obj_path, "w1", L*0.2, weight_w, 32, (-L/2 + weight_w/2, 0, 0), (0,90,0), base_color)
        create_prism_mesh(stage, obj_path, "w2", L*0.2, weight_w, 32, (L/2 - weight_w/2, 0, 0), (0,90,0), base_color)
    elif shape_type == "STEP_BLOCK":
        create_box_mesh(stage, obj_path, "b1", (L, L/2, L/2), (0, -L/4, -L/4), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "b2", (L, L/2, L), (0, L/4, 0), (0,0,0), base_color)
    elif shape_type == "V_BLOCK":
        create_box_mesh(stage, obj_path, "b1", (L*0.3, L, L*0.5), (L*0.35, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "b2", (L*0.3, L, L*0.5), (-L*0.35, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "base", (L, L, L*0.2), (0, 0, -L*0.35), (0,0,0), base_color)
    elif shape_type == "PARALLEL_BAR":
        create_box_mesh(stage, obj_path, "b", (L, t*2, L*0.3), (0,0,0), (0,0,0), base_color)
    elif shape_type == "PLUS_SIGN":
        create_box_mesh(stage, obj_path, "h", (L, t, t), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "v", (t, L, t), (0,0,0), (0,0,0), base_color)
    elif shape_type == "SQUARE_FRAME":
        offset = (L - t) / 2
        create_box_mesh(stage, obj_path, "s1", (L, t, t), (0, offset, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s2", (L, t, t), (0, -offset, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s3", (t, L, t), (offset, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s4", (t, L, t), (-offset, 0, 0), (0,0,0), base_color)
    elif shape_type == "CORNER_XYZ":
        al = L
        create_box_mesh(stage, obj_path, "x", (al, t, t), (0, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "y", (t, al, t), (0, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "z", (t, t, al), (0, 0, 0), (0,0,0), base_color)
    elif shape_type == "HAMMER_HEAD":
        head_w = L * 0.3
        handle_len = L - head_w
        create_box_mesh(stage, obj_path, "hdl", (t, t, handle_len), (0, 0, -head_w/2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "hd", (L*0.4, t*2, head_w), (0, 0, handle_len/2), (0,0,0), base_color)
    elif shape_type == "OFFSET_T":
        create_box_mesh(stage, obj_path, "h", (L, t, t), (0, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "v", (t, L*0.5, t), (L*0.25, L*0.25, 0), (0,0,0), base_color)
    elif shape_type == "THREE_STEP_STAIRS":
        sh, sw = L/3, L/3
        create_box_mesh(stage, obj_path, "st1", (L, sw, sh), (0, -sw, -sh), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "st2", (L, sw, sh*2), (0, 0, -sh/2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "st3", (L, sw, L), (0, sw, 0), (0,0,0), base_color)
    elif shape_type == "SPOOL_SHAPE":
        fr, fh, ch, cr = L*0.4, t, L*0.6, L*0.12
        create_prism_mesh(stage, obj_path, "c", cr, ch, 32, (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "t", fr, fh, 32, (0,0, ch/2), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "b", fr, fh, 32, (0,0, -ch/2), (0,0,0), base_color)
    elif shape_type == "RAIL_PROFILE":
        h = L * 0.6
        create_box_mesh(stage, obj_path, "b", (L*0.5, L, t), (0,0, -h/2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "w", (t, L, h), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "h", (L*0.3, L, t*1.5), (0,0, h/2), (0,0,0), base_color)
    elif shape_type == "CROSS_3D_SOLID":
        create_box_mesh(stage, obj_path, "x", (L, L*0.25, L*0.25), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "y", (L*0.25, L, L*0.25), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "z", (L*0.25, L*0.25, L), (0,0,0), (0,0,0), base_color)
    elif shape_type == "DOUBLE_U_BACK":
        offset = (L - t)/2
        create_box_mesh(stage, obj_path, "c", (t, L, L*0.5), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "f1", (L*0.5, L, t), (L*0.25, 0, offset), (0,0,0), base_color) 
        create_box_mesh(stage, obj_path, "f2", (L*0.5, L, t), (L*0.25, 0, -offset), (0,0,0), base_color) 
        create_box_mesh(stage, obj_path, "f3", (L*0.5, L, t), (-L*0.25, 0, offset), (0,0,0), base_color) 
        create_box_mesh(stage, obj_path, "f4", (L*0.5, L, t), (-L*0.25, 0, -offset), (0,0,0), base_color) 
    elif shape_type == "GUSSET_PLATE":
        create_prism_mesh(stage, obj_path, "tri", L*0.5, t, 3, (0,0,0), (0,0,0), base_color)
    elif shape_type == "TRIANGLE_BRACKET":
        create_box_mesh(stage, obj_path, "v", (t, L*0.5, L*0.5), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "h", (L*0.4, L*0.5, t), (L*0.2+t/2, 0, -L*0.25+t/2), (0,0,0), base_color)
    elif shape_type == "SHELF_BRACKET":
        create_box_mesh(stage, obj_path, "v", (t, L*0.1, L*0.8), (-L*0.3, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "h", (L*0.6, L*0.1, t), (0, 0, -L*0.4), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "diag", t, L*0.7, 4, (0,0,0), (0,45,0), base_color)
    elif shape_type == "DIN_RAIL_SEGMENT":
        create_box_mesh(stage, obj_path, "back", (L*0.5, L, t), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s1", (t, L, t*2), (L*0.25, 0, t), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "s2", (t, L, t*2), (-L*0.25, 0, t), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "l1", (L*0.1, L, t), (L*0.3, 0, t*2), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "l2", (L*0.1, L, t), (-L*0.3, 0, t*2), (0,0,0), base_color)
    elif shape_type == "SLOTTED_RAIL":
        beam_l = L - 2*t
        create_box_mesh(stage, obj_path, "b1", (beam_l, L*0.2, t), (0, L*0.2, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "b2", (beam_l, L*0.2, t), (0, -L*0.2, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "e1", (t, L*0.6, t), (L/2 - t/2, 0, 0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "e2", (t, L*0.6, t), (-L/2 + t/2, 0, 0), (0,0,0), base_color)
    elif shape_type == "BEARING_HOUSING":
        create_box_mesh(stage, obj_path, "b", (L, L*0.2, L*0.6), (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "h", L*0.25, L*0.22, 32, (0,0,0), (90,0,0), base_color)
    elif shape_type == "MOTOR_MOUNT_PLATE":
        create_box_mesh(stage, obj_path, "p", (L*0.8, L*0.8, t), (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "c", L*0.25, t*2, 32, (0,0,0), (0,0,0), base_color)
    elif shape_type == "CHAIN_LINK":
        create_prism_mesh(stage, obj_path, "s1", t, L*0.4, 8, (L*0.2, 0, 0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "s2", t, L*0.4, 8, (-L*0.2, 0, 0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "a1", t, L*0.4, 8, (0, 0, L*0.2), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "a2", t, L*0.4, 8, (0, 0, -L*0.2), (90,0,0), base_color)
    elif shape_type == "CARABINER_D":
        create_prism_mesh(stage, obj_path, "back", t, L*0.8, 8, (-L*0.15, 0, 0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "top", t, L*0.4, 8, (0, 0, L*0.4), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "bot", t, L*0.4, 8, (0, 0, -L*0.4), (90,0,0), base_color)
        create_prism_mesh(stage, obj_path, "gate", t, L*0.6, 8, (L*0.15, 0, 0), (0,0,-15), base_color)
    elif shape_type == "PADLOCK_BODY":
        create_box_mesh(stage, obj_path, "b", (L*0.6, t*3, L*0.5), (0,0, -L*0.15), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "s", t, L*0.6, 8, (0, 0, L*0.2), (0,90,0), base_color)
    elif shape_type == "WRENCH_OPEN_END":
        create_box_mesh(stage, obj_path, "bar", (t, L*0.2, L*0.8), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "h1", (t, L*0.3, L*0.2), (0, L*0.15, L*0.4), (0,0,15), base_color)
        create_box_mesh(stage, obj_path, "h2", (t, L*0.3, L*0.2), (0, -L*0.15, L*0.4), (0,0,-15), base_color)
    elif shape_type == "WRENCH_BOX_END":
        create_box_mesh(stage, obj_path, "bar", (t, L*0.2, L*0.8), (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "box", L*0.2, t, 6, (0,0, L*0.4), (90,0,0), base_color)
    elif shape_type == "HEX_KEY_L":
        create_prism_mesh(stage, obj_path, "l1", t, L*0.8, 6, (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "l2", t, L*0.3, 6, (L*0.15, 0, -L*0.4), (0,90,0), base_color)
    elif shape_type == "SOCKET_MOCK":
        create_prism_mesh(stage, obj_path, "cyl", L*0.2, L*0.4, 32, (0,0,0), (0,0,0), base_color)
        create_prism_mesh(stage, obj_path, "hex", L*0.15, t, 6, (0,0, L*0.2), (0,0,0), base_color)
    elif shape_type == "DRIVER_BIT_PHILLIPS":
        create_prism_mesh(stage, obj_path, "shk", t, L*0.6, 6, (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "t1", (t*2, t*0.5, t*2), (0,0, L*0.3), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "t2", (t*0.5, t*2, t*2), (0,0, L*0.3), (0,0,0), base_color)
    elif shape_type == "DRIVER_BIT_FLAT":
        create_prism_mesh(stage, obj_path, "shk", t, L*0.6, 6, (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "tip", (t*3, t*0.5, t*2), (0,0, L*0.3), (0,0,0), base_color)
    elif shape_type.startswith("CHAR_"):
        char = shape_type.split("_")[1]
        h, w = L, L*0.7
        th = t
        if char in "ABDOPQR": 
            create_box_mesh(stage, obj_path, "l", (th, th, h), (-w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "r", (th, th, h), (w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "t", (w, th, th), (0, 0, h/2), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "b", (w, th, th), (0, 0, -h/2), (0,0,0), base_color)
        elif char in "CEFG": 
            create_box_mesh(stage, obj_path, "l", (th, th, h), (-w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "t", (w, th, th), (0, 0, h/2), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "b", (w, th, th), (0, 0, -h/2), (0,0,0), base_color)
            if char in "EFG": create_box_mesh(stage, obj_path, "m", (w*0.7, th, th), (0, 0, 0), (0,0,0), base_color)
        elif char in "HIJ": 
            create_box_mesh(stage, obj_path, "c", (th, th, h), (0,0,0), (0,0,0), base_color)
            if char != "I": 
                create_box_mesh(stage, obj_path, "t", (w, th, th), (0, 0, h/2), (0,0,0), base_color)
                create_box_mesh(stage, obj_path, "b", (w, th, th), (0, 0, -h/2), (0,0,0), base_color)
        elif char in "K":
            create_box_mesh(stage, obj_path, "l", (th, th, h), (-w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "k1", (th, th, h*0.6), (0, 0, h*0.25), (0,45,0), base_color)
            create_box_mesh(stage, obj_path, "k2", (th, th, h*0.6), (0, 0, -h*0.25), (0,-45,0), base_color)
        elif char in "L":
            create_box_mesh(stage, obj_path, "l", (th, th, h), (-w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "b", (w, th, th), (0, 0, -h/2), (0,0,0), base_color)
        elif char in "MNW":
            create_box_mesh(stage, obj_path, "l", (th, th, h), (-w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "r", (th, th, h), (w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "d", (th, th, h*1.1), (0,0,0), (0,30,0), base_color)
        elif char in "STXYZ": 
            create_box_mesh(stage, obj_path, "c", (th, th, h), (0,0,0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "x", (th, th, h), (0,0,0), (0,90,0), base_color)
        elif char in "UV":
            create_box_mesh(stage, obj_path, "l", (th, th, h), (-w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "r", (th, th, h), (w/2, 0, 0), (0,0,0), base_color)
            create_box_mesh(stage, obj_path, "b", (w, th, th), (0, 0, -h/2), (0,0,0), base_color)
    elif shape_type.startswith("DIGIT_"):
        create_box_mesh(stage, obj_path, "frame", (L*0.6, t, L), (0,0,0), (0,0,0), base_color)
        create_box_mesh(stage, obj_path, "cut", (L*0.4, t*2, L*0.8), (0,0,0), (0,0,0), base_color)

    return obj_path

# ==========================================
# 4. UTILS & TUNING
# ==========================================
def create_sanitized_config(original_path, new_obj_path):
    if not os.path.exists(original_path): data = {}
    else:
        with open(original_path, 'r') as f: data = yaml.safe_load(f) or {}

    keys_to_kill = ['object_path', 'mesh_path']
    for k in keys_to_kill:
        if k in data: del data[k]
    
    if 'object' in data:
        if isinstance(data['object'], dict):
            if 'path' in data['object']: del data['object']['path']
            if 'mesh_path' in data['object']: del data['object']['mesh_path']
        else:
            del data['object']
            data['object'] = {}
    else: data['object'] = {}

    data['object']['path'] = new_obj_path
    data['object_path'] = new_obj_path 
    temp_path = os.path.join(os.getcwd(), "temp_sanitized_config.yaml")
    with open(temp_path, 'w') as f: yaml.dump(data, f)
    return temp_path

def generate_raycast_pointcloud(target_prim_path, num_samples):
    qi = get_physx_scene_query_interface()
    points = []
    
    scan_radius = GLOBAL_MAX_SIZE * 2.5 
    target_jitter = GLOBAL_MAX_SIZE * 0.6 
    
    attempts = 0
    max_attempts = num_samples * 5 
    
    while len(points) < num_samples and attempts < max_attempts:
        u = np.random.uniform(-1, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        x = np.sqrt(1 - u**2) * np.cos(theta)
        y = np.sqrt(1 - u**2) * np.sin(theta)
        z = u
        origin = np.array([x, y, z]) * scan_radius
        
        target = np.random.uniform(-target_jitter, target_jitter, size=3)
        
        direction = target - origin
        dist = np.linalg.norm(direction)
        if dist < 1e-4: continue
        direction = direction / dist
        
        hit = qi.raycast_closest(
            carb.Float3(origin[0], origin[1], origin[2]),    
            carb.Float3(direction[0], direction[1], direction[2]), 
            dist * 1.5                                       
        )
        
        if hit["hit"]:
            if target_prim_path in hit["rigidBody"]:
                p = hit["position"]
                points.append([p.x, p.y, p.z])
                
        attempts += 1
        
    if len(points) == 0:
        print("AVISO: Raycast não encontrou pontos! Verifique colisores.")
        return np.zeros((0, 3))
        
    pc = np.array(points)
    if len(pc) > num_samples:
        indices = np.random.choice(len(pc), num_samples, replace=False)
        pc = pc[indices]
        
    return pc

def force_teleport(prim_path, position, orientation):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim: return
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    op_translate = None
    op_orient = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate: op_translate = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient: op_orient = op
    if not op_translate: op_translate = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    if not op_orient: op_orient = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    pos_val = Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
    if hasattr(orientation, "GetReal"): rot_val = Gf.Quatd(orientation)
    else: rot_val = Gf.Quatd(float(orientation[0]), Gf.Vec3d(float(orientation[1]), float(orientation[2]), float(orientation[3])))
    op_translate.Set(pos_val)
    op_orient.Set(rot_val)
    rb_api = UsdPhysics.RigidBodyAPI(prim)
    if rb_api:
        vel = rb_api.CreateVelocityAttr()
        ang = rb_api.CreateAngularVelocityAttr()
        vel.Set(Gf.Vec3f(0,0,0))
        ang.Set(Gf.Vec3f(0,0,0))

def apply_joint_targets(stage, targets):
    for joint_path, val in targets.items():
        prim = stage.GetPrimAtPath(joint_path)
        if not prim: continue
        attr = prim.GetAttribute("drive:angular:physics:targetPosition")
        if not attr: attr = prim.GetAttribute("drive:linear:physics:targetPosition")
        if attr: attr.Set(val)

def improve_gripper_physics(stage, gripper_path):
    print(f"--- OTIMIZANDO FÍSICA DA GARRA (LIMITES HARDWARE 64/64): {gripper_path} ---")
    for prim in Usd.PrimRange(stage.GetPrimAtPath(gripper_path)):
        if prim.IsA(UsdGeom.Mesh): 
            physxUtils.setCollider(prim, approximationShape="convexDecomposition")
            if not prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
                PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            decomp_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
            decomp_api.CreateErrorPercentageAttr(0.01) 
            decomp_api.CreateMaxConvexHullsAttr(64)
            decomp_api.CreateHullVertexLimitAttr(64)
            decomp_api.CreateShrinkWrapAttr(False)
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision = PhysxSchema.PhysxCollisionAPI(prim)
            physx_collision.CreateContactOffsetAttr(0.002) 
            physx_collision.CreateRestOffsetAttr(0.000)

# --- EXECUÇÃO OTIMIZADA COM LIMPEZA DE MEMÓRIA ---
async def run_single_iteration(iteration_index, parent_output_dir):
    world = None
    grasping_manager = None
    pc_data = None
    results = None
    sensor_map = {}
    
    try:
        nuke_physics_logs()
        
        current_iter_dir = os.path.join(parent_output_dir, f"run_{iteration_index}")
        os.makedirs(current_iter_dir, exist_ok=True)
        
        print(f"\n=============================================")
        print(f" ITERAÇÃO {iteration_index} de {TOTAL_ITERATIONS}")
        print(f" RAM CLEANUP ATIVADO")
        print(f"=============================================\n")
        
        if not os.path.exists(STAGE_PATH):
            print(f"ERRO: Stage não encontrado em {STAGE_PATH}")
            return
        
        await omni.usd.get_context().open_stage_async(STAGE_PATH)
        for _ in range(10): await omni.kit.app.get_app().next_update_async()
        stage = omni.usd.get_context().get_stage()

        generated_obj_path = generate_industrial_object(stage)
        
        for _ in range(20): await omni.kit.app.get_app().next_update_async()

        temp_config = create_sanitized_config(ORIGINAL_CONFIG_PATH, generated_obj_path)

        world = World()
        await world.initialize_simulation_context_async()
        world.set_simulation_dt(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
        
        world.play()
        for _ in range(5): world.step(render=False)
        
        pc_data = generate_raycast_pointcloud(generated_obj_path, num_samples=DENSITY_SAMPLES)
        world.pause() 

        grasping_manager = GraspingManager()
        grasping_manager.clear() 
        grasping_manager.load_config(temp_config)
        
        if hasattr(grasping_manager, "_object_path"):
            grasping_manager._object_path = generated_obj_path
        if hasattr(grasping_manager, "_config") and isinstance(grasping_manager._config, dict):
            grasping_manager._config["object_path"] = generated_obj_path
            if "object" not in grasping_manager._config: grasping_manager._config["object"] = {}
            if isinstance(grasping_manager._config["object"], dict):
                grasping_manager._config["object"]["path"] = generated_obj_path

        obj_prim = stage.GetPrimAtPath(generated_obj_path)
        if obj_prim:
            physx_rb = PhysxSchema.PhysxRigidBodyAPI(obj_prim)
            physx_rb.GetSleepThresholdAttr().Set(0.0)

        if grasping_manager.gripper_path:
            improve_gripper_physics(stage, grasping_manager.gripper_path)

        grasping_manager.grasp_locations = []
        
        if pc_data is not None and len(pc_data) > 0:
            np.save(os.path.join(current_iter_dir, "object_pointcloud.npy"), pc_data)
        
        try:
            grasping_manager.generate_grasp_poses()
            grasping_manager.store_initial_gripper_pose()
        except Exception as e:
            print(f"ERRO CRÍTICO poses: {e}")
            return 
            
        del pc_data
        pc_data = None 

        if grasping_manager.gripper_path:
            for fname in ["left_finger", "right_finger"]:
                sensor_path = f"{grasping_manager.gripper_path}/{fname}/Contact_Sensor"
                try:
                    sensor_map[fname] = ContactSensor(prim_path=sensor_path, name=f"S_{fname}", min_threshold=0, max_threshold=1e6, radius=-1)
                except: pass

        await world.reset_async()
        world.pause() 

        poses = grasping_manager.get_grasp_poses(in_world_frame=True)
        phases = grasping_manager.grasp_phases
        pregrasp = grasping_manager.joint_pregrasp_states
        results = []

        for i, pose in enumerate(poses):
            await world.reset_async()
            world.pause()
            force_teleport(grasping_manager.gripper_path, pose[0], pose[1])
            world.play() 

            if pregrasp:
                apply_joint_targets(stage, pregrasp)
                for _ in range(15): world.step(render=False)

            max_f = 0.0
            success = False
            
            for phase in phases:
                target_duration = getattr(phase, "duration", 1.0) 
                steps = int(target_duration * 60) 
                
                apply_joint_targets(stage, getattr(phase, "joint_drive_targets", {}))
                
                for s in range(steps):
                    world.step(render=False) 
                    
                    if s % RENDER_SKIP == 0: 
                        await omni.kit.app.get_app().next_update_async()

                    f_sum = 0.0
                    touch = 0
                    for sensor in sensor_map.values():
                        reading = sensor.get_current_frame()
                        in_contact = False
                        if "in_contact" in reading: in_contact = reading["in_contact"]
                        current_f = 0.0
                        if "force" in reading: current_f = np.linalg.norm(reading["force"])
                        f_sum += current_f
                        if current_f > 0.01 or (in_contact and current_f > 0.001): 
                            touch += 1
                    
                    if touch >= 2: success = True
                    if f_sum > max_f: max_f = f_sum
            
            ee_prim_path = f"{grasping_manager.gripper_path}/end_effector"
            final_pos = [0.0, 0.0, 0.0]
            final_rot = [1.0, 0.0, 0.0, 0.0] 

            if stage.GetPrimAtPath(ee_prim_path):
                w_pos, w_rot = get_world_pose(ee_prim_path)
                final_pos = w_pos.tolist()
                final_rot = w_rot.tolist() 
            else:
                w_pos, w_rot = get_world_pose(grasping_manager.gripper_path)
                final_pos = w_pos.tolist()
                final_rot = w_rot.tolist()

            status_msg = ""
            if max_f < FORCE_THRESHOLD and not success:
                status_msg = f"FAIL (Sem contato/Força Baixa: {max_f:.12f})"
            elif success:
                status_msg = f"SUCCESS (Força: {max_f:.3f})"
            else:
                status_msg = "FAIL (Indeterminado)"

            print(f"Pose {i+1}/{len(poses)}: {status_msg}")

            results.append({
                "grasp_id": i,
                "pose_pos": final_pos,
                "pose_rot": final_rot,
                "contact_success": bool(success),
                "max_force": float(max_f)
            })

        with open(os.path.join(current_iter_dir, "results_safe.yaml"), 'w') as f:
            yaml.dump(results, f)
            
        if os.path.exists(temp_config): os.remove(temp_config)
    
    finally:
        print(f"--- Limpando memória da iteração {iteration_index} ---")
        if grasping_manager: grasping_manager.clear()
        if world: world.clear_instance()
        del pc_data
        del results
        del sensor_map
        del grasping_manager
        del world
        gc.collect()
        await omni.kit.app.get_app().next_update_async()

async def run_batch_process():
    if OUTPUT_DIR_ROOT: os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)
    for i in range(TOTAL_ITERATIONS):
        await run_single_iteration(i, OUTPUT_DIR_ROOT)
    print("BATCH CONCLUÍDO!")

asyncio.ensure_future(run_batch_process())