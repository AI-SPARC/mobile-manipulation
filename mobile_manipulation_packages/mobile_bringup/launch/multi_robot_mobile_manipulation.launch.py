import os
import yaml
import tempfile
import copy
import xml.etree.ElementTree as ET
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from os import path

# =============================================================================
# 1. FUNÇÕES AUXILIARES DE CORREÇÃO (YAML & BT)
# =============================================================================

def analyze_bt_xml(xml_path):
    analysis = {'actions': set(), 'planner_type': 'd_star'}
    try:
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            for elem in tree.getroot().iter():
                tag = elem.tag.split('}')[-1]
                analysis['actions'].add(tag)
                if 'ID' in elem.attrib: analysis['actions'].add(elem.attrib['ID'])
                if (tag == 'ComputePath' or elem.attrib.get('ID') == 'ComputePath') and 'planner' in elem.attrib:
                    analysis['planner_type'] = elem.attrib['planner']
    except Exception: pass
    return analysis

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def dump_yaml_to_tempfile(content_dict):
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp_file:
            yaml.dump(content_dict, tmp_file, default_flow_style=False)
            return tmp_file.name
    except Exception as e:
        print(f"[LAUNCH ERROR] Temp YAML failure: {e}")
        return None

def fix_task_yaml_prefixes(file_path, prefix):
    """
    Lê um arquivo YAML de tarefa (Pick & Place) e injeta o prefixo
    nos nomes de links e juntas (panda_link8 -> robot1_panda_link8).
    Retorna o caminho do arquivo temporário corrigido.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Substituições inteligentes para garantir que o nó Manipulation encontre os links
        # Evita dupla substituição se já tiver prefixo
        if f"{prefix}panda" not in content:
            content = content.replace("panda_link", f"{prefix}panda_link")
            content = content.replace("panda_joint", f"{prefix}panda_joint")
            content = content.replace("panda_hand", f"{prefix}panda_hand")
            content = content.replace("panda_finger", f"{prefix}panda_finger")
            # Corrige referências soltas ao braço se houver
            content = content.replace("frame_id: panda_link", f"frame_id: {prefix}panda_link")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmp_file:
            tmp_file.write(content)
            return tmp_file.name
    except Exception as e:
        print(f"[LAUNCH ERROR] Falha ao corrigir YAML de tarefa: {e}")
        return file_path # Retorna o original se falhar

def transform_yaml_for_multi_robots(original_yaml, prefixes):
    if original_yaml is None: return {}
    multi_robot_yaml = {}
    
    global_keys = ['planner_configs', 'trajectory_execution', 'moveit_controller_manager', 
                   'moveit_simple_controller_manager', 'robot_description_planning']
    
    # --- JOINT LIMITS ---
    if 'joint_limits' in original_yaml:
        multi_robot_yaml['joint_limits'] = {}
        for prefix in prefixes:
            for joint, limits in original_yaml['joint_limits'].items():
                clean_joint = joint.replace("panda_", "")
                new_joint_name = f"{prefix}panda_{clean_joint}"
                multi_robot_yaml['joint_limits'][new_joint_name] = copy.deepcopy(limits)
            
            # Injeta aceleração para a mão (Resolve erro AddTimeOptimalParameterization)
            for finger in ['finger_joint1', 'finger_joint2']:
                finger_name = f"{prefix}panda_{finger}"
                if finger_name not in multi_robot_yaml['joint_limits']:
                    multi_robot_yaml['joint_limits'][finger_name] = {
                        'has_velocity_limits': True, 'max_velocity': 0.2,
                        'has_acceleration_limits': True, 'max_acceleration': 5.0
                    }
                else:
                    multi_robot_yaml['joint_limits'][finger_name]['has_acceleration_limits'] = True
                    multi_robot_yaml['joint_limits'][finger_name]['max_acceleration'] = 5.0
        return multi_robot_yaml

    # --- KINEMATICS E OMPL ---
    for key, value in original_yaml.items():
        if key in global_keys:
            multi_robot_yaml[key] = value
        else:
            for prefix in prefixes:
                new_group_name = f"{prefix}{key}"
                new_config = copy.deepcopy(value)
                
                # Garante que tip_name tenha o prefixo se existir
                if 'tip_name' in new_config and 'panda_' in new_config['tip_name']:
                    clean_tip = new_config['tip_name'].replace("panda_", "")
                    new_config['tip_name'] = f"{prefix}panda_{clean_tip}"
                
                multi_robot_yaml[new_group_name] = new_config
                
    return multi_robot_yaml

def generate_controllers_dynamic(prefixes):
    ros2_ctrl = {
        "controller_manager": {
            "ros__parameters": {
                "update_rate": 100,
                "joint_state_broadcaster": {"type": "joint_state_broadcaster/JointStateBroadcaster"}
            }
        }
    }
    moveit_ctrl = {
        "moveit_simple_controller_manager": {
            "controller_names": [],
            "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        }
    }
    for prefix in prefixes:
        arm_controller = f"{prefix}panda_arm_controller"
        hand_controller = f"{prefix}panda_hand_controller"
        
        # ROS2 Control
        ros2_ctrl["controller_manager"]["ros__parameters"][arm_controller] = {"type": "joint_trajectory_controller/JointTrajectoryController"}
        ros2_ctrl["controller_manager"]["ros__parameters"][hand_controller] = {"type": "position_controllers/GripperActionController"}
        
        ros2_ctrl[arm_controller] = {"ros__parameters": {
            "joints": [f"{prefix}panda_joint{i}" for i in range(1, 8)],
            "command_interfaces": ["position"], "state_interfaces": ["position", "velocity"],
            "state_publish_rate": 100.0, "action_monitor_rate": 20.0
        }}
        ros2_ctrl[hand_controller] = {"ros__parameters": {
            "joint": f"{prefix}panda_finger_joint1",
            "goal_tolerance": 0.01, "stalled_velocity_threshold": 0.01, "stall_timeout": 0.2
        }}

        # MoveIt
        moveit_ctrl["moveit_simple_controller_manager"]["controller_names"].extend([arm_controller, hand_controller])
        moveit_ctrl["moveit_simple_controller_manager"][arm_controller] = {
            "action_ns": "follow_joint_trajectory", "type": "FollowJointTrajectory", "default": True,
            "joints": [f"{prefix}panda_joint{i}" for i in range(1, 8)]
        }
        moveit_ctrl["moveit_simple_controller_manager"][hand_controller] = {
            "action_ns": "gripper_cmd", "type": "GripperCommand", "default": True,
            "joints": [f"{prefix}panda_finger_joint1"]
        }
    return ros2_ctrl, moveit_ctrl

# =============================================================================
# 2. LAUNCH PRINCIPAL
# =============================================================================

def generate_launch_description():
    robots_prefixes = ['robot1_', 'robot2_', 'robot3_']
    config_pkg = "vai_se_ferrar_moveit_config" 
    pkg_name_task = 'task_planning'
    pkg_name_bringup = 'mobile_bringup'

    ros2_control_hardware_type = DeclareLaunchArgument("ros2_control_hardware_type", default_value="isaac")
    use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")
    rviz_config_arg = DeclareLaunchArgument("rviz_config", default_value="moveit.rviz")

    # --- 1. Carregar Configs ---
    raw_kinematics = load_yaml(config_pkg, "config/kinematics.yaml")
    raw_joint_limits = load_yaml(config_pkg, "config/joint_limits.yaml")
    raw_ompl = load_yaml(config_pkg, "config/ompl_planning.yaml")
    
    multi_kinematics_content = transform_yaml_for_multi_robots(raw_kinematics, robots_prefixes)
    multi_joint_limits_content = transform_yaml_for_multi_robots(raw_joint_limits, robots_prefixes)
    multi_ompl_content = transform_yaml_for_multi_robots(raw_ompl, robots_prefixes)
    
    multi_kinematics_param = {"robot_description_kinematics": multi_kinematics_content}
    multi_joint_limits_param = {"robot_description_planning": multi_joint_limits_content}
    
    dict_ros2_ctrl, dict_moveit_ctrl = generate_controllers_dynamic(robots_prefixes)
    ros2_ctrl_yaml_file = dump_yaml_to_tempfile(dict_ros2_ctrl)

    # --- 2. MoveIt Config ---
    moveit_config = (
        MoveItConfigsBuilder("multi_panda_system", package_name=config_pkg)
        .robot_description(
            file_path="xacro_config/multi_robots.urdf.xacro", 
            mappings={"ros2_control_hardware_type": LaunchConfiguration("ros2_control_hardware_type")}
        )
        .robot_description_semantic(file_path="xacro_config/multi_robots.srdf.xacro")
        # Força OMPL para evitar erro de ID vazio do Pilz
        .planning_pipelines(pipelines=["ompl"]) 
        .to_moveit_configs()
    )

    # --- 3. PATHS e CORREÇÃO DE ARQUIVOS DE TAREFA ---
    bt_file = os.path.join(get_package_share_directory(pkg_name_task), 'bt', 'storage_boxes.xml')
    bt_analysis = analyze_bt_xml(bt_file)
    bt_actions = bt_analysis['actions']
    planner_choice = bt_analysis['planner_type']

    # Caminhos originais
    original_pick_yaml = os.path.join(get_package_share_directory(pkg_name_bringup), 'config', 'pick_and_place_poses.yaml')
    label_to_storage_yaml = os.path.join(get_package_share_directory(pkg_name_bringup), 'config', 'labels_to_storage.yaml')
    storage_poses_yaml = os.path.join(get_package_share_directory(pkg_name_bringup), 'config', 'storages.yaml')
    labels_yaml_file = os.path.join(get_package_share_directory(pkg_name_bringup), 'config', 'labels.yaml')
    
    # === FIX: GERA VERSÃO CORRIGIDA DO YAML DE TAREFA ===
    # Isso substitui 'panda_link8' por 'robot1_panda_link8' dentro do arquivo que o nó manipulation lê
    corrected_pick_yaml = fix_task_yaml_prefixes(original_pick_yaml, "robot1_")
    print(f"[LAUNCH FIX] YAML de tarefa corrigido gerado em: {corrected_pick_yaml}")

    map_dir = os.path.join(get_package_share_directory(pkg_name_bringup), 'maps')
    map_yaml_filename = os.path.join(map_dir, 'mobile_manipulation_organize.yaml')
    map_pgm_filename = os.path.join(map_dir, 'mobile_manipulation_organize.png')

    # --- 4. Nós Principais ---

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            multi_kinematics_param,
            multi_joint_limits_param,
            multi_ompl_content,
            dict_moveit_ctrl,
            {"default_planning_pipeline": "ompl"}, 
            {"num_planning_attempts": 10},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        name="robot_state_publisher", output="both",
        parameters=[moveit_config.robot_description, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )
    
    ros2_control_node = Node(
        package="controller_manager", executable="ros2_control_node",
        parameters=[ros2_ctrl_yaml_file, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        remappings=[("/controller_manager/robot_description", "/robot_description")],
        output="screen",
    )

    rviz_file = os.path.join(get_package_share_directory("isaacsim_moveit"), "rviz", "moveit.rviz")
    if not os.path.exists(rviz_file):
        rviz_file = os.path.join(get_package_share_directory(config_pkg), "config", "moveit.rviz")

    rviz_node = Node(
        package="rviz2", executable="rviz2", name="rviz2", output="log",
        arguments=["-d", rviz_file],
        parameters=[
            moveit_config.robot_description, moveit_config.robot_description_semantic,
            multi_ompl_content, multi_kinematics_param, multi_joint_limits_param,
            {"use_sim_time": LaunchConfiguration("use_sim_time")}
        ],
    )

    spawners = [Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster", "-c", "/controller_manager"])]
    for prefix in robots_prefixes:
        spawners.append(Node(package="controller_manager", executable="spawner", arguments=[f"{prefix}panda_arm_controller", "-c", "/controller_manager"]))
        spawners.append(Node(package="controller_manager", executable="spawner", arguments=[f"{prefix}panda_hand_controller", "-c", "/controller_manager"]))

    # --- 5. Task e Manipulation ---

    server_args = ["--ros-args", "--log-level", "info", "--"]
    if 'GetStorageInfo' not in bt_actions: server_args.append("--no-storage")
    if 'Organize' not in bt_actions: server_args.append("--no-organize")
    if 'IsGripperHoldingObject' not in bt_actions: server_args.append("--no-gripper")

    server_node = Node(
        package="task_planning", executable="server_node", name="server_node", output="screen",
        parameters=[
            {'yaml_file': corrected_pick_yaml}, # USA O YAML CORRIGIDO AQUI
            {'bt_xml_path': bt_file},
            {'label_to_storage_yaml_file': label_to_storage_yaml}, 
            {'storage_poses_yaml_file': storage_poses_yaml},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=server_args,
    )

    synchronize_isaac = Node(package='isaacsim_moveit', executable='synchronize_isaac_sim_labels', name='synchronize_isaac_sim_labels', output='screen')

    manipulation = Node(
        package="manipulation", executable="manipulation", name="manipulation", output="screen",
        parameters=[
            moveit_config.robot_description, moveit_config.robot_description_semantic,
            multi_kinematics_param, multi_joint_limits_param,
            moveit_config.planning_pipelines, moveit_config.trajectory_execution, 
            moveit_config.planning_scene_monitor,
            {'yaml_file': corrected_pick_yaml}, # USA O YAML CORRIGIDO AQUI TAMBÉM
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    add_collision = Node(
        package="manipulation", executable="add_collision", name="add_collision", output="screen",
        parameters=[
            moveit_config.robot_description, moveit_config.robot_description_semantic,
            multi_kinematics_param, multi_joint_limits_param,
            moveit_config.planning_pipelines, moveit_config.trajectory_execution, 
            moveit_config.planning_scene_monitor,
            {'yaml_file': labels_yaml_file}, {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        remappings=[('/boxes_detection_array', '/bbox_3d_with_labels')],
        arguments=["--ros-args", "--log-level", "info"],
    )

    nav_params = [
        {'path_resolution': 0.05}, {'security_distance': 0.4}, {'iterations_before_verification': 20},
        {"use_sim_time": LaunchConfiguration("use_sim_time")}
    ]
    a_star = Node(package="navigation", executable="a_star", name="a_star", output="screen", parameters=nav_params)
    d_star = Node(package="navigation", executable="d_star", name="d_star", output="screen", parameters=nav_params)
    controller = Node(package="navigation", executable="controller", name="controller", output="screen", parameters=nav_params)
    
    obstacle_graph = Node(
        package="navigation", executable="obstacle_graph_with_occupancy_grid", name="obstacle_graph_with_occupancy_grid", output="screen",
        parameters=[
            {'map_yaml_file': map_yaml_filename}, {'map_image_file': map_pgm_filename},
            {'max_security_distance': 0.45}, {'obstacle_graph_resolution': 0.05},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    # --- 6. Final List ---
    
    final_launch_list = [
        ros2_control_hardware_type, use_sim_time, rviz_config_arg,
        robot_state_publisher, move_group_node, ros2_control_node, rviz_node,
        *spawners,
        server_node, synchronize_isaac
    ]

    if 'ComputePath' in bt_actions or 'ComputePathToPose' in bt_actions:
        print(f"[LAUNCH] >> Navegação detectada (Planner: {planner_choice})")
        if planner_choice == 'a_star': final_launch_list.append(a_star)
        else: final_launch_list.append(d_star)
        final_launch_list.append(obstacle_graph)

    if 'PickObject' in bt_actions or 'PlaceObject' in bt_actions or 'DetectObject' in bt_actions:
        print("[LAUNCH] >> Manipulação detectada")
        final_launch_list.append(manipulation)
        final_launch_list.append(add_collision)

    if 'NavigateTo' in bt_actions or 'FollowPath' in bt_actions:
        print("[LAUNCH] >> Controlador detectado")
        final_launch_list.append(controller)

    return LaunchDescription(final_launch_list)