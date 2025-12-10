import os
import xml.etree.ElementTree as ET
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from os import path
import yaml

def analyze_bt_xml(xml_path):
    """
    Analisa a BT e retorna:
    1. Um set com todas as Actions encontradas.
    2. O tipo de planejador preferido.
    """
    analysis = {
        'actions': set(),
        'planner_type': 'd_star' 
    }

    print(f"[LAUNCH DEBUG] Tentando ler BT em: {xml_path}")

    try:
        if not os.path.exists(xml_path):
            print(f"[LAUNCH ERROR] ARQUIVO NÃO ENCONTRADO! Verifique se o XML foi instalado corretamente.")
            return analysis
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            analysis['actions'].add(tag_name)

            if 'ID' in elem.attrib:
                analysis['actions'].add(elem.attrib['ID'])
            
            if tag_name == 'ComputePath' or elem.attrib.get('ID') == 'ComputePath':
                if 'planner' in elem.attrib:
                    analysis['planner_type'] = elem.attrib['planner']
                    print(f"[LAUNCH] >> Planner definido na BT: '{analysis['planner_type']}'")

        print(f"[LAUNCH DEBUG] Actions detectadas: {analysis['actions']}")

    except Exception as e:
        print(f"[LAUNCH ERROR] Erro ao ler BT XML: {e}")
    
    return analysis

def load_yaml(package_name: str, file_path: str):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = path.join(package_path, file_path)
    return parse_yaml(absolute_file_path)


def parse_yaml(absolute_file_path: str):
    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def generate_launch_description():

    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",
        description="ROS2 control hardware interface type",
    )

    use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock if true",
    )

    moveit_config = (
        MoveItConfigsBuilder("vai_se_ferrar")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": LaunchConfiguration("ros2_control_hardware_type")
            },
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .to_moveit_configs()
    )

    pkg_name_task = 'task_planning'
    pkg_name_bringup = 'mobile_bringup'

    bt_file = os.path.join(
        get_package_share_directory(pkg_name_task),
        'bt',
        'store_boxes.xml'
    )


    bt_analysis = analyze_bt_xml(bt_file)
    bt_actions = bt_analysis['actions']
    planner_choice = bt_analysis['planner_type']

    pick_place_yaml = os.path.join(
        get_package_share_directory(pkg_name_bringup),
        'config',
        'pick_and_place_poses.yaml'
    )

    label_to_storage_yaml = os.path.join(
        get_package_share_directory(pkg_name_bringup),
        'config',
        'labels_to_storage.yaml'
    )

    storage_poses_yaml = os.path.join(
        get_package_share_directory(pkg_name_bringup),
        'config',
        'storages.yaml'
    )
  
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"num_planning_attempts": 200},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("vai_se_ferrar_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            ros2_controllers_path,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )

    robot_description_kinematics = {
        "robot_description_kinematics": load_yaml(
            "vai_se_ferrar_moveit_config", path.join("config", "kinematics.yaml")
        )
    }

    robot_description_joint_limits = {
        "robot_description_planning": load_yaml(
            "vai_se_ferrar_moveit_config", path.join("config", "joint_limits.yaml")
        )
    }

    manipulation = Node(
        package="manipulation",
        executable="manipulation",
        name="manipulation",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            robot_description_joint_limits,  
            moveit_config.trajectory_execution,
            robot_description_kinematics,
            moveit_config.planning_scene_monitor,
            {'yaml_file': pick_place_yaml},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    a_star = Node(
        package="navigation",
        executable="a_star",
        name="a_star",
        output="screen",
        parameters=[
            {'path_resolution': 0.05},
            {'security_distance': 0.0},
            {'iterations_before_verification': 20},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )
    
    d_star = Node(
        package="navigation",
        executable="d_star",
        name="d_star",
        output="screen",
        parameters=[
            {'path_resolution': 0.05},
            {'security_distance': 0.4},
            {'iterations_before_verification': 20},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    controller = Node(
        package="navigation",
        executable="controller",
        name="controller",
        output="screen",
        parameters=[
            {'path_resolution': 0.05},
            {'iterations_before_verification': 20},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )



    server_arguments = ["--ros-args", "--log-level", "info"]
    

    server_arguments.append("--")

    if 'GetStorageInfo' not in bt_actions:
        print("[LAUNCH] >> 'GetStorageInfo' não encontrado. Desativando StorageNode.")
        server_arguments.append("--no-storage")
    else:
        print("[LAUNCH] >> 'GetStorageInfo' detectado. Ativando StorageNode.")

    if 'ComputePoseToOrganize' not in bt_actions:
        print("[LAUNCH] >> 'ComputePoseToOrganize' não encontrado. Desativando OrganizeNode.")
        server_arguments.append("--no-organize")
    else:
        print("[LAUNCH] >> 'Organize' detectado. Ativando OrganizeNode.")

    if 'IsGripperHoldingObject' not in bt_actions:
        print("[LAUNCH] >> 'IsGripperHoldingObject' não encontrado. Desativando GripperNode.")
        server_arguments.append("--no-gripper")
    else:
        print("[LAUNCH] >> 'IsGripperHoldingObject' detectado. Ativando GripperNode.")

    server_node = Node(
        package="task_planning",
        executable="server_node",
        name="server_node",
        output="screen",
        parameters=[
            {'yaml_file': pick_place_yaml},
            {'bt_xml_path': bt_file},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            
            {'label_to_storage_yaml_file': label_to_storage_yaml},
            {'storage_poses_yaml_file': storage_poses_yaml},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic, # SRDF (Crucial!)
            moveit_config.robot_description_kinematics
        ],
        # AQUI USAMOS A LISTA GERADA DINAMICAMENTE
        arguments=server_arguments, 
    )

    labels_yaml_file = os.path.join(
        get_package_share_directory(pkg_name_bringup),
        'config',
        'labels.yaml'
    )

    add_collision = Node(
        package="manipulation",
        executable="add_collision",
        name="add_collision",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            robot_description_joint_limits,  
            moveit_config.trajectory_execution,
            robot_description_kinematics,
            moveit_config.planning_scene_monitor,
            {'yaml_file': labels_yaml_file},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        remappings=[
            ('/boxes_detection_array', '/bbox_3d_with_labels')
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    map_dir = os.path.join(
        get_package_share_directory('mobile_bringup'),
        'maps'
    )
    map_yaml_filename = os.path.join(map_dir, 'multiple_storages.yaml')
    map_pgm_filename = os.path.join(map_dir, 'multiple_storages.png')

    obstacle_graph_with_occupancy_grid = Node(
        package="navigation",
        executable="obstacle_graph_with_occupancy_grid",
        name="obstacle_graph_with_occupancy_grid",
        output="screen",
        parameters=[
            {'map_yaml_file': map_yaml_filename},
            {'map_image_file': map_pgm_filename},
            {'max_security_distance': 0.45},
            {'obstacle_graph_resolution': 0.05},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    synchronize_isaac = Node(
        package='isaacsim_moveit',
        executable='synchronize_isaac_sim_labels',
        name='synchronize_isaac_sim_labels',
        output='screen',
    )
    
    rviz_config_file = os.path.join(
        get_package_share_directory("isaacsim_moveit"),
        "rviz",
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
   
        ],
    )
    
    final_launch_list = [
        ros2_control_hardware_type,
        # rviz_node,
        use_sim_time,
        robot_state_publisher,
        move_group_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        panda_arm_controller_spawner,
        server_node,           
        synchronize_isaac,
    ]


    if 'ComputePath' in bt_actions or 'ComputePathToPose' in bt_actions:
        print(f"[LAUNCH] >> Navegação detectada na BT. Algoritmo escolhido: {planner_choice}")
        
        if planner_choice == 'a_star':
            final_launch_list.append(a_star)
        final_launch_list.append(obstacle_graph_with_occupancy_grid)

    if 'PickObject' in bt_actions or 'PlaceObject' in bt_actions or 'DetectObject' in bt_actions:
        print("[LAUNCH] >> Manipulação (Pick/Place/Detect) detectada na BT.")
        final_launch_list.append(manipulation)
        final_launch_list.append(add_collision)

    if 'NavigateTo' in bt_actions or 'FollowPath' in bt_actions:
        print("[LAUNCH] >> Controlador de caminho (NavigateTo) detectado.")
        final_launch_list.append(controller)

    return LaunchDescription(final_launch_list)