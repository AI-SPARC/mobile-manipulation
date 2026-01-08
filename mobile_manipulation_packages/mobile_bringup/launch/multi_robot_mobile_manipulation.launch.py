import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from os import path
import yaml

# =============================================================================
# FUNÇÃO AUXILIAR PARA CARREGAR YAML
# =============================================================================
def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

# =============================================================================
# GERAÇÃO DO LAUNCH
# =============================================================================
def generate_launch_description():
    
    # ---------------------------------------------------------
    # 1. DEFINIÇÃO DE ARGUMENTOS E CONSTANTES
    # ---------------------------------------------------------
    num_robots = 6  
    
    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type", 
        default_value="isaac",
        description="Tipo de interface de hardware (isaac ou mock)"
    )
    
    use_sim_time = DeclareLaunchArgument(
        "use_sim_time", 
        default_value="true",
        description="Usar relógio de simulação"
    )

    # Nomes dos pacotes
    pkg_name_task = 'task_planning'
    pkg_name_bringup = 'mobile_bringup'
    # NOME EXATO DO SEU PACOTE:
    pkg_moveit_config = 'vai_se_ferrar_moveit_config' 

    # Caminhos de Arquivos de Configuração
    bt_file = os.path.join(get_package_share_directory(pkg_name_task), 'bt', 'store_boxes.xml')
    
    pick_place_yaml = os.path.join(get_package_share_directory(pkg_name_bringup), 'config', 'pick_and_place_poses.yaml')
    label_to_storage_yaml = os.path.join(get_package_share_directory(pkg_name_bringup), 'config', 'labels_to_storage.yaml')
    storage_poses_yaml = os.path.join(get_package_share_directory(pkg_name_bringup), 'config', 'storages.yaml')
    labels_yaml_file = os.path.join(get_package_share_directory(pkg_name_bringup), 'config', 'labels.yaml')

    # Mapas
    map_dir = os.path.join(get_package_share_directory('mobile_bringup'), 'maps')
    map_yaml_file = os.path.join(map_dir, 'multiple_storages.yaml')
    map_image_file = os.path.join(map_dir, 'multiple_storages.png')

    # Carregamento manual de configs
    kinematics_yaml = load_yaml(pkg_moveit_config, "config/kinematics.yaml")
    joint_limits_yaml = load_yaml(pkg_moveit_config, "config/joint_limits.yaml")

    ld_actions = [ros2_control_hardware_type, use_sim_time]

    # ---------------------------------------------------------
    # 2. LOOP DE CRIAÇÃO DOS ROBÔS
    # ---------------------------------------------------------
    for i in range(num_robots):
        
        if i == 0:
            robot_name = "robot_0"
        else:
            robot_name = f"robot_{i:02d}"

        prefix = f"{robot_name}/"
        
        print(f"[LAUNCH] Configurando namespace: {robot_name} | Prefixo: {prefix}")

        # --- CORREÇÃO AQUI ---
        # Passamos "panda" como nome do robô (convenção) e 
        # package_name EXPLÍCITO para evitar o erro "package_moveit_config_moveit_config"
        moveit_config = (
            MoveItConfigsBuilder("panda", package_name=pkg_moveit_config)
            .robot_description(
                file_path="config/panda.urdf.xacro",
                mappings={
                    "ros2_control_hardware_type": LaunchConfiguration("ros2_control_hardware_type"),
                    "prefix": prefix 
                },
            )
            .robot_description_semantic(file_path="config/panda.srdf")
            .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
            .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
            .robot_description_kinematics(file_path="config/kinematics.yaml")
            .to_moveit_configs()
        )

        robot_state_publisher = Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="both",
            parameters=[
                moveit_config.robot_description,
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
                {"frame_prefix": prefix} 
            ],
        )

        ros2_controllers_path = os.path.join(
            get_package_share_directory(pkg_moveit_config),
            "config", "ros2_controllers.yaml"
        )
        
        ros2_control_node = Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                ros2_controllers_path,
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
            remappings=[("/controller_manager/robot_description", "robot_description")],
            output="screen",
        )

        spawners = [
            Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster", "-c", "controller_manager"]),
            Node(package="controller_manager", executable="spawner", arguments=["panda_arm_controller", "-c", "controller_manager"]),
            Node(package="controller_manager", executable="spawner", arguments=["panda_hand_controller", "-c", "controller_manager"]),
        ]

        move_group_node = Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {"num_planning_attempts": 10},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
            arguments=["--ros-args", "--log-level", "error"], 
        )

        manipulation_node = Node(
            package="manipulation",
            executable="manipulation",
            name="manipulation",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                {"robot_description_kinematics": kinematics_yaml},
                moveit_config.planning_pipelines,
                {"robot_description_planning": joint_limits_yaml},
                moveit_config.trajectory_execution,
                moveit_config.planning_scene_monitor,
                {'yaml_file': pick_place_yaml},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        )
        
        add_collision_node = Node(
            package="manipulation",
            executable="add_collision",
            name="add_collision",
            output="screen",
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                {"robot_description_kinematics": kinematics_yaml},
                moveit_config.planning_pipelines,
                {"robot_description_planning": joint_limits_yaml},  
                moveit_config.trajectory_execution,
                {"robot_description_kinematics": kinematics_yaml},
                moveit_config.planning_scene_monitor,
                {'yaml_file': labels_yaml_file},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
            remappings=[
                ('/boxes_detection_array', '/bbox_3d_with_labels')
            ]
        )

        nav_params = [
            {'path_resolution': 0.05}, 
            {'iterations_before_verification': 20}, 
            {"use_sim_time": LaunchConfiguration("use_sim_time")}
        ]

        a_star = Node(package="navigation", executable="a_star", name="a_star", parameters=nav_params)
        
        # d_star = Node(
        #     package="navigation", executable="d_star", name="d_star", 
        #     parameters=[{'security_distance': 0.4}, *nav_params]
        # )

        controller = Node(
            package="navigation", executable="controller", name="controller", 
            parameters=nav_params
        )

        obstacle_graph = Node(
            package="navigation",
            executable="obstacle_graph_with_occupancy_grid",
            name="obstacle_graph_with_occupancy_grid",
            output="screen",
            parameters=[
                {'map_yaml_file': map_yaml_file},
                {'map_image_file': map_image_file},
                {'max_security_distance': 0.45},
                {'obstacle_graph_resolution': 0.05},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ]
        )

        robot_group = GroupAction([
            PushRosNamespace(robot_name),
            robot_state_publisher,
            ros2_control_node,
            *spawners,
            move_group_node,
            manipulation_node,
            add_collision_node,
            a_star,     
            controller,
            obstacle_graph
        ])

        ld_actions.append(robot_group)

    # ---------------------------------------------------------
    # 3. NÓS GLOBAIS
    # ---------------------------------------------------------
    
    # --- CORREÇÃO AQUI TAMBÉM ---
    server_moveit_config = (
        MoveItConfigsBuilder("panda", package_name=pkg_moveit_config)
        .robot_description(file_path="config/panda.urdf.xacro", mappings={"ros2_control_hardware_type": "isaac", "prefix": ""})
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .to_moveit_configs()
    )

    server_node = Node(
        package="task_planning",
        executable="server_node",
        name="server_node",
        output="screen",
        parameters=[
            {'yaml_file': pick_place_yaml},
            {'bt_xml_path': bt_file},
            {'label_to_storage_yaml_file': label_to_storage_yaml},
            {'storage_poses_yaml_file': storage_poses_yaml},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            
            server_moveit_config.robot_description,
            server_moveit_config.robot_description_semantic,
            server_moveit_config.robot_description_kinematics
        ],
        arguments=["--ros-args", "--log-level", "info"] 
    )

    synchronize_isaac = Node(
        package='isaacsim_moveit',
        executable='synchronize_isaac_sim_labels',
        name='synchronize_isaac_sim_labels',
        output='screen',
    )
    
    ld_actions.append(server_node)
    ld_actions.append(synchronize_isaac)

    return LaunchDescription(ld_actions)