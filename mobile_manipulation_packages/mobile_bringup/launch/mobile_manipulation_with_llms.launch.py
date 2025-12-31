import os
import yaml
import tempfile
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from os import path


def load_yaml(package_name: str, file_path: str):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def generate_launch_description():

    # =========================================================================
    # ARGUMENTOS
    # =========================================================================
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

    # =========================================================================
    # MOVEIT CONFIG
    # =========================================================================
    moveit_config = (
        MoveItConfigsBuilder("vai_se_ferrar")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={
                "ros2_control_hardware_type": "isaac"
            },
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .to_moveit_configs()
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

    
    pkg_task = get_package_share_directory('task_planning')
    pkg_bringup = get_package_share_directory('mobile_bringup')
    pkg_moveit = get_package_share_directory('vai_se_ferrar_moveit_config')
    
    pick_place_yaml = os.path.join(pkg_bringup, 'config', 'pick_and_place_poses.yaml')
    label_to_storage_yaml = os.path.join(pkg_bringup, 'config', 'labels_to_storage.yaml')
    storage_poses_yaml = os.path.join(pkg_bringup, 'config', 'storages.yaml')
    labels_yaml = os.path.join(pkg_bringup, 'config', 'labels.yaml')
    subtrees_path = os.path.join(pkg_task, 'bt', 'LLM_subtrees')
    database_path = '/home/momesso/pibic/src/mobile_manipulation_packages/llms/db/robot_world_data.db'
    
    map_dir = os.path.join(pkg_bringup, 'maps')
    map_yaml = os.path.join(map_dir, 'multiple_storages.yaml')
    map_image = os.path.join(map_dir, 'multiple_storages.png')

    ros2_controllers_path = os.path.join(pkg_moveit, 'config', 'ros2_controllers.yaml')

    base_params_file = os.path.join(pkg_bringup, 'config', 'ros_parameters.yaml')
    
    with open(base_params_file, 'r') as f:
        params_dict = yaml.safe_load(f)
    
   
    params_dict['server_node_with_llms']['ros__parameters']['yaml_file'] = pick_place_yaml
    params_dict['server_node_with_llms']['ros__parameters']['subtrees_path'] = subtrees_path
    params_dict['server_node_with_llms']['ros__parameters']['database_path'] = database_path
    
    params_dict['storage_node']['ros__parameters']['label_to_storage_yaml_file'] = label_to_storage_yaml
    params_dict['storage_node']['ros__parameters']['storage_poses_yaml_file'] = storage_poses_yaml
    
    params_dict['world_state_node']['ros__parameters']['database_path'] = database_path
    
    params_dict['obstacle_graph_with_occupancy_grid']['ros__parameters']['map_yaml_file'] = map_yaml
    params_dict['obstacle_graph_with_occupancy_grid']['ros__parameters']['map_image_file'] = map_image
    
    params_dict['manipulation']['ros__parameters']['yaml_file'] = pick_place_yaml
    
    params_dict['add_collision_objects']['ros__parameters']['yaml_file'] = labels_yaml

    
    params_file = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.yaml', 
        delete=False,
        prefix='ros_params_'
    )
    yaml.dump(params_dict, params_file, default_flow_style=False)
    params_file.close()
    
    print(f"[LAUNCH] Parâmetros escritos em: {params_file.name}")


    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            params_file.name,
            moveit_config.to_dict(),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
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

    server_node_with_llms = Node(
        package="task_planning",
        executable="server_node_with_llms",
        name="server_node_with_llms",
        output="screen",
        parameters=[
            params_file.name,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    bridge_to_phi_35 = Node(
        package="llms",
        executable="bridge_to_phi_3_5",
        name="bridge_to_phi_3_5",
        output="screen",
        parameters=[
            params_file.name,
            
        ],
    )

    a_star = Node(
        package="navigation",
        executable="a_star",
        name="a_star",
        output="screen",
        parameters=[
            params_file.name,
        ],
    )

    controller = Node(
        package="navigation",
        executable="controller",
        name="controller",
        output="screen",
        parameters=[
            params_file.name,
        ],
    )

    obstacle_graph = Node(
        package="navigation",
        executable="obstacle_graph_with_occupancy_grid",
        name="obstacle_graph_with_occupancy_grid",
        output="screen",
        parameters=[
            params_file.name,
        ],
    )

    manipulation = Node(
        package="manipulation",
        executable="manipulation",
        name="manipulation",
        output="screen",
        parameters=[
            params_file.name,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            robot_description_joint_limits,
            moveit_config.trajectory_execution,
            robot_description_kinematics,
            moveit_config.planning_scene_monitor,
        ],
    )

    synchronize_isaac = Node(
        package='isaacsim_moveit',
        executable='synchronize_isaac_sim_labels',
        name='synchronize_isaac_sim_labels',
        output='screen',
        parameters=[
            params_file.name,
        ],
    )

    add_collision = Node(
        package="manipulation",
        executable="add_collision",
        name="add_collision",
        output="screen",
        parameters=[
            params_file.name,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            robot_description_joint_limits,
            moveit_config.trajectory_execution,
            robot_description_kinematics,
            moveit_config.planning_scene_monitor,
        ],
        remappings=[
            ('/boxes_detection_array', '/bbox_3d_with_labels')
        ],
    )

    
    return LaunchDescription([
        ros2_control_hardware_type,
        use_sim_time,
        robot_state_publisher,
        move_group_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        panda_arm_controller_spawner,
        server_node_with_llms,
        bridge_to_phi_35,
        a_star,
        controller,
        obstacle_graph,
        manipulation,
        synchronize_isaac,
    ])