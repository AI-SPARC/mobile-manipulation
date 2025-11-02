import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from os import path
import yaml

def generate_launch_description():

    denso_robot_description_pkg = "denso_robot_description"
    denso_robot_moveit_config_pkg = "denso_robot_moveit_config"

    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="1",
        description="Modo de operação do robô (0 = padrão, 1 = inclui solda)",
    )

    isaac_sim_arg = DeclareLaunchArgument(
        "isaac",
        default_value="true",
        description="Ativa a interface de hardware para o Isaac Sim"
    )

    gazebo_sim_arg = DeclareLaunchArgument(
        "sim_gazebo",
        default_value="false",
        description="Ativa a interface de hardware para o Gazebo"
    )

    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",
        description="ROS2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )

    use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock if true",
    )
    
    planning_pipeline = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": [
                "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planning_request_adapters/ValidateWorkspaceBounds",
                "default_planning_request_adapters/CheckStartStateBounds",
                "default_planning_request_adapters/CheckStartStateCollision",
            ],
            "response_adapters": [
                "default_planning_response_adapters/AddTimeOptimalParameterization",
                "default_planning_response_adapters/ValidateSolution",
                "default_planning_response_adapters/DisplayMotionPath",
            ],
            "start_state_max_bounds_error": 0.31416,
        },
    }
    _ompl_yaml = load_yaml(
        denso_robot_moveit_config_pkg, path.join("config", "ompl_planning.yaml")
    )
    planning_pipeline["ompl"].update(_ompl_yaml)

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    
    moveit_configs_builder = MoveItConfigsBuilder("denso_robot")

    robot_description_path = os.path.join(
        get_package_share_directory(denso_robot_description_pkg),
        "urdf",
        "denso_robot.urdf.xacro",
    )
    robot_description_semantic_path = os.path.join(
        get_package_share_directory(denso_robot_moveit_config_pkg),
        "srdf",
        "denso_robot.srdf.xacro",
    )
    controllers_path = os.path.join(
        get_package_share_directory(denso_robot_moveit_config_pkg),
        "config",
        "controllers.yaml",
    )

    moveit_configs_builder.robot_description(
        file_path=robot_description_path,
        mappings={
            "ros2_control_hardware_type": LaunchConfiguration("ros2_control_hardware_type"),
            "isaac": LaunchConfiguration("isaac"),
            "sim_gazebo": LaunchConfiguration("sim_gazebo"),
            "mode": LaunchConfiguration("mode"),
        },
    )

    moveit_configs_builder.robot_description_semantic(
        file_path=robot_description_semantic_path,
        mappings={
            "mode": LaunchConfiguration("mode"),  
        },
    )
    moveit_configs_builder.trajectory_execution(file_path=controllers_path)
    
    moveit_configs_builder.planning_pipelines(
        pipelines=["ompl"], default_planning_pipeline="ompl"
    )
    
    moveit_configs = moveit_configs_builder.to_moveit_configs()

    moveit_configs.planning_pipelines = planning_pipeline
    moveit_configs.planning_scene_monitor = planning_scene_monitor_parameters

    robot_description_joint_limits = {
        "robot_description_planning": load_yaml(
            denso_robot_moveit_config_pkg, path.join("config", "joint_limits.yaml")
        )
    }


    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_configs.to_dict(),
            robot_description_joint_limits,  
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_config_file = os.path.join(
        get_package_share_directory("isaacsim_moveit"),
        "rviz",
        "moveit.rviz",
    )

    _robot_description_kinematics_yaml = load_yaml(
        denso_robot_moveit_config_pkg, path.join("config", "kinematics.yaml")
    )
    robot_description_kinematics = {
        "robot_description_kinematics": _robot_description_kinematics_yaml
    }

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_configs.robot_description,
            moveit_configs.robot_description_semantic,
            moveit_configs.robot_description_kinematics,
            moveit_configs.planning_pipelines,
            robot_description_joint_limits,  
            robot_description_kinematics,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    world2robot_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher_world_to_robot",
        output="log",
        arguments=[
            "0.0",
            "-0.64",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "world",
            "panda_link0"],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_configs.robot_description,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory("denso_robot_bringup"),
        "config",
        "denso_robot_controllers.yaml",
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
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    pkg_name = 'object_manipulation'

    yaml_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'Poses.yaml'
    )

    welding = Node(
        package="object_manipulation",
        executable="welding",
        name="welding",
        output="screen",
        parameters=[
            moveit_configs.robot_description,
            moveit_configs.robot_description_semantic,
            moveit_configs.robot_description_kinematics,
            moveit_configs.planning_pipelines,
            robot_description_joint_limits,  
            moveit_configs.trajectory_execution,
            robot_description_kinematics,
            planning_pipeline,                          
            moveit_configs.planning_scene_monitor,
            {'yaml_file': yaml_file},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    denso_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["trajectory_controller", "-c", "/controller_manager"],
    )

    gui = os.path.join(
        get_package_share_directory("isaacsim_moveit"),
        "maps",
        "denso_welding.usd",
    )

    return LaunchDescription(
        [
        mode_arg, 
        isaac_sim_arg,
        gazebo_sim_arg,
        ros2_control_hardware_type,
        use_sim_time,
        #     rviz_node,
        #     world2robot_tf_node,
        #     robot_state_publisher,
        #     move_group_node,
        #     ros2_control_node,
        #     joint_state_broadcaster_spawner,
        #     denso_arm_controller,
            welding,

            # Node(
            #     package='object_manipulation',
            #     executable='synchronize_isaac_sim_labels',
            #     name='synchronize_isaac_sim_labels',
            #     output='screen',
            # ),

            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource([os.path.join(
            #                 get_package_share_directory("isaacsim"), "launch", "run_isaacsim.launch.py"
            #             ),
            #         ]
            #     ),
            #     launch_arguments={
            #         'version': '5.0.0',
            #         'play_sim_on_start': 'False',
            #         'gui': gui,
            #     }.items(),
            # ),
        ]
    )


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
