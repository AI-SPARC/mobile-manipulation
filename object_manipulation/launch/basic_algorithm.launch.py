#!/usr/bin/env python3
"""Launch basic_algorithm with full MoveIt configuration."""

from os import path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml

def load_yaml(package_name, file_path):
    from ament_index_python.packages import get_package_share_directory
    package_path = get_package_share_directory(package_name)
    abs_file_path = path.join(package_path, file_path)
    try:
        with open(abs_file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def generate_launch_description():

    # Declare launch arguments
    declared_arguments = [
        DeclareLaunchArgument("description_package", default_value="panda_description"),
        DeclareLaunchArgument("description_filepath", default_value=path.join("urdf", "panda.urdf.xacro")),
        DeclareLaunchArgument("moveit_config_package", default_value="panda_moveit_config"),
        DeclareLaunchArgument("name", default_value="panda"),
        DeclareLaunchArgument("prefix", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("log_level", default_value="info"),
        DeclareLaunchArgument("node_name", default_value="basic_algorithm"),  # nome dinâmico
    ]

    # Launch configurations
    description_package = LaunchConfiguration("description_package")
    description_filepath = LaunchConfiguration("description_filepath")
    moveit_config_package = LaunchConfiguration("moveit_config_package")
    name = LaunchConfiguration("name")
    prefix = LaunchConfiguration("prefix")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")
    node_name = LaunchConfiguration("node_name")  # nova configuração

    # robot_description (URDF)
    robot_description = {"robot_description": Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([FindPackageShare(description_package), description_filepath]), " ",
        "name:=", name
    ])}

    # robot_description_semantic (SRDF)
    robot_description_semantic = {"robot_description_semantic": Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([FindPackageShare(moveit_config_package), "srdf", "panda.srdf.xacro"]), " ",
        "name:=", name
    ])}

    # Kinematics
    kinematics_yaml = load_yaml("panda_moveit_config", path.join("config", "kinematics.yaml"))
    robot_description_kinematics = {"robot_description_kinematics": kinematics_yaml}

    # Joint limits
    joint_limits_yaml = load_yaml("panda_moveit_config", path.join("config", "joint_limits.yaml"))
    robot_description_planning = {"robot_description_planning": joint_limits_yaml}

    # OMPL planning
    ompl_yaml = load_yaml("panda_moveit_config", path.join("config", "ompl_planning.yaml"))
    planning_pipeline = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": ompl_yaml
    }

    # MoveIt controllers
    controller_yaml = load_yaml("panda_moveit_config", path.join("config", "moveit_controller_manager.yaml"))
    moveit_controller_manager = {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": controller_yaml
    }

    # Node basic_algorithm com nome dinâmico
    basic_algorithm_node = Node(
        package="object_manipulation",
        executable="basic_algorithm",
        name=node_name,  # nome vem do LaunchArgument
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            planning_pipeline,
            moveit_controller_manager,
            {"use_sim_time": use_sim_time}
        ],
        arguments=["--ros-args", "--log-level", log_level]
    )

    rounded_points = Node(
        package="drl",
        executable="rounded_points",
        name=node_name,  # nome vem do LaunchArgument
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time}
        ],
        arguments=["--ros-args", "--log-level", log_level]
    )

    return LaunchDescription(declared_arguments + [basic_algorithm_node] + [rounded_points])
