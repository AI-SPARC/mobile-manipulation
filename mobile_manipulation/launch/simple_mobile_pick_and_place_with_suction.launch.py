# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    # Command-line arguments
    ros2_control_hardware_type = DeclareLaunchArgument(
        "ros2_control_hardware_type",
        default_value="isaac",
        description="ROS2 control hardware interface type to use for the launch file -- possible values: [mock_components, isaac]",
    )

    # Declare use_sim_time argument
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
                "ros2_control_hardware_type": LaunchConfiguration(
                    "ros2_control_hardware_type"
                )
            },
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .to_moveit_configs()
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

    _robot_description_kinematics_yaml = load_yaml(
        "vai_se_ferrar_moveit_config", path.join("config", "kinematics.yaml")
    )
    robot_description_kinematics = {
        "robot_description_kinematics": _robot_description_kinematics_yaml
    }

    # RViz
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
            # robot_description_kinematics,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

   

    # Publish TF
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

    # ros2_control using FakeSystem as hardware
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
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    panda_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )


    robot_description_joint_limits = {
        "robot_description_planning": load_yaml(
            "vai_se_ferrar_moveit_config", path.join("config", "joint_limits.yaml")
        )
    }




    pkg_name = 'mobile_manipulation'

    yaml_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'pick_and_place_poses.yaml'
    )

    occupancy_grid_yaml = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'empty.yaml'
    )

  
    simple_manipulation_node = Node(
        package="mobile_manipulation",
        executable="simple_manipulation_node",
        name="simple_manipulation_node",
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
            {'yaml_file': yaml_file},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    a_star = Node(
        package="mobile_manipulation",
        executable="a_star",
        name="a_star",
        output="screen",
        parameters=[
            {'yaml_file': occupancy_grid_yaml},
            {'path_resolution': 0.05},
            {'security_distance': 0.45},
            {'iterations_before_verification': 20},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    controller = Node(
        package="mobile_manipulation",
        executable="controller",
        name="controller",
        output="screen",
        parameters=[
            {'yaml_file': occupancy_grid_yaml},
            {'path_resolution': 0.05},
            {'iterations_before_verification': 20},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    server_node = Node(
        package="mobile_manipulation",
        executable="server_node",
        name="server_node",
        output="screen",
        parameters=[
            {'yaml_file': yaml_file},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )


    labels_yaml_file = os.path.join(
        get_package_share_directory(pkg_name),
        'config',
        'labels.yaml'
    )

    add_collision = Node(
        package="mobile_manipulation",
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
            {'move_group': "panda_arm"},
        ],
        remappings=[
            ('/boxes_detection_array', '/bbox_3d_with_labels')
            
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    
   
    return LaunchDescription(
        [
            ros2_control_hardware_type,
            use_sim_time,  # Declare use_sim_time argument here
            # rviz_node,
            # world2robot_tf_node,
            # hand2camera_tf_node,
            robot_state_publisher,
            move_group_node,
            ros2_control_node,
            joint_state_broadcaster_spawner,
            panda_arm_controller_spawner,
            simple_manipulation_node,
            add_collision,
            server_node,
            a_star,
            controller,

            Node(
                package='isaacsim_moveit',
                executable='synchronize_isaac_sim_labels',
                name='synchronize_isaac_sim_labels',
                output='screen',
            ),
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