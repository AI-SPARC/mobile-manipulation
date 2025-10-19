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

    
    gui = os.path.join(
        get_package_share_directory("isaacsim_moveit"),
        "maps",
        "denso_welding_with_trajectory.usd",
    )

    return LaunchDescription(
        [
           
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                            get_package_share_directory("isaacsim"), "launch", "run_isaacsim.launch.py"
                        ),
                    ]
                ),
                launch_arguments={
                    'version': '5.0.0',
                    'play_sim_on_start': 'False',
                    'gui': gui,
                }.items(),
            ),
        ]
    )
