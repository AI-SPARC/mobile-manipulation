import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    
    parameters = [{
        "path_resolution": 0.05,
        "simulation_base_speed": 2.0,
        "min_robot_gap": 2.0,
        "robot_radius": 0.3,
        "time_gap_tolerance": 2.0,
        "animation_rate_ms": 20
    }]

      
    return LaunchDescription([

        

        Node(
            package='fleet_coordination',
            executable='adjust_velocities_fleet_management',
            output='screen',
            parameters=parameters,
        ),

        



    ])