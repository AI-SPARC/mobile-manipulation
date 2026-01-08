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
        "simulation_base_speed": 1.0,
        "min_robot_gap": 2.5,
        "robot_radius": 0.3,
        "time_gap_tolerance": 1.0,
        "animation_rate_ms": 20,
        "min_robot_count": 10,
        "max_robot_count": 50,
        "map_limit_x": 25.0, 
        "map_limit_y": 25.0
    }]

      
    return LaunchDescription([

        

        Node(
            package='fleet_coordination',
            executable='fleet_management',
            output='screen',
            parameters=parameters,
        ),

        Node(
            package='fleet_coordination',
            executable='generate_tests',
            output='screen',
            parameters=parameters,
        )



    ])