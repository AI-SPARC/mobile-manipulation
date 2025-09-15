from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    
    
    return LaunchDescription([
       

        Node(
            package='drl',
            executable='rounded_points',
            name='rounded_points',
            output='screen',
        ),

        Node(
            package='drl',
            executable='train_model',
            name='train_model',
            output='screen',
        ),




    ])
