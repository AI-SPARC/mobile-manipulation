from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ros_gz_sim_pkg_path = get_package_share_directory('ros_gz_sim')
    simulation_pkg_path = FindPackageShare('gazebo')
    gz_launch_path = PathJoinSubstitution([ros_gz_sim_pkg_path, 'launch', 'gz_sim.launch.py'])

    # Caminho para o YAML que mapeia todos os tópicos do seu mundo
    bridge_yaml = PathJoinSubstitution([simulation_pkg_path, 'config', 'ros2_bridge_world_1.yaml'])

    return LaunchDescription([
        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH',
            PathJoinSubstitution([simulation_pkg_path, 'models'])
        ),
        SetEnvironmentVariable(
            'GZ_SIM_PLUGIN_PATH',
            PathJoinSubstitution([simulation_pkg_path, 'plugins'])
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={
                'gz_args': [PathJoinSubstitution([simulation_pkg_path, 'worlds/world_1.sdf'])],
                'on_exit_shutdown': 'True'
            }.items(),
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            output='screen',
            parameters=[{'config_file': bridge_yaml}]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='depth_camera1_to_map',
            arguments=['0.2', '0.5', '1.1', '-0.52', '0', '0', 'map', 'depth_optical_frame_1']
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='depth_camera2_to_map',
            arguments=['0.2', '-0.5', '1.1', '0.52', '0', '0', 'map', 'depth_optical_frame_2']
        )

    ])
