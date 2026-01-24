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
        "pcd_path": "/home/momesso/toma4.pcd",
        "gripper_mesh_path": "/home/momesso/hand_and_finger.stl",
        "gripper_mesh_scale": 1.0,
        "grid_res": 0.005,         
        "cloud_voxel_size": 0.001,
        
        "cylinder_radius": 0.01,  
        "cylinder_height": 0.01,
        "analysis_step_size": 0.01,
        "mls_radius": 0.0045,
        "use_mls_smoothing": True,
        "num_best_grasps": 1,
        "finger_offset": 0.01,
        "min_points_per_segment": 2, 
        "weight_orientation": 0.5, 
        "weight_symmetry": 0.5,
        "weight_planarity": 0.0,
        "max_gripper_width": 0.07,
        "gripper_finger_depth": 0.08,
        "gripper_collision_threshold": 5,
        "gripper_structure_thickness": 0.005,
        "num_collision_checks": 1
    }]

  
      
    return LaunchDescription([

        

        Node(
            package='vision',
            executable='generate_grasp_poses',
            output='screen',
            parameters=parameters,
        ),

      

    ])