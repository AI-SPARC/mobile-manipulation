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
        "use_pcd_file": True,
        "pcd_path": "/home/momesso/pcds/toma_9.pcd",
        "gripper_mesh_path": "/home/momesso/hand_and_fingers.obj",
        "gripper_mesh_scale": 1.0,
        "grid_res": 0.0075,         
        "cloud_voxel_size": 0.001,

        "mesh_offset_x": 0.025,
        "mesh_offset_y": 0.0,
        "mesh_offset_z": 0.0,
        "num_benchmark_runs": 100,
        "enable_ray_animation": False,
        "step_by_step": False,
        "animation_delay_ms": 500,
        "mesh_rot_roll": 1.57,
        "mesh_rot_pitch": 0.0,
        "mesh_rot_yaw": 1.57, 
        "rotation_step_deg": 55.0,
        "cylinder_radius": 0.005,  
        "cylinder_height": 0.01,
        "analysis_step_size": 0.01,
        "mean_filter_k": 15,
        "use_mean_filter": True,
        "num_best_grasps": 1,
        "finger_offset": 0.027,
        "min_points_per_segment": 2, 
        "weight_orientation": 0.5, 
        "weight_symmetry": 0.5,
        "weight_planarity": 0.0,
        "max_gripper_width": 0.07,
        "gripper_finger_depth": 0.08,
        "gripper_collision_threshold": 5,
        "gripper_structure_thickness": 0.005,
        "num_collision_checks": 1,
        "num_random_orientations": 1
    }]

  
      
    return LaunchDescription([

        

        Node(
            package='vision',
            executable='generate_grasp_poses_main',
            output='screen',
            parameters=parameters,
        ),

      

    ])