from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
   
    parameters = {
        'use_sim_time': True,
        'main_frame_id': 'camera_link'
    }

   
    remappings = [
        ('/scan', '/front_2d_lidar/scan'),
        ('/camera/rgb/image_raw', '/camera/camera/color/image_raw'),
        ('/camera/depth/image_rect_raw', '/camera/camera/depth/image_rect_raw'),
        ('/camera/depth/camera_info', '/camera/camera/color/camera_info'),
    ]

    slam_core = Node(
        package='slam_core',     
        executable='slam_core',  
        name='slam_core',     
        output='screen',                 
        parameters=[parameters],
        remappings=remappings 
    )

   
    return LaunchDescription([
        slam_core
    ])