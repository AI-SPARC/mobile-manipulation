from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Parâmetros separados para cada nó
    slam_parameters = {
        'use_sim_time': True,
        'main_frame_id': 'base_link'
    }

    add_noise_parameters = {
        'use_sim_time': True,
        'baseline': 0.05,
        'subpixel_error': 0.08,
        'publish_pointcloud': True
    }


    add_noise_remappings = [
        ('image_in', '/camera/depth/image_perfect'),              
        ('camera_info_in', '/camera/depth/camera_info_perfect'),           
        ('image_out', '/camera/depth/image_rect_raw'),     
        ('camera_info_out', '/camera/depth/camera_info'),   
        ('noisy_cloud', '/camera/depth/noisy_cloud')               
    ]

   
    slam_core_remappings = [
        ('/scan', '/front_2d_lidar/scan'),
        ('/camera/rgb/image_raw', '/camera/rgb/image_raw'),             
        ('/camera/depth/image_rect_raw', '/camera/depth/image_rect_raw'),
        ('/camera/depth/camera_info', '/camera/depth/camera_info'),
        ('/ground_truth', '/ground_truth') 
    ]

    # Inicialização dos Nós
    add_noise_node = Node(
        package='slam_core',     
        executable='add_noise',  
        name='add_noise',     
        output='screen',                 
        parameters=[add_noise_parameters],
        remappings=add_noise_remappings 
    )

    slam_core_node = Node(
        package='slam_core',     
        executable='slam_core',  
        name='slam_core',     
        output='screen',                 
        parameters=[slam_parameters],
        remappings=slam_core_remappings 
    )

    return LaunchDescription([
        add_noise_node,
        slam_core_node
    ])