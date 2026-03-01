from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    
    num_cameras = 1

    slam_parameters = {
        'use_sim_time': True,
        'main_frame_id': 'base_link',
        'use_imu': True,
        'num_cameras': num_cameras, 
    }

    add_noise_parameters = {
        'use_sim_time': True,
        'baseline': 0.05,
        'subpixel_error': 0.08,
        'publish_pointcloud': True
    }


    nodes_to_start = []

    
    for i in range(num_cameras):
        camera_ns = f'/camera_{i}'
        
        add_noise_remappings = [
            ('image_in', f'{camera_ns}/depth/image_perfect'),              
            ('camera_info_in', f'{camera_ns}/depth/camera_info_perfect'),           
            ('image_out', f'{camera_ns}/depth/image_rect_raw'),     
            ('camera_info_out', f'{camera_ns}/depth/camera_info'),   
            ('noisy_cloud', f'{camera_ns}/depth/noisy_cloud')               
        ]

        add_noise_node = Node(
            package='slam_core',     
            executable='add_noise',  
            name=f'add_noise_{i}', 
            output='screen',                 
            parameters=[add_noise_parameters],
            remappings=add_noise_remappings 
        )
        
       
        nodes_to_start.append(add_noise_node)


  
    slam_core_remappings = [
        ('/scan', '/front_2d_lidar/scan'),
        ('/camera/rgb/image_raw', '/camera/rgb/image_raw'),             
        ('/camera/depth/image_rect_raw', '/camera/depth/image_rect_raw'),
        ('/camera/depth/camera_info', '/camera/depth/camera_info'),
        ('/ground_truth', '/ground_truth') 
    ]

    slam_core_node = Node(
        package='slam_core',     
        executable='slam_core',  
        name='slam_core',     
        output='screen',                 
        parameters=[slam_parameters],
        remappings=slam_core_remappings 
    )

    nodes_to_start.append(slam_core_node)

    return LaunchDescription(nodes_to_start)