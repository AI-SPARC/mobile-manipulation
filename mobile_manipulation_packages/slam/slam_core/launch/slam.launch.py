from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
   
    num_robots = 1
    
   
    robots_config = {
        'robot_0': 1,  
        # 'robot_1': 1,   
    }

    add_noise_parameters = {
        'use_sim_time': True,
        'baseline': 0.05,
        'subpixel_error': 0.08,
        'publish_pointcloud': True
    }

    
    slam_parameters = {
        'use_sim_time': True,
        'main_frame_id': 'base_link', 
        'use_imu': True,
        'num_robots': num_robots,
        'use_ground_truth': True     
    }

    nodes_to_start = []

   
    gtsam_remappings = [
        ('/tf', '/tf'),
        ('/tf_static', '/tf_static')
    ]

    for i, (robot_ns, num_cameras) in enumerate(robots_config.items()):
        gtsam_remappings.extend([
            (f'slam/camera_factors_{i}',     f'/{robot_ns}/slam/camera_factors'),
            (f'loop_closure/dino_image_{i}', f'/{robot_ns}/loop_closure/dino_image'),
            (f'loop_closure/depth_image_{i}',f'/{robot_ns}/loop_closure/depth_image'),
            (f'ground_truth_{i}',            f'/{robot_ns}/ground_truth'),
            (f'odom_{i}',                    f'/{robot_ns}/odom'),
            (f'gtsam_graph_{i}',             f'/{robot_ns}/gtsam_graph'),
            (f'gtsam_path_{i}',              f'/{robot_ns}/gtsam_path'),
            (f'camera_info_{i}',             f'/{robot_ns}/camera_0/depth/camera_info')
        ])

    gtsam_node = Node(
        package='slam_core',     
        executable='gtsam_and_loop_closure',  
        name='gtsam_and_loop_closure',     
        output='screen',                 
        parameters=[slam_parameters],
        remappings=gtsam_remappings 
    )
    nodes_to_start.append(gtsam_node)


   
    for robot_ns, num_cameras in robots_config.items():
        
        
        slam_core_remappings = [
            
            ('/scan', f'/{robot_ns}/front_2d_lidar/scan'),
            ('/tf', f'/{robot_ns}/tf'),
            ('/tf_static', f'/{robot_ns}/tf_static')
            
           
        ]

        current_slam_params = {**slam_parameters, 'num_cameras': num_cameras}

        slam_core_node = Node(
            package='slam_core',     
            executable='slam_core',  
            name='slam_core',     
            namespace=robot_ns,              
            output='screen',                 
            parameters=[current_slam_params],
            remappings=slam_core_remappings 
        )
        nodes_to_start.append(slam_core_node)

       
        for c in range(num_cameras):
            add_noise_remappings = [
                
                ('image_in',        f'camera_{c}/depth/image_perfect'),              
                ('camera_info_in',  f'camera_{c}/depth/camera_info_perfect'),           
                ('image_out',       f'camera_{c}/depth/image_rect_raw'),     
                ('camera_info_out', f'camera_{c}/depth/camera_info'),   
                ('noisy_cloud',     f'camera_{c}/depth/noisy_cloud'),
                
                
                ('/tf',        f'/{robot_ns}/tf'),
                ('/tf_static', f'/{robot_ns}/tf_static')
            ]

            add_noise_node = Node(
                package='slam_core',     
                executable='add_noise',  
                name=f'add_noise_c{c}', 
                namespace=robot_ns,          
                output='screen',                 
                parameters=[add_noise_parameters],
                remappings=add_noise_remappings 
            )
            nodes_to_start.append(add_noise_node)

  
    return LaunchDescription(nodes_to_start)