from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    num_robots = 1
    
    robots_config = {
        'robot_0': 1,  
        #  'robot_1': 1,   
    }

    add_noise_parameters = {
        'use_sim_time': True,
        'baseline': 0.05,
        'subpixel_error': 0.08,
        'publish_pointcloud': True
    }

    nodes_to_start = []

   
    for robot_ns, num_cameras in robots_config.items():
        
      
        for c in range(num_cameras):
            camera_ns = f'/{robot_ns}/camera_{c}' 
            
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
                name=f'add_noise_{robot_ns}_c{c}', 
                output='screen',                 
                parameters=[add_noise_parameters],
                remappings=add_noise_remappings 
            )
            
            nodes_to_start.append(add_noise_node)


       
        slam_parameters = {
            'use_sim_time': True,
            'main_frame_id': 'base_link', 
            'use_imu': True,
            'num_robots': num_robots,
            'num_cameras': num_cameras, 
            'robot_namespace': robot_ns, 
            'use_ground_truth': True     
        }

   
        slam_core_remappings = [
            ('/scan', f'/{robot_ns}/front_2d_lidar/scan'),
            ('/tf', f'/{robot_ns}/tf'),
            ('/tf_static', f'/{robot_ns}/tf_static')
        ]

        slam_core_node = Node(
            package='slam_core',     
            executable='slam_core',  
            name=f'slam_core_{robot_ns}',     
            output='screen',                 
            parameters=[slam_parameters],
            remappings=slam_core_remappings 
        )

        nodes_to_start.append(slam_core_node)

    gtsam_node = Node(
        package='slam_core',     
        executable='gtsam_and_loop_closure',  
        name='gtsam_and_loop_closure',     
        output='screen',                 
        parameters=[slam_parameters],
        remappings=slam_core_remappings 
    )

    nodes_to_start.append(gtsam_node)

    return LaunchDescription(nodes_to_start)