from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    num_robots = 1
    num_cameras = 1 # Altere para a quantidade de câmeras que precisar
    
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

    # Remapeamentos do GTSAM ajustados para a raiz (sem o namespace do robô)
    gtsam_remappings = [
        ('/tf', '/tf'),
        ('/tf_static', '/tf_static'),
        ('slam/camera_factors_0', '/slam/camera_factors'),
        ('loop_closure/dino_image_0', '/loop_closure/dino_image'),
        ('loop_closure/depth_image_0', '/loop_closure/depth_image'),
        ('ground_truth_0', '/ground_truth'),
        ('odom_0', '/odom'),
        ('gtsam_graph_0', '/gtsam_graph'),
        ('gtsam_path_0', '/gtsam_path'),
        ('camera_info_0', '/camera_0/depth/camera_info')
    ]

    gtsam_node = Node(
        package='slam_core',     
        executable='gtsam_and_loop_closure',  
        name='gtsam_and_loop_closure',     
        output='screen',                 
        parameters=[slam_parameters],
        remappings=gtsam_remappings 
    )
    nodes_to_start.append(gtsam_node)

    # Remapeamentos do slam_core (sem namespace)
    slam_core_remappings = [
        ('/scan', '/front_2d_lidar/scan'),
        ('/tf', '/tf'),
        ('/tf_static', '/tf_static')
    ]

    current_slam_params = {**slam_parameters, 'num_cameras': num_cameras}

    slam_core_node = Node(
        package='slam_core',     
        executable='slam_core',  
        name='slam_core',                   
        output='screen',                 
        parameters=[current_slam_params],
        remappings=slam_core_remappings 
    )
    nodes_to_start.append(slam_core_node)


    ground_truth = Node(
        package='slam_core',     
        executable='ground_truth',  
        name='ground_truth',                   
        output='screen',                 
        parameters=[current_slam_params],
    )
    nodes_to_start.append(ground_truth)

    

    # Laço dinâmico para múltiplas câmeras, sem namespace de robô
    for c in range(num_cameras):
        add_noise_remappings = [
            ('image_in',        f'/camera_{c}/depth/image_perfect'),              
            ('camera_info_in',  f'/camera_{c}/depth/camera_info_perfect'),           
            ('image_out',       f'/camera_{c}/depth/image_rect_raw'),     
            ('camera_info_out', f'/camera_{c}/depth/camera_info'),   
            ('noisy_cloud',     f'/camera_{c}/depth/noisy_cloud'),
            ('/tf',             '/tf'),
            ('/tf_static',      '/tf_static')
        ]

        add_noise_node = Node(
            package='slam_core',     
            executable='add_noise',  
            name=f'add_noise_c{c}',        
            output='screen',                 
            parameters=[add_noise_parameters],
            remappings=add_noise_remappings 
        )
        nodes_to_start.append(add_noise_node)

    return LaunchDescription(nodes_to_start)