from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # ========================================================================
    # 1. CONFIGURAÇÕES GERAIS E DICIONÁRIOS
    # ========================================================================
    num_robots = 1
    
    # Dicionário mapeando: 'nome_do_namespace' -> numero_de_cameras
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

    # Parâmetros base para os nós de SLAM
    slam_parameters = {
        'use_sim_time': True,
        'main_frame_id': 'base_link', 
        'use_imu': True,
        'num_robots': num_robots,
        'use_ground_truth': True     
    }

    nodes_to_start = []

    # ========================================================================
    # 2. NÓ CENTRAL: GTSAM & LOOP CLOSURE
    # ========================================================================
    # Este nó é global. Mapeamos os tópicos indexados (_0, _1) para os namespaces reais.
    gtsam_remappings = [
        ('/tf', '/robot_0/tf'),
        ('/tf_static', '/robot_0/tf_static')
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


    # ========================================================================
    # 3. NÓS ESPECÍFICOS POR ROBÔ (SLAM CORE E CÂMERAS)
    # ========================================================================
    for robot_ns, num_cameras in robots_config.items():
        
        # --------------------------------------------------------------------
        # 3.1 NÓ: SLAM CORE (Um por robô)
        # --------------------------------------------------------------------
        slam_core_remappings = [
            # Tópicos absolutos (começam com /) que vêm do Isaac Sim
            ('/scan', f'/{robot_ns}/front_2d_lidar/scan'),
            ('/tf', f'/{robot_ns}/tf'),
            ('/tf_static', f'/{robot_ns}/tf_static')
            
            # Os tópicos relativos (como slam/camera_factors) não precisam mais 
            # de remapping aqui, pois o argumento 'namespace' abaixo cuida disso!
        ]

        current_slam_params = {**slam_parameters, 'num_cameras': num_cameras}

        slam_core_node = Node(
            package='slam_core',     
            executable='slam_core',  
            name='slam_core',     
            namespace=robot_ns,              # <--- Isola o nó e seus tópicos automaticamente
            output='screen',                 
            parameters=[current_slam_params],
            remappings=slam_core_remappings 
        )
        nodes_to_start.append(slam_core_node)

        # --------------------------------------------------------------------
        # 3.2 NÓS: ADD NOISE (Um por câmera, por robô)
        # --------------------------------------------------------------------
        for c in range(num_cameras):
            add_noise_remappings = [
                # Tópicos relativos: não precisam da barra '/' no começo.
                # O ROS 2 vai juntar o namespace do robô com isso aqui.
                ('image_in',        f'camera_{c}/depth/image_perfect'),              
                ('camera_info_in',  f'camera_{c}/depth/camera_info_perfect'),           
                ('image_out',       f'camera_{c}/depth/image_rect_raw'),     
                ('camera_info_out', f'camera_{c}/depth/camera_info'),   
                ('noisy_cloud',     f'camera_{c}/depth/noisy_cloud'),
                
                # Tópicos absolutos do sistema
                ('/tf',        f'/{robot_ns}/tf'),
                ('/tf_static', f'/{robot_ns}/tf_static')
            ]

            add_noise_node = Node(
                package='slam_core',     
                executable='add_noise',  
                name=f'add_noise_c{c}', 
                namespace=robot_ns,          # <--- Isola o nó e seus tópicos automaticamente
                output='screen',                 
                parameters=[add_noise_parameters],
                remappings=add_noise_remappings 
            )
            nodes_to_start.append(add_noise_node)

    # ========================================================================
    # 4. RETORNA A LISTA FINAL
    # ========================================================================
    return LaunchDescription(nodes_to_start)