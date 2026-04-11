from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    num_robots = 1

    robots_config = {
        'robot_0': 1,
    }

    slam_parameters = {
        'use_sim_time': True,
        'main_frame_id': 'camera_link',
        'use_imu': True,
        'num_robots': num_robots,
        'use_ground_truth': False,
    }

    # ── GTSAM remappings ──────────────────────────────────────────────────────
    gtsam_remappings = [
        ('/tf',        '/tf'),
        ('/tf_static', '/tf_static'),
    ]

    for i, (robot_ns, _) in enumerate(robots_config.items()):
        gtsam_remappings.extend([
            (f'slam/camera_factors_{i}',      f'/{robot_ns}/slam/camera_factors'),
            (f'loop_closure/dino_image_{i}',  f'/{robot_ns}/loop_closure/dino_image'),
            (f'loop_closure/depth_image_{i}', f'/{robot_ns}/loop_closure/depth_image'),
            (f'ground_truth_{i}',             f'/{robot_ns}/ground_truth'),
            (f'odom_{i}',                     f'/{robot_ns}/odom'),
            (f'gtsam_graph_{i}',              f'/{robot_ns}/gtsam_graph'),
            (f'gtsam_path_{i}',               f'/{robot_ns}/gtsam_path'),
            # camera_info → aligned_depth_to_color (mesma frame que a depth alinhada)
            (f'camera_info_{i}',
             '/camera/camera/aligned_depth_to_color/camera_info'),
        ])

    gtsam_node = Node(
        package='slam_core',
        executable='gtsam_and_loop_closure',
        name='gtsam_and_loop_closure',
        output='screen',
        parameters=[slam_parameters],
        remappings=gtsam_remappings,
    )

    # ── slam_core nodes (one per robot) ───────────────────────────────────────
    nodes_to_start = [gtsam_node]

    for robot_ns, num_cameras in robots_config.items():
        slam_core_remappings = [
            ('/tf',        '/tf'),
            ('/tf_static', '/tf_static'),

            # RGB color
            (f'/{robot_ns}/camera_0/rgb/image_raw',
             '/camera/camera/color/image_raw'),

            # Depth alinhado ao color (evita distorção de projeção)
            (f'/{robot_ns}/camera_0/depth/image_rect_raw',
             '/camera/camera/aligned_depth_to_color/image_raw'),

            # Camera info do aligned_depth_to_color
            # (intrínseca é a mesma do color quando alinhado)
            (f'/{robot_ns}/camera_0/depth/camera_info',
             '/camera/camera/aligned_depth_to_color/camera_info'),
        ]

        current_slam_params = {**slam_parameters, 'num_cameras': num_cameras}

        slam_core_node = Node(
            package='slam_core',
            executable='slam_core',
            name='slam_core',
            namespace=robot_ns,
            output='screen',
            parameters=[current_slam_params],
            remappings=slam_core_remappings,
        )
        nodes_to_start.append(slam_core_node)

    return LaunchDescription(nodes_to_start)