import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():
    # =========================================================================
    # 1. DEFINIÇÃO DE CAMINHOS (Baseado no seu pedido)
    # =========================================================================
    
    # Nome do seu pacote (onde estão os arquivos modificados)
    MY_PACKAGE_NAME = 'gazebo'
    
    try:
        pkg_my_package = get_package_share_directory(MY_PACKAGE_NAME)
    except Exception:
        raise Exception(f"Pacote '{MY_PACKAGE_NAME}' não encontrado. Verifique se compilou e deu source.")

    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_turtlebot3_description = get_package_share_directory('turtlebot3_description')

    # Caminho do SDF modificado (Física e Sensores)
    # Local: gazebo/models/turtlebot3_waffle/model.sdf
    sdf_path_default = os.path.join(pkg_my_package, 'models', 'turtlebot3_waffle', 'model.sdf')

    # Caminho do Mundo modificado
    # Local: gazebo/worlds/turtlebot3_house.world
    world_path_default = os.path.join(pkg_my_package, 'worlds', 'turtlebot3_house.world')

    # Caminho do URDF oficial (Árvore de TF e Estrutura para o Robot State Publisher)
    urdf_path = os.path.join(pkg_turtlebot3_description, 'urdf', 'turtlebot3_waffle.urdf')

    # =========================================================================
    # 2. CONFIGURAÇÃO DE AMBIENTE (MESHES)
    # =========================================================================
    # Adicionamos a pasta 'models' do seu pacote ao path do Gazebo para ele achar as texturas
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[
            os.path.join(os.environ.get('AMENT_PREFIX_PATH', ''), 'share'),
            ':',
            os.path.join(pkg_my_package, 'models'), # Para achar meshes locais
            ':',
            os.environ.get('GZ_SIM_RESOURCE_PATH', '')
        ]
    )

    # =========================================================================
    # 3. NÓS E PROCESSOS
    # =========================================================================

    # A. Robot State Publisher (Gera o TF: odom -> base_link -> camera_link)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_path]),
            'use_sim_time': True
        }]
    )

    # B. Gazebo Simulator
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        # Carrega o mundo definido na variável world_path_default
        launch_arguments={'gz_args': ['-r ', LaunchConfiguration('world')]}.items(),
    )

    # C. Spawnar o Robô
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'turtlebot3_waffle',
            '-file', LaunchConfiguration('sdf_path'),
            '-x', '-2.0',
            '-y', '-0.5',
            '-z', '0.01'
        ],
        output='screen'
    )

    # D. Bridge (Ponte ROS 2 <-> Gazebo Harmonic)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'use_sim_time': True}],
        arguments=[
            # --- NAVEGAÇÃO & CONTROLE ---
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            
            # CRUCIAL: Joint States para o Robot State Publisher saber onde estão as rodas
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            
            # --- CÂMERA & NUVEM DE PONTOS (RTAB-MAP) ---
            # 1. Imagem RGB
            '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            # 2. Imagem Profundidade
            '/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
            # 3. Nuvem de Pontos (PointCloud2)
            '/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
            # 4. Info da Câmera
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'
        ],
        # Remapeamentos para padronizar com o esperado pelo RTAB-Map ou Nav2
        remappings=[
            ('/camera/image', '/camera/rgb/image_raw'),
            ('/camera/depth_image', '/camera/depth/image_rect_raw'),
            ('/camera/points', '/camera/depth/points'),
            ('/camera/camera_info', '/camera/depth/camera_info')
        ],
        output='screen'
    )

    tf_fix = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = [
            '--x', '0', '--y', '0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'odom', # Pai (Do URDF)
            '--child-frame-id', 'turtlebot3_waffle/camera_rgb_frame/intel_realsense_r200' # Filho (Do Gazebo)
        ],
        output='screen'
    )


    return LaunchDescription([
        gz_resource_path,
        DeclareLaunchArgument('world', default_value=world_path_default, description='Caminho do Mundo'),
        DeclareLaunchArgument('sdf_path', default_value=sdf_path_default, description='Caminho do Robô'),
        robot_state_publisher,
        gz_sim,
        spawn_robot,
        bridge,
        tf_fix
    ])