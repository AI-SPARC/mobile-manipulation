import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction # <--- Adicione TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    home = os.path.expanduser("~")
    
    server_script = os.path.join(
        home, 
        "pibic/src/mobile_manipulation_packages/slam/grounding_dino_ros2/grounding_dino_ros2/dino_server.py"
    )

    # 1. Define o processo do Servidor (Começa imediatamente)
    dino_server_process = ExecuteProcess(
        cmd=['conda', 'run', '-n', 'gdino', '--no-capture-output', 'python', server_script],
        output='screen'
    )

    # 2. Define o Nó do Cliente (Mas não inicia ainda)
    dino_client_node = Node(
        package='grounding_dino_ros2',
        executable='dino_node',
        name='grounding_dino_client',
        output='screen',
        parameters=[{'prompt': 'chair . cone .'}]
    )

    return LaunchDescription([
        dino_server_process,
        
        # 3. Espera 15 segundos antes de lançar o cliente
        TimerAction(
            period=15.0, 
            actions=[dino_client_node]
        )
    ])