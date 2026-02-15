import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'grounding_dino_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- ADICIONE ESTA LINHA ---
        # Ela copia todos os arquivos da pasta 'launch' para o local de instalação
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='momesso',
    maintainer_email='momessoalves@gmail.com',
    description='Pacote para integrar GroundingDINO ao ROS 2 via Socket',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dino_node = grounding_dino_ros2.dino_node:main'
        ],
    },
)