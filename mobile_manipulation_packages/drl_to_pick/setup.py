from setuptools import find_packages, setup

package_name = 'drl_to_pick'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='momesso',
    maintainer_email='momessoalves@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_model = drl_to_pick.train_model:main',
            'train_model_pcl = drl_to_pick.train_model_pcl:main'
        ],
    },
)
