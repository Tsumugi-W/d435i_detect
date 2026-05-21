from setuptools import setup, find_packages
from glob import glob

package_name = 'panel_detection'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/scripts', glob('scripts/*')),
        ('share/' + package_name + '/weights', glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zile',
    maintainer_email='zile@todo.todo',
    description='Panel pose detection with YOLOv5 + depth camera (self-contained ROS2 package)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'panel_detect_node = panel_detection.node_panel_detect:main',
        ],
    },
)
