import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('panel_detection')
    default_config = os.path.join(pkg_dir, 'config', 'panel_detection.yaml')

    config_arg = DeclareLaunchArgument(
        'config_path',
        default_value=default_config,
        description='Path to panel_detection.yaml config file'
    )

    node = Node(
        package='panel_detection',
        executable='panel_detect_node',
        name='panel_detection_node',
        output='screen',
        parameters=[{
            'config_path': LaunchConfiguration('config_path'),
        }],
    )

    return LaunchDescription([
        config_arg,
        node,
    ])
