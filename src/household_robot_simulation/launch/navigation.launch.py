import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('household_robot_simulation')
    
    # Default Paths
    default_map_path = os.path.join(pkg_share, 'maps', 'house_map.yaml')
    default_params_file = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    
    # Launch Configurations
    map_yaml_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    
    # Declare Launch Arguments
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=default_map_path,
        description='Full path to map yaml file to load'
    )
    
    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS 2 parameters file to use'
    )
    
    # Map Server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml_file}, {'use_sim_time': True}]
    )
    
    # AMCL Localization
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}]
    )
    
    # Planner Server (Global Path)
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}]
    )
    
    # Controller Server (Local path tracking)
    # Remapped to cmd_vel_raw to pass through our safety watchdog!
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}],
        remappings=[('/cmd_vel', '/cmd_vel_raw')]
    )
    
    # Behavior Server (Recovery actions like spin/backup)
    # Remapped to cmd_vel_raw to pass through our safety watchdog!
    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}],
        remappings=[('/cmd_vel', '/cmd_vel_raw')]
    )
    
    # BT Navigator
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file, {'use_sim_time': True}]
    )
    
    # Lifecycle Manager for Localization (starts first)
    lifecycle_manager_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[params_file, {
            'use_sim_time': True,
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }]
    )
    
    # Lifecycle Manager for Navigation (delayed 8s to let localization publish the map frame)
    lifecycle_manager_navigation = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                output='screen',
                parameters=[params_file, {
                    'use_sim_time': True,
                    'autostart': True,
                    'node_names': ['planner_server', 'controller_server', 'behavior_server', 'bt_navigator']
                }]
            )
        ]
    )
    
    return LaunchDescription([
        declare_map_yaml_cmd,
        declare_params_file_cmd,
        map_server,
        amcl,
        planner_server,
        controller_server,
        behavior_server,
        bt_navigator,
        lifecycle_manager_localization,
        lifecycle_manager_navigation
    ])
