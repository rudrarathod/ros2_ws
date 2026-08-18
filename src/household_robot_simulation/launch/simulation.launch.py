import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get package directories
    pkg_share = get_package_share_directory('household_robot_simulation')
    gz_sim_share = get_package_share_directory('ros_gz_sim')
    
    # Paths
    world_file = os.path.join(pkg_share, 'worlds', 'house.sdf')
    urdf_file = os.path.join(pkg_share, 'urdf', 'custom_robot.urdf')
    
    # Declare launch argument for the world file path
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_file,
        description='Path to the SDF world file to load'
    )
    
    # Include Gazebo Sim launch file
    # We append ' -r' to run the simulation automatically on startup
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': [LaunchConfiguration('world'), ' -r']
        }.items()
    )
    
    # Read URDF file contents
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
        
    # Robot State Publisher Node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}]
    )
    
    # Gazebo Sim Spawn Node (spawns the robot from /robot_description topic)
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-name', 'custom_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.08'
        ]
    )
    
    # ROS 2 to Gazebo Sim Bridge Config file path
    bridge_config = os.path.join(pkg_share, 'config', 'bridge_config.yaml')
    
    # ROS 2 to Gazebo Sim Bridge Node using YAML configuration (enables frame_id override)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{'config_file': bridge_config, 'use_sim_time': True}]
    )
    
    # Path to RViz configuration
    rviz_config = os.path.join(pkg_share, 'rviz', 'robot.rviz')
    
    # RViz Node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}]
    )
    
    # Static TF Publisher from lidar_link to Gazebo's lumped sensor frame
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0.04',
            '--yaw', '0',
            '--pitch', '0',
            '--roll', '0',
            '--frame-id', 'lidar_link',
            '--child-frame-id', 'custom_robot/base_footprint/gpu_lidar'
        ],
        parameters=[{'use_sim_time': True}]
    )
    
    # Static TF Publisher from camera_link to Gazebo's lumped sensor frame
    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--yaw', '0',
            '--pitch', '0',
            '--roll', '0',
            '--frame-id', 'camera_link',
            '--child-frame-id', 'custom_robot/base_footprint/camera'
        ],
        parameters=[{'use_sim_time': True}]
    )
    
    # Static TF Publisher from imu_link to Gazebo's lumped sensor frame
    static_tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--yaw', '0',
            '--pitch', '0',
            '--roll', '0',
            '--frame-id', 'imu_link',
            '--child-frame-id', 'custom_robot/base_footprint/imu'
        ],
        parameters=[{'use_sim_time': True}]
    )
    
    # Declare launch argument for enabling SLAM
    slam_arg = DeclareLaunchArgument(
        'slam',
        default_value='true',
        description='Whether to run SLAM (slam_toolbox)'
    )
    
    # Path to SLAM config
    slam_config = os.path.join(pkg_share, 'config', 'slam_toolbox_params.yaml')

    # SLAM Toolbox online mapping
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('slam_toolbox'),
                'launch',
                'online_sync_launch.py'
            )
        ),
        condition=IfCondition(LaunchConfiguration('slam')),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file': slam_config
        }.items()
    )
    
    # EKF Node configuration file path
    ekf_config = os.path.join(pkg_share, 'config', 'ekf.yaml')
    
    # EKF Node from robot_localization
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config, {'use_sim_time': True}]
    )

    # Safety Filter Node
    safety_filter = Node(
        package='household_robot_simulation',
        executable='cmd_vel_safety_filter.py',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        world_arg,
        slam_arg,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge,
        rviz,
        static_tf_lidar,
        static_tf_camera,
        static_tf_imu,
        ekf_node,
        safety_filter,
        slam_toolbox
    ])
