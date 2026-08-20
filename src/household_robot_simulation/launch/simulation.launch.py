import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction, AppendEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    # Get package directories
    pkg_share = get_package_share_directory('household_robot_simulation')
    gz_sim_share = get_package_share_directory('ros_gz_sim')
    
    # Set Gazebo resource path to resolve package:// and model:// URIs
    pkg_share_parent = os.path.dirname(pkg_share)
    set_gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        pkg_share_parent
    )
    
    # Paths
    world_file = os.path.join(pkg_share, 'worlds', 'house.sdf')
    urdf_file = os.path.join(pkg_share, 'urdf', 'custom_robot.urdf')
    
    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_file,
        description='Path to the SDF world file to load'
    )
    
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Whether to run Gazebo in headless mode (no GUI)'
    )
    
    # Include Gazebo Sim launch file
    # We append ' -r' to run the simulation automatically on startup
    # If headless is true, we pass -s to run only the server
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': PythonExpression([
                '"-s -r " + "', LaunchConfiguration('world'), '" if "', LaunchConfiguration('headless'), '" == "true" else "-r " + "', LaunchConfiguration('world'), '"'
            ])
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
        parameters=[{'config_file': bridge_config}]
    )
    
    # Path to RViz configuration
    rviz_config = os.path.join(pkg_share, 'rviz', 'robot.rviz')
    
    # RViz Node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        condition=UnlessCondition(LaunchConfiguration('headless'))
    )
    
    # Declare launch argument for enabling SLAM
    slam_arg = DeclareLaunchArgument(
        'slam',
        default_value='true',
        description='Whether to run SLAM (slam_toolbox)'
    )
    
    # Declare launch argument for enabling navigation
    nav_arg = DeclareLaunchArgument(
        'nav',
        default_value='false',
        description='Whether to run autonomous navigation (Nav2)'
    )
    
    # Declare launch argument for enabling vision processor
    vision_arg = DeclareLaunchArgument(
        'vision',
        default_value='true',
        description='Whether to run the OpenCV vision processor'
    )
    
    # Declare launch argument for enabling YOLO object detection
    yolo_arg = DeclareLaunchArgument(
        'yolo',
        default_value='false',
        description='Whether to run the YOLO object detection node'
    )
    
    # Declare launch argument for enabling Person Follower
    follow_arg = DeclareLaunchArgument(
        'follow',
        default_value='true',
        description='Whether to run the person follower node'
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
    
    # Nav2 Autonomous Navigation bringup (delayed by 5s to allow clock to start)
    navigation = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_share, 'launch', 'navigation.launch.py')
                ),
                condition=IfCondition(LaunchConfiguration('nav')),
                launch_arguments={
                    'use_sim_time': 'true'
                }.items()
            )
        ]
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
        parameters=[{
            'use_sim_time': True,
            'safety_padding': 0.28
        }]
    )

    # Vision Processor Node
    vision_processor = Node(
        package='household_robot_simulation',
        executable='vision_processor.py',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('vision'))
    )

    # YOLO Detector Node
    yolo_detector = Node(
        package='household_robot_simulation',
        executable='yolo_detector.py',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('yolo'))
    )

    # Person Follower Node
    person_follower = Node(
        package='household_robot_simulation',
        executable='person_follower.py',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('follow'))
    )
    
    return LaunchDescription([
        set_gz_resource_path,
        world_arg,
        headless_arg,
        slam_arg,
        nav_arg,
        vision_arg,
        yolo_arg,
        follow_arg,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge,
        rviz,
        ekf_node,
        safety_filter,
        vision_processor,
        yolo_detector,
        person_follower,
        slam_toolbox,
        navigation
    ])
