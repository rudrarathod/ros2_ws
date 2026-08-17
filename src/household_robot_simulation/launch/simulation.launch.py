import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
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
        parameters=[{'robot_description': robot_desc}]
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
            '-z', '0.1'
        ]
    )
    
    # ROS 2 to Gazebo Sim Bridge Node (bridges /cmd_vel)
    # Gazebo Harmonic uses gz.msgs.Twist, ROS2 Jazzy uses geometry_msgs/msg/Twist
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'
        ]
    )
    
    return LaunchDescription([
        world_arg,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge
    ])
