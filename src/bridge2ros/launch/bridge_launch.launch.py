from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


package_name = 'ros_gz_sim'
launch_file_name = 'gz_server.launch.pi'


def generate_launch_description():
    # Path to your SDF file
    # TODO make local path

    sdf_file_path = '/home/parallels/ackerman/src/bridge2ros/bridge2ros/Ackermanmulti.sdf'
    sdf_file_car = '/home/parallels/ackerman/src/bridge2ros/bridge2ros/Models/car.sdf'
    rviz_file_path = '/home/parallels/ackerman/src/bridge2ros/bridge2ros/rviz/gpu_lidar_bridge.rviz'

    with open(sdf_file_car, 'r') as infp:
        robot_desc = infp.read()
        
    pkg_ros_gz_sim_demos = get_package_share_directory('ros_gz_sim_demos')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                   'launch',
                                   'gz_sim.launch.py'])]),
        launch_arguments={
            'gz_args': sdf_file_path
        }.items(),
    )


    gz_server_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                   'launch',
                                   'gz_sim.launch.py'])]),
        launch_arguments=[('gz_args', sdf_file_path), ])

    # Bridge
    bridge_lidar = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                   '/lidar/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked'],
        output='screen'
        )
    bridge_cmd_vel = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist'],
        output='screen'
        )
    
    bridge_pose = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/Car/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry'],
        output='screen'
        )
    
    # Rviz
    rviz = Node(
       package='rviz2',
       executable='rviz2',
       arguments=['-d', rviz_file_path],
       condition=IfCondition(LaunchConfiguration('rviz'))

        )

    
    # Controller
    visual = Node(
        package='bridge2ros',
        executable='visual',
        )
    cbf = Node(
        package='bridge2ros',
        executable='cbf',
        )
    
    joint_state_publisher_gui = Node(
     	package='joint_state_publisher_gui',
     	executable='joint_state_publisher_gui',
     	name='joint_state_publisher_gui',
     	arguments=[sdf_file_car],
     	output=['screen']
	 )
    
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[ 
            {'use_sim_time': True},
            {'robot_description': robot_desc},
        ]
    )
    
    #Launch Gazebo with the SDF file
    #ld = LaunchDescription()
    # ld.add_action(gz_server_description)
    # ld.add_action(bridge_lidar)
    # ld.add_action(bridge_cmd_vel)
    # #ld.add_action(dummy)
    # #ld.add_action(cbf)
    # #ld.add_action(joint_state_publisher_gui)
    # ld.add_action(robot_state_publisher)
    # ld.add_action(rviz)

    return LaunchDescription([
    ld,
    DeclareLaunchArgument('rviz', default_value='true',
                            description='Open RViz.'),
    bridge_lidar,
    bridge_cmd_vel,
    bridge_pose,
    visual
    ])
    return ld
