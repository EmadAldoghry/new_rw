import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_rw_dir = get_package_share_directory('rw')

    # --- Declare Common Launch Arguments ---
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')
    waypoints_yaml_path_arg = DeclareLaunchArgument('waypoints_yaml_path',
        default_value=PathJoinSubstitution([pkg_rw_dir, 'config', 'nav_waypoints.yaml']))

    # --- Node Definitions ---

    # 1. Waypoint Publisher Node
    waypoint_publisher_node = Node(
        package='rw_py', executable='waypoint_publisher', name='waypoint_publisher_node',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'waypoints_yaml_path': LaunchConfiguration('waypoints_yaml_path')},
        ]
    )

    # 2. Visual Fusion Node (with segmentation, fusion, and visualization)
    visual_fusion_node = Node(
        package='rw_py',
        executable='visual_fuser',
        name='visual_fusion_node',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'input_image_topic': '/camera/image'},
            {'input_pc_topic': '/scan_02/points'},
            {'camera_optical_frame': 'front_camera_link_optical'},
            {'lidar_frame': 'front_lidar_link_optical'},
            {'hfov': 1.25},
            {'output_window': 'Fused View'}, # Set to "" to disable
            # Add segmentation params if you want to override defaults
            {'black_v_max': 50}, 
        ]
    )

    # 3. Centroid Calculator Node (Service)
    centroid_calculator_node = Node(
        package='rw_py', executable='centroid_calculator', name='centroid_calculator_node',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # 4. Navigation Logic Node
    navigation_logic_node = Node(
        package='rw_py', executable='navigation_logic', name='navigation_logic_node',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'correction_activation_distance': 7.0},
            {'robot_base_frame': 'base_link'},
        ]
    )

    # 5. Orchestrator Node
    orchestrator_node = Node(
        package='rw_py', executable='orchestrator', name='orchestrator_node',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'local_target_arrival_threshold': 0.35},
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        waypoints_yaml_path_arg,
        waypoint_publisher_node,
        visual_fusion_node,
        centroid_calculator_node,
        navigation_logic_node,
        orchestrator_node,
    ])