import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Get package share directory for rw_py (where your Python nodes are)
    pkg_rw_py_dir = get_package_share_directory('rw_py')
    # Get package share directory for rw (where waypoints.yaml might be, if not overridden)
    pkg_rw_dir = get_package_share_directory('rw')

    # --- Declare Launch Arguments ---
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info', # Set to 'debug' for more verbose output from follower
        description='Logging level (debug, info, warn, error, fatal)'
    )

    waypoints_yaml_path_arg = DeclareLaunchArgument(
        'waypoints_yaml_path',
        default_value=PathJoinSubstitution([pkg_rw_dir, 'config', 'nav_waypoints.yaml']),
        description='Full path to the waypoints YAML file'
    )

    # Segmentation Node Parameters
    segmentation_node_name_arg = DeclareLaunchArgument(
        'segmentation_node_name',
        default_value='roi_lidar_fusion_node_activated',
        description='Name of the ROI Lidar Fusion node'
    )
    corrected_goal_topic_arg = DeclareLaunchArgument(
        'corrected_local_goal_topic',
        default_value='/corrected_local_goal',
        description='Topic for corrected local goals from segmentation node'
    )
    camera_optical_frame_arg = DeclareLaunchArgument(
        'camera_optical_frame',
        default_value='front_camera_link_optical',
        description='TF frame of the camera optical center'
    )
    lidar_optical_frame_arg = DeclareLaunchArgument(
        'lidar_optical_frame',
        default_value='front_lidar_link_optical',
        description='TF frame of the Lidar sensor'
    )
    navigation_frame_arg = DeclareLaunchArgument(
        'navigation_frame_rlfn',
        default_value='map',
        description='Navigation frame for corrected goals published by RLFN'
    )
    # New parameters for segmentation node's crack following logic
    crack_continuity_threshold_arg = DeclareLaunchArgument(
        'crack_continuity_threshold_m',
        default_value='0.5', # Meters
        description='Max distance for a new point to be considered part of the same crack'
    )
    min_crack_pixels_arg = DeclareLaunchArgument(
        'min_crack_pixels_for_valid_target',
        default_value='50', # Pixels
        description='Min number of black pixels for a blob to be a valid crack segment target'
    )


    # Waypoint Follower Parameters
    correction_activation_distance_arg = DeclareLaunchArgument(
        'correction_activation_distance',
        default_value='2.0', # Meters - Reduced distance to activate search when closer to global WP
        description='Distance to global waypoint (crack start area) to activate crack search'
    )
    local_target_arrival_threshold_arg = DeclareLaunchArgument( # For crack segments
        'local_target_arrival_threshold',
        default_value='0.25', # Meters - Tighter threshold for arriving at crack segments
        description='Threshold to consider a local crack segment target reached'
    )
    local_goal_update_threshold_arg = DeclareLaunchArgument( # For crack segments
        'local_goal_update_threshold',
        default_value='0.15', # Meters - More sensitive to changes along the crack
        description='Minimum distance change for updating local NavToPose goal along a crack'
    )
    robot_base_frame_arg = DeclareLaunchArgument(
        'robot_base_frame',
        default_value='base_link',
        description='Robot base frame for TF lookups in WaypointFollower'
    )
    global_frame_arg = DeclareLaunchArgument(
        'global_frame_wlf',
        default_value='map',
        description='Global frame for navigation and waypoints in WaypointFollower'
    )
    # Timeout for waiting for the first crack point after activation
    crack_search_timeout_arg = DeclareLaunchArgument(
        'crack_search_timeout_s',
        default_value='15.0', # Seconds
        description='Timeout for waiting for the first crack point from segmentation'
    )


    # --- Node Definitions ---

    roi_lidar_fusion_node = Node(
        package='rw_py',
        executable='fusion_segmentation_node', # Make sure this is your segmentation node executable
        name=LaunchConfiguration('segmentation_node_name'),
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'input_image_topic': '/camera/image'},
            {'input_pc_topic': '/scan_02/points'},
            {'output_corrected_goal_topic': LaunchConfiguration('corrected_local_goal_topic')},
            {'navigation_frame': LaunchConfiguration('navigation_frame_rlfn')},
            {'camera_optical_frame': LaunchConfiguration('camera_optical_frame')},
            {'lidar_optical_frame': LaunchConfiguration('lidar_optical_frame')},
            {'output_window': 'Fused View'}, # Set to "" to disable CV window
            {'img_w': 1920}, {'img_h': 1200}, {'hfov': 1.25},
            {'enable_black_segmentation': True},
            {'black_h_max': 180}, {'black_s_max': 255}, {'black_v_max': 0}, # Adjusted V_max
            {'min_dist_colorize': 1.0}, {'max_dist_colorize': 10.0},
            {'point_display_mode': 2},
            # New conceptual parameters for segmentation node
            {'crack_continuity_threshold_m': LaunchConfiguration('crack_continuity_threshold_m')},
            {'min_crack_pixels_for_valid_target': LaunchConfiguration('min_crack_pixels_for_valid_target')},
        ]
    )

    waypoint_follower_corrected_node = Node(
        package='rw_py',
        executable='follow_waypoints', # Your WaypointFollowerCorrected node
        name='waypoint_follower_corrected_node',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'waypoints_yaml_path': LaunchConfiguration('waypoints_yaml_path')},
            {'correction_activation_distance': LaunchConfiguration('correction_activation_distance')},
            {'local_target_arrival_threshold': LaunchConfiguration('local_target_arrival_threshold')},
            {'local_goal_update_threshold': LaunchConfiguration('local_goal_update_threshold')},
            {'segmentation_node_name': LaunchConfiguration('segmentation_node_name')},
            {'corrected_local_goal_topic': LaunchConfiguration('corrected_local_goal_topic')},
            {'robot_base_frame': LaunchConfiguration('robot_base_frame')},
            {'global_frame': LaunchConfiguration('global_frame_wlf')},
            {'crack_search_timeout_s': LaunchConfiguration('crack_search_timeout_s')},
        ]
    )

    return LaunchDescription([
        use_sim_time_arg, log_level_arg, waypoints_yaml_path_arg,
        segmentation_node_name_arg, corrected_goal_topic_arg,
        camera_optical_frame_arg, lidar_optical_frame_arg, navigation_frame_arg,
        crack_continuity_threshold_arg, min_crack_pixels_arg, # For segmentation node
        correction_activation_distance_arg, local_target_arrival_threshold_arg,
        local_goal_update_threshold_arg, robot_base_frame_arg, global_frame_arg,
        crack_search_timeout_arg, # For follower node

        LogInfo(msg=["Launching CRACK FOLLOWING waypoint navigation."]),
        LogInfo(msg=["  waypoints_yaml_path: ", LaunchConfiguration('waypoints_yaml_path')]),
        LogInfo(msg=["  segmentation_node_name: ", LaunchConfiguration('segmentation_node_name')]),
        LogInfo(msg=["  correction_activation_distance (for crack search): ", LaunchConfiguration('correction_activation_distance')]),

        roi_lidar_fusion_node,
        waypoint_follower_corrected_node,
    ])