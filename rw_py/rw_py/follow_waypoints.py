#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter, ParameterType # Keep ParameterType for Descriptor
from rcl_interfaces.msg import ParameterDescriptor # For fallback declaration

from geometry_msgs.msg import PoseStamped as GeometryPoseStamped, Point, Quaternion
from nav2_msgs.action import NavigateToPose
import yaml
from pathlib import Path
import os
from ament_index_python.packages import get_package_share_directory
import tf2_ros
import math
from enum import Enum, auto
import traceback
from action_msgs.msg import GoalStatus
import time
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import SetBool

class NavState(Enum):
    IDLE = auto()
    LOADING_WAYPOINTS = auto()
    SELECTING_NEXT_CRACK_START_WAYPOINT = auto()
    NAVIGATING_TO_CRACK_START_AREA = auto()
    ACTIVATING_CRACK_SEARCH = auto()
    SEARCHING_FOR_CRACK_START = auto()
    FOLLOWING_CRACK = auto()
    CRACK_FOLLOWING_COMPLETE = auto()
    WAYPOINT_SEQUENCE_COMPLETE = auto()
    MISSION_FAILED = auto()

class WaypointFollowerCorrected(Node):
    def __init__(self):
        super().__init__('waypoint_follower_corrected_node_crack_mode')

        if not self.has_parameter("use_sim_time"):
            self.get_logger().warn("'use_sim_time' parameter not found from launch, declaring with default True.")
            self.declare_parameter("use_sim_time", True, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL, description='Use simulation (Gazebo) clock if true'))

        param_use_sim_time = self.get_parameter("use_sim_time")
        if param_use_sim_time.get_parameter_value().bool_value:
            self.get_logger().info(f"Node '{self.get_name()}' is setting its clock to use simulation time.")
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)]) # Use Parameter.Type.BOOL
        else:
            self.get_logger().info(f"Node '{self.get_name()}' is using system (wall) time.")
        
        self.get_logger().info(f"Node '{self.get_name()}' __init__ started after use_sim_time handling.")

        default_yaml_path = self._get_default_yaml_path('rw', 'config', 'nav_waypoints.yaml')
        self.declare_parameter('waypoints_yaml_path', default_yaml_path)
        self.declare_parameter('correction_activation_distance', 2.0)
        self.declare_parameter('local_target_arrival_threshold', 0.25)
        self.declare_parameter('local_goal_update_threshold', 0.15)
        self.declare_parameter('segmentation_node_name', 'roi_lidar_fusion_node_activated')
        self.declare_parameter('corrected_local_goal_topic', '/corrected_local_goal')
        self.declare_parameter('robot_base_frame', 'base_link') # Consider changing default to 'base_footprint'
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('crack_search_timeout_s', 15.0)

        self.yaml_path_ = self.get_parameter('waypoints_yaml_path').get_parameter_value().string_value
        self.correction_activation_dist_ = self.get_parameter('correction_activation_distance').get_parameter_value().double_value
        self.local_arrival_thresh_ = self.get_parameter('local_target_arrival_threshold').get_parameter_value().double_value
        self.local_goal_update_threshold_ = self.get_parameter('local_goal_update_threshold').get_parameter_value().double_value
        self.segmentation_node_name_param_ = self.get_parameter('segmentation_node_name').get_parameter_value().string_value 
        self.corrected_goal_topic_ = self.get_parameter('corrected_local_goal_topic').get_parameter_value().string_value
        self.robot_base_frame_ = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        self.global_frame_ = self.get_parameter('global_frame').get_parameter_value().string_value
        self.crack_search_timeout_s_ = self.get_parameter('crack_search_timeout_s').get_parameter_value().double_value
        
        self.all_waypoints_ = []
        self.last_truly_completed_global_idx_ = -1
        self.current_target_global_crack_start_wp_idx_ = -1
        self.is_crack_start_locked_ = False
        self.current_actual_crack_start_pose_ = None
        self.last_sent_crack_segment_pose_ = None
        self.state_ = NavState.IDLE 
        self.search_timer_ = None

        self.tf_buffer_ = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=10.0))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        self.get_logger().info(f"TF Buffer and Listener initialized for node '{self.get_name()}'.")
        
        self._tf_ready = False
        self._wait_for_tf() 

        self._navigate_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._current_nav_to_pose_goal_handle = None

        self.activation_srv_name_ = f"/{self.segmentation_node_name_param_}/activate_segmentation"
        self.segmentation_activation_client_ = self.create_client(SetBool, self.activation_srv_name_)
        self.get_logger().info(f"Attempting to create service client for: {self.activation_srv_name_}")

        self.corrected_goal_sub_ = None
        self.latest_corrected_goal_ = None
        
        self.marker_publisher_ = self.create_publisher(MarkerArray, '~/debug_waypoints_markers', 10)
        self.marker_publish_timer_ = self.create_timer(1.0, self.publish_waypoint_markers)

        self.get_logger().info(f"Node '{self.get_name()}' initialized. Current state: {self.state_.name}")
        
        if self.state_ != NavState.MISSION_FAILED: 
            self._change_state_and_process(NavState.LOADING_WAYPOINTS)
        else: 
            self._process_state_actions()

    # ... (rest of WaypointFollowerCorrected methods as before, no changes needed for this specific error) ...
    def _wait_for_tf(self):
        self.get_logger().info("Waiting for TF tree...")
        for i in range(20):
            current_tf_time = self.get_clock().now()
            if self.tf_buffer_.can_transform(
                self.global_frame_, 
                self.robot_base_frame_, 
                current_tf_time, 
                timeout=RclpyDuration(seconds=0.05) 
            ):
                self._tf_ready = True
                self.get_logger().info(f"TF is ready ({self.global_frame_} -> {self.robot_base_frame_})."); 
                return
            self.get_logger().info(f"Waiting for TF ({self.global_frame_} -> {self.robot_base_frame_})... try {i+1}/20")
            time.sleep(0.25) 
        
        if not self._tf_ready:
            self.get_logger().error("TF tree did not become available after 20 retries. Mission will fail.")
            self.state_ = NavState.MISSION_FAILED

    def _get_default_yaml_path(self, package_name, config_dir, file_name):
        try: return os.path.join(get_package_share_directory(package_name), config_dir, file_name)
        except Exception as e: self.get_logger().error(f"Error getting default YAML path: {e}"); return ""

    def _change_state_and_process(self, new_state: NavState):
        if self.state_ == new_state and new_state not in [NavState.LOADING_WAYPOINTS, NavState.SELECTING_NEXT_CRACK_START_WAYPOINT]:
             self.get_logger().debug(f"Already in state {new_state.name} or re-entry blocked. No transition.")
             return
        self.get_logger().info(f"STATE: {self.state_.name} -> {new_state.name}")
        old_state = self.state_
        self.state_ = new_state

        if old_state == NavState.SEARCHING_FOR_CRACK_START and self.search_timer_:
            if not self.search_timer_.is_canceled(): self.search_timer_.cancel()
            self.destroy_timer(self.search_timer_); self.search_timer_ = None
        if old_state in [NavState.SEARCHING_FOR_CRACK_START, NavState.FOLLOWING_CRACK] and \
           new_state not in [NavState.SEARCHING_FOR_CRACK_START, NavState.FOLLOWING_CRACK, NavState.MISSION_FAILED]:
            self._activate_segmentation_node(False)
            self._destroy_corrected_goal_subscriber()
            self.is_crack_start_locked_ = False
            self.current_actual_crack_start_pose_ = None
            self.last_sent_crack_segment_pose_ = None
            self.latest_corrected_goal_ = None
        
        self._process_state_actions()

    def _process_state_actions(self):
        self.get_logger().debug(f"Processing actions for state: {self.state_.name}")
        if self.state_ == NavState.LOADING_WAYPOINTS:
            if not self.yaml_path_ or not Path(self.yaml_path_).is_file():
                self.get_logger().error(f"Waypoint YAML '{self.yaml_path_}' not found."); self._change_state_and_process(NavState.MISSION_FAILED); return
            if self.load_waypoints_from_yaml(Path(self.yaml_path_)):
                if self.all_waypoints_:
                    self.last_truly_completed_global_idx_ = -1
                    self._change_state_and_process(NavState.SELECTING_NEXT_CRACK_START_WAYPOINT)
                else:
                    self.get_logger().info("No waypoints loaded."); self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE)
            else:
                self.get_logger().error("Failed to load waypoints."); self._change_state_and_process(NavState.MISSION_FAILED)

        elif self.state_ == NavState.SELECTING_NEXT_CRACK_START_WAYPOINT:
            next_crack_start_idx = -1
            for i in range(self.last_truly_completed_global_idx_ + 1, len(self.all_waypoints_)):
                if i % 2 == 0: 
                    next_crack_start_idx = i
                    break
            if next_crack_start_idx != -1:
                self.current_target_global_crack_start_wp_idx_ = next_crack_start_idx
                target_pose = self.all_waypoints_[self.current_target_global_crack_start_wp_idx_]
                self.get_logger().info(f"Selected next global crack start: Waypoint {self.current_target_global_crack_start_wp_idx_} at P(x={target_pose.pose.position.x:.2f}, y={target_pose.pose.position.y:.2f}).")
                self._send_navigate_to_pose_goal(target_pose, is_global_wp=True)
            else:
                self.get_logger().info("All crack start waypoints processed.")
                self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE)
        
        elif self.state_ == NavState.ACTIVATING_CRACK_SEARCH:
            self.get_logger().info(f"Activating crack search for global WP {self.current_target_global_crack_start_wp_idx_}.")
            if self._activate_segmentation_node(True):
                self._create_corrected_goal_subscriber()
                self.is_crack_start_locked_ = False
                self.latest_corrected_goal_ = None
                self.search_timer_ = self.create_timer(self.crack_search_timeout_s_, self._crack_search_timeout_cb)
                self._change_state_and_process(NavState.SEARCHING_FOR_CRACK_START)
            else:
                self.get_logger().error("Failed to activate segmentation. Skipping this crack.")
                self._handle_crack_search_failure()

        elif self.state_ == NavState.CRACK_FOLLOWING_COMPLETE:
             self.get_logger().info(f"Crack associated with global WP {self.current_target_global_crack_start_wp_idx_} is complete.")
             self.last_truly_completed_global_idx_ = self.current_target_global_crack_start_wp_idx_
             self._change_state_and_process(NavState.SELECTING_NEXT_CRACK_START_WAYPOINT)

        elif self.state_ == NavState.WAYPOINT_SEQUENCE_COMPLETE:
            self.get_logger().info("All waypoints processed. Mission successful!")

        elif self.state_ == NavState.MISSION_FAILED:
            self.get_logger().error("Mission failed. Attempting cleanup.")
            self._cancel_current_navigation_action()
            self._activate_segmentation_node(False)

    def _crack_search_timeout_cb(self):
        if self.state_ == NavState.SEARCHING_FOR_CRACK_START:
            self.get_logger().warn(f"Timeout waiting for crack start for global WP {self.current_target_global_crack_start_wp_idx_}.")
            if self.search_timer_ and not self.search_timer_.is_canceled(): self.search_timer_.cancel()
            if self.search_timer_: self.destroy_timer(self.search_timer_); self.search_timer_ = None
            self._handle_crack_search_failure()
            
    def _handle_crack_search_failure(self):
        self.get_logger().info(f"Crack search failed for global WP {self.current_target_global_crack_start_wp_idx_}. Moving to next.")
        self.last_truly_completed_global_idx_ = self.current_target_global_crack_start_wp_idx_
        self._change_state_and_process(NavState.SELECTING_NEXT_CRACK_START_WAYPOINT)

    def load_waypoints_from_yaml(self, yaml_file_path: Path) -> bool:
        try:
            with open(yaml_file_path, 'r') as file: yaml_data = yaml.safe_load(file)
            if not yaml_data or 'poses' not in yaml_data:
                self.get_logger().error(f"YAML '{yaml_file_path}' empty or no 'poses' key."); return False
            loaded_waypoints = []
            for i, pose_entry in enumerate(yaml_data['poses']):
                try:
                    ps_msg = GeometryPoseStamped()
                    header_data = pose_entry.get('header', {})
                    ps_msg.header.frame_id = header_data.get('frame_id', self.global_frame_)
                    ps_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_block = pose_entry.get('pose', {})
                    pos_data = pose_block.get('position', {}); orient_data = pose_block.get('orientation', {})
                    ps_msg.pose.position.x = float(pos_data.get('x',0.0))
                    ps_msg.pose.position.y = float(pos_data.get('y',0.0))
                    ps_msg.pose.position.z = float(pos_data.get('z',0.0))
                    ps_msg.pose.orientation.x=float(orient_data.get('x',0.0))
                    ps_msg.pose.orientation.y=float(orient_data.get('y',0.0))
                    ps_msg.pose.orientation.z=float(orient_data.get('z',0.0))
                    ps_msg.pose.orientation.w=float(orient_data.get('w',1.0))
                    loaded_waypoints.append(ps_msg)
                except Exception as e: self.get_logger().error(f"Error parsing waypoint {i+1}: {e}"); return False
            self.all_waypoints_ = loaded_waypoints
            self.get_logger().info(f"Successfully loaded {len(self.all_waypoints_)} waypoints from '{yaml_file_path}'.")
            return True
        except Exception as e: self.get_logger().error(f"Error loading YAML file '{yaml_file_path}': {e}"); return False

    def _create_corrected_goal_subscriber(self):
        if self.corrected_goal_sub_ is None:
            self.get_logger().info(f"Creating subscriber for corrected goals on '{self.corrected_goal_topic_}'.")
            _qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.corrected_goal_sub_ = self.create_subscription(GeometryPoseStamped, self.corrected_goal_topic_, self.corrected_goal_callback, _qos)

    def _destroy_corrected_goal_subscriber(self):
        if self.corrected_goal_sub_:
            self.get_logger().info("Destroying subscriber for corrected goals.")
            self.destroy_subscription(self.corrected_goal_sub_); self.corrected_goal_sub_ = None
            self.latest_corrected_goal_ = None

    def get_robot_pose(self) -> GeometryPoseStamped | None:
        if not self._tf_ready:
            self.get_logger().warn("TF not ready, cannot get robot pose.", throttle_duration_sec=2.0)
            return None
        try:
            current_tf_time = self.get_clock().now()
            transform = self.tf_buffer_.lookup_transform(
                self.global_frame_, 
                self.robot_base_frame_, 
                current_tf_time, 
                timeout=RclpyDuration(seconds=0.1)
            )
            pose = GeometryPoseStamped()
            pose.header.stamp = transform.header.stamp 
            pose.header.frame_id = self.global_frame_
            pose.pose.position = transform.transform.translation
            pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().warn(f"TF Robot Pose Error looking up {self.global_frame_} -> {self.robot_base_frame_}: {e}", throttle_duration_sec=2.0)
            return None

    def _activate_segmentation_node(self, activate: bool):
        if not self.segmentation_activation_client_.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"Service '{self.activation_srv_name_}' not ready for {'activation' if activate else 'deactivation'}.")
            return False
        req = SetBool.Request(); req.data = activate
        future = self.segmentation_activation_client_.call_async(req)
        self.get_logger().info(f"Requested segmentation node {'activation' if activate else 'deactivation'}.")
        return True 

    def _cancel_current_navigation_action(self):
        if self._current_nav_to_pose_goal_handle and self._current_nav_to_pose_goal_handle.is_active: 
            self.get_logger().info(f"Requesting cancellation of current NavigateToPose goal (ID: ...{bytes(self._current_nav_to_pose_goal_handle.goal_id.uuid).hex()[-6:]}).")
            self._current_nav_to_pose_goal_handle.cancel_goal_async()
        else:
            self.get_logger().debug("No active NavToPose goal to cancel or handle not set.")

    def _send_navigate_to_pose_goal(self, target_pose: GeometryPoseStamped, is_global_wp: bool = False):
        if not self._tf_ready:
            self.get_logger().error("Cannot send NavToPose goal, TF not ready."); self._change_state_and_process(NavState.MISSION_FAILED); return
        
        self._cancel_current_navigation_action() 

        if not self._navigate_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("'navigate_to_pose' action server not available.")
            self._change_state_and_process(NavState.MISSION_FAILED); return

        goal_msg = NavigateToPose.Goal(); goal_msg.pose = target_pose
        goal_type = "GLOBAL WP (crack start area)" if is_global_wp else "LOCAL CRACK SEGMENT"
        self.get_logger().info(f"Sending {goal_type} to NavigateToPose: P(x={target_pose.pose.position.x:.2f}, y={target_pose.pose.position.y:.2f}) in {target_pose.header.frame_id}")
        
        send_goal_future = self._navigate_to_pose_client.send_goal_async(
            goal_msg, feedback_callback=lambda fb_msg: self.navigate_to_pose_feedback_cb(fb_msg, is_global_wp)
        )
        send_goal_future.add_done_callback(lambda future: self.navigate_to_pose_goal_response_cb(future, is_global_wp))
        
        if not is_global_wp: 
            self.last_sent_crack_segment_pose_ = target_pose

    def navigate_to_pose_goal_response_cb(self, future, is_global_wp_goal: bool):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Exception getting goal handle: {e}")
            goal_handle = None

        if not goal_handle or not goal_handle.accepted:
            goal_type = "Global WP" if is_global_wp_goal else "Local Crack Segment"
            self.get_logger().error(f"NavigateToPose goal for {goal_type} REJECTED or future failed.")
            self._current_nav_to_pose_goal_handle = None
            if self.state_ != NavState.MISSION_FAILED:
                if is_global_wp_goal:
                    self.get_logger().error("Failed to navigate to global WP area. Skipping this crack.")
                    self._handle_crack_search_failure()
                else:
                    self.get_logger().error("Failed to navigate to local crack segment. Ending crack follow.")
                    self._handle_crack_following_ended(success=False)
            return

        self._current_nav_to_pose_goal_handle = goal_handle
        goal_type = "Global WP" if is_global_wp_goal else "Local Crack Segment"
        self.get_logger().info(f"NavigateToPose goal for {goal_type} (ID: ...{bytes(goal_handle.goal_id.uuid).hex()[-6:]}) ACCEPTED.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda fut: self.navigate_to_pose_result_cb(fut, is_global_wp_goal))

        if is_global_wp_goal and self.state_ == NavState.SELECTING_NEXT_CRACK_START_WAYPOINT:
            self._change_state_and_process(NavState.NAVIGATING_TO_CRACK_START_AREA)
        elif not is_global_wp_goal and self.state_ == NavState.SEARCHING_FOR_CRACK_START:
             if self.search_timer_ and not self.search_timer_.is_canceled(): self.search_timer_.cancel()
             if self.search_timer_: self.destroy_timer(self.search_timer_); self.search_timer_ = None
             self._change_state_and_process(NavState.FOLLOWING_CRACK)

    def navigate_to_pose_feedback_cb(self, feedback_msg: NavigateToPose.Feedback, is_global_wp_goal: bool):
        if is_global_wp_goal and self.state_ == NavState.NAVIGATING_TO_CRACK_START_AREA:
            robot_pose = self.get_robot_pose()
            if robot_pose and self.current_target_global_crack_start_wp_idx_ != -1:
                target_wp = self.all_waypoints_[self.current_target_global_crack_start_wp_idx_]
                dist_sq = (robot_pose.pose.position.x - target_wp.pose.position.x)**2 + \
                          (robot_pose.pose.position.y - target_wp.pose.position.y)**2
                if dist_sq < self.correction_activation_dist_**2:
                    self.get_logger().info(f"Near global WP {self.current_target_global_crack_start_wp_idx_} (dist: {math.sqrt(dist_sq):.2f}m). Activating crack search.")
                    self._cancel_current_navigation_action()
        
        elif not is_global_wp_goal and self.state_ == NavState.FOLLOWING_CRACK:
            robot_pose = self.get_robot_pose()
            if robot_pose and self.last_sent_crack_segment_pose_ and self.last_sent_crack_segment_pose_.header.frame_id:
                dist_sq = (robot_pose.pose.position.x - self.last_sent_crack_segment_pose_.pose.position.x)**2 + \
                          (robot_pose.pose.position.y - self.last_sent_crack_segment_pose_.pose.position.y)**2
                if dist_sq < self.local_arrival_thresh_**2:
                    self.get_logger().info(f"Local crack segment arrival by proximity (dist {math.sqrt(dist_sq):.2f}m). Cancelling NavToPose.")
                    self._cancel_current_navigation_action()

    def navigate_to_pose_result_cb(self, future, is_global_wp_goal: bool):
        try:
            action_result_wrapper = future.result()
        except Exception as e:
            self.get_logger().error(f"Exception getting action result: {e}")
            action_result_wrapper = None
            
        if not action_result_wrapper:
            self.get_logger().error("NavToPose result future had no wrapper or future failed.");
            if self.state_ != NavState.MISSION_FAILED:
                self._change_state_and_process(NavState.MISSION_FAILED) 
            return
        
        status = action_result_wrapper.status
        status_name = {getattr(GoalStatus, name): name for name in dir(GoalStatus) if name.startswith("STATUS_")}.get(status, "UNKNOWN_STATUS")
        result_goal_id_hex = bytes(action_result_wrapper.goal_id.uuid).hex()

        current_handle_id_hex = "N/A"
        if self._current_nav_to_pose_goal_handle:
            current_handle_id_hex = bytes(self._current_nav_to_pose_goal_handle.goal_id.uuid).hex()
        
        if self._current_nav_to_pose_goal_handle and result_goal_id_hex == current_handle_id_hex:
            self.get_logger().debug(f"Result for current goal handle ...{result_goal_id_hex[-6:]} received.")
            self._current_nav_to_pose_goal_handle = None 
        else:
            if status == GoalStatus.STATUS_CANCELED:
                 self.get_logger().info(f"NavToPose CANCELED (ID ...{result_goal_id_hex[-6:]}). May be due to preemption or proximity.")
            else:
                self.get_logger().warn(f"NavToPose result for goal ID ...{result_goal_id_hex[-6:]} (status: {status_name}) does not match current handle ID ...{current_handle_id_hex[-6:]} or no current handle. This might be a late response for an old goal.")
                if result_goal_id_hex != current_handle_id_hex: 
                    return 

        goal_type = "Global WP" if is_global_wp_goal else "Local Crack Segment"
        self.get_logger().info(f"NavigateToPose for {goal_type} (ID: ...{result_goal_id_hex[-6:]}) finished with status: {status_name}")

        if is_global_wp_goal:
            if self.state_ != NavState.NAVIGATING_TO_CRACK_START_AREA:
                self.get_logger().warn(f"NavToPose (global) result in unexpected state {self.state_.name}. Goal ID ...{result_goal_id_hex[-6:]}. Ignoring unless critical."); return
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"Reached global WP {self.current_target_global_crack_start_wp_idx_} area. Activating search.")
                self._change_state_and_process(NavState.ACTIVATING_CRACK_SEARCH)
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().info("Global WP NavToPose CANCELED (likely by proximity for search activation).")
                self._change_state_and_process(NavState.ACTIVATING_CRACK_SEARCH)
            else: 
                self.get_logger().error(f"Navigation to global WP {self.current_target_global_crack_start_wp_idx_} FAILED/ABORTED. Skipping crack.")
                self._handle_crack_search_failure()
        
        else: 
            if self.state_ not in [NavState.FOLLOWING_CRACK, NavState.SEARCHING_FOR_CRACK_START]: 
                 self.get_logger().warn(f"NavToPose (local) result in unexpected state {self.state_.name}. Goal ID ...{result_goal_id_hex[-6:]}. Ignoring."); return

            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("Local crack segment navigation SUCCEEDED. Waiting for next segment from segmentation.")
            elif status == GoalStatus.STATUS_CANCELED:
                is_prox_arrival = self.last_sent_crack_segment_pose_ and self._check_arrival_at_pose(self.last_sent_crack_segment_pose_, self.local_arrival_thresh_)
                if is_prox_arrival:
                    self.get_logger().info("Local crack segment NavToPose CANCELED (by proximity). Waiting for next segment.")
                else:
                    self.get_logger().info("Local crack segment NavToPose CANCELED (likely preempted by new corrected goal).")
                    if self.latest_corrected_goal_ and self.latest_corrected_goal_.header.frame_id:
                        self.get_logger().info("Attempting to send latest corrected goal after cancellation.")
                        self._send_navigate_to_pose_goal(self.latest_corrected_goal_, is_global_wp=False)
                    else:
                        self.get_logger().info("No valid new local goal available after cancellation. Assuming crack ended.")
                        self._handle_crack_following_ended(success=True) 
            else: 
                self.get_logger().error(f"Local crack segment navigation FAILED or ABORTED (Status: {status_name}). Ending crack follow.")
                self._handle_crack_following_ended(success=False)

    def corrected_goal_callback(self, msg: GeometryPoseStamped):
        self.latest_corrected_goal_ = msg
        
        if not msg.header.frame_id:
            self.get_logger().info("Corrected goal received with EMPTY frame_id (crack end / no target).")
            if self.state_ == NavState.SEARCHING_FOR_CRACK_START:
                self.get_logger().warn("Segmentation reported no target during initial crack search.")
                if self.search_timer_ and not self.search_timer_.is_canceled(): self.search_timer_.cancel()
                if self.search_timer_: self.destroy_timer(self.search_timer_); self.search_timer_ = None
                self._handle_crack_search_failure()
            elif self.state_ == NavState.FOLLOWING_CRACK:
                self.get_logger().info("Segmentation reported crack end. Finalizing this crack.")
                if self._current_nav_to_pose_goal_handle and self._current_nav_to_pose_goal_handle.is_active:
                    self._cancel_current_navigation_action() 
                else:
                    self._handle_crack_following_ended(success=True) 
            return

        if self.state_ == NavState.SEARCHING_FOR_CRACK_START:
            if not self.is_crack_start_locked_:
                self.get_logger().info(f"ACTION: First valid crack start detected at P(x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}). Locking in.")
                self.current_actual_crack_start_pose_ = msg
                self.is_crack_start_locked_ = True
                self._send_navigate_to_pose_goal(msg, is_global_wp=False) 
            else:
                 self.get_logger().debug("Already locked on a crack start, but received another goal in SEARCHING state. Storing.")
        elif self.state_ == NavState.FOLLOWING_CRACK:
            self.get_logger().info(f"CORRECTION: Received updated crack segment at P(x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}).")
            significant_change = True
            if self.last_sent_crack_segment_pose_:
                dist_sq_diff = (msg.pose.position.x - self.last_sent_crack_segment_pose_.pose.position.x)**2 + \
                               (msg.pose.position.y - self.last_sent_crack_segment_pose_.pose.position.y)**2
                if math.sqrt(dist_sq_diff) < self.local_goal_update_threshold_:
                    significant_change = False
            
            if not self._current_nav_to_pose_goal_handle or not self._current_nav_to_pose_goal_handle.is_active:
                self.get_logger().info("ACTION: No active local NavToPose. Sending new corrected goal.")
                self._send_navigate_to_pose_goal(msg, is_global_wp=False)
            elif significant_change:
                self.get_logger().info("ACTION: Crack segment UPDATED significantly. Preempting current NavToPose.")
                self._cancel_current_navigation_action() 
            else:
                self.get_logger().debug("New crack segment not significantly different. Not preempting.")
        else:
            self.get_logger().debug(f"Corrected goal received in unexpected state {self.state_.name}. Storing but not acting.")

    def _handle_crack_following_ended(self, success: bool = True):
        self.get_logger().info(f"Crack following for global WP {self.current_target_global_crack_start_wp_idx_} ended. Success: {success}")
        if success:
            self._change_state_and_process(NavState.CRACK_FOLLOWING_COMPLETE)
        else:
            self.get_logger().warn("Crack following ended due to an issue. Skipping to next global WP.")
            self.last_truly_completed_global_idx_ = self.current_target_global_crack_start_wp_idx_
            self._change_state_and_process(NavState.SELECTING_NEXT_CRACK_START_WAYPOINT)

    def _check_arrival_at_pose(self, target_pose: GeometryPoseStamped, threshold: float):
        robot_pose = self.get_robot_pose()
        if not robot_pose or not target_pose: return False
        dist_sq = (robot_pose.pose.position.x - target_pose.pose.position.x)**2 + \
                  (robot_pose.pose.position.y - target_pose.pose.position.y)**2
        return dist_sq < threshold**2

    def publish_waypoint_markers(self):
        marker_array = MarkerArray(); now = self.get_clock().now().to_msg()
        for i, pose_stamped in enumerate(self.all_waypoints_):
            if i % 2 != 0: continue
            marker = Marker(); marker.header.frame_id = self.global_frame_; marker.header.stamp = now
            marker.ns = "global_crack_starts"; marker.id = i; marker.type = Marker.ARROW; marker.action = Marker.ADD
            marker.pose = pose_stamped.pose
            marker.scale.x = 0.6; marker.scale.y = 0.1; marker.scale.z = 0.1
            if i == self.current_target_global_crack_start_wp_idx_ and \
               self.state_ in [NavState.NAVIGATING_TO_CRACK_START_AREA, NavState.ACTIVATING_CRACK_SEARCH, NavState.SEARCHING_FOR_CRACK_START]:
                marker.color.r = 1.0; marker.color.g = 0.5; marker.color.b = 0.0; marker.color.a = 1.0 
            elif i <= self.last_truly_completed_global_idx_:
                marker.color.r = 0.0; marker.color.g = 0.5; marker.color.b = 0.0; marker.color.a = 0.7 
            else:
                marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 0.8 
            marker.lifetime = RclpyDuration(seconds=2.0).to_msg(); marker_array.markers.append(marker)

        if self.current_actual_crack_start_pose_ and self.state_ in [NavState.SEARCHING_FOR_CRACK_START, NavState.FOLLOWING_CRACK, NavState.CRACK_FOLLOWING_COMPLETE]:
            actual_start_marker = Marker(); actual_start_marker.header.frame_id = self.global_frame_; actual_start_marker.header.stamp = now
            actual_start_marker.ns = "actual_crack_start"; actual_start_marker.id = 0; actual_start_marker.type = Marker.SPHERE; actual_start_marker.action = Marker.ADD
            actual_start_marker.pose = self.current_actual_crack_start_pose_.pose
            actual_start_marker.scale.x = 0.5; actual_start_marker.scale.y = 0.5; actual_start_marker.scale.z = 0.5
            actual_start_marker.color.r = 1.0; actual_start_marker.color.g = 1.0; actual_start_marker.color.b = 0.0; actual_start_marker.color.a = 1.0 
            actual_start_marker.lifetime = RclpyDuration(seconds=2.0).to_msg(); marker_array.markers.append(actual_start_marker)

        if self.last_sent_crack_segment_pose_ and self.state_ == NavState.FOLLOWING_CRACK:
            segment_marker = Marker(); segment_marker.header.frame_id = self.global_frame_; segment_marker.header.stamp = now
            segment_marker.ns = "current_crack_segment"; segment_marker.id = 0; segment_marker.type = Marker.CUBE; segment_marker.action = Marker.ADD
            segment_marker.pose = self.last_sent_crack_segment_pose_.pose
            segment_marker.scale.x = 0.3; segment_marker.scale.y = 0.3; segment_marker.scale.z = 0.3
            segment_marker.color.r = 1.0; segment_marker.color.g = 0.0; segment_marker.color.b = 0.0; segment_marker.color.a = 1.0 
            segment_marker.lifetime = RclpyDuration(seconds=2.0).to_msg(); marker_array.markers.append(segment_marker)
            if self.current_actual_crack_start_pose_:
                line_marker = Marker(); line_marker.header.frame_id = self.global_frame_; line_marker.header.stamp = now
                line_marker.ns = "crack_path_estimate"; line_marker.id = 0; line_marker.type = Marker.LINE_STRIP; line_marker.action = Marker.ADD
                line_marker.scale.x = 0.05
                line_marker.color.r = 1.0; line_marker.color.g = 0.7; line_marker.color.b = 0.8; line_marker.color.a = 0.8 
                p_start = self.current_actual_crack_start_pose_.pose.position
                p_curr  = self.last_sent_crack_segment_pose_.pose.position
                line_marker.points.extend([Point(x=p_start.x, y=p_start.y, z=p_start.z), Point(x=p_curr.x, y=p_curr.y, z=p_curr.z)])
                line_marker.lifetime = RclpyDuration(seconds=2.0).to_msg(); marker_array.markers.append(line_marker)

        if marker_array.markers: self.marker_publisher_.publish(marker_array)

    def destroy_node(self):
        self.get_logger().info(f"Destroying node '{self.get_name()}'...")
        if self.search_timer_ and not self.search_timer_.is_canceled(): self.search_timer_.cancel()
        if self.search_timer_: self.destroy_timer(self.search_timer_); self.search_timer_ = None
        self._activate_segmentation_node(False)
        self._cancel_current_navigation_action()
        if self.marker_publish_timer_ and not self.marker_publish_timer_.is_canceled(): self.marker_publish_timer_.cancel()
        super().destroy_node()
        self.get_logger().info(f"Node '{self.get_name()}' destroyed.")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointFollowerCorrected()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info('Node interrupted (KeyboardInterrupt).')
    except SystemExit:
        if node: node.get_logger().info('Node shutting down (SystemExit).')
    except Exception as e:
        err_node_name = node.get_name() if node else "WaypointFollowerCorrected (during init)"
        rclpy.logging.get_logger(err_node_name).error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("WaypointFollowerCorrected (Crack Mode) main finally block finished.")

if __name__ == '__main__':
    main()