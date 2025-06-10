#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration

from geometry_msgs.msg import PoseStamped, PoseArray, Point
from nav2_msgs.action import FollowWaypoints, NavigateToPose
from action_msgs.msg import GoalStatus
from enum import Enum, auto
from visualization_msgs.msg import Marker, MarkerArray

from rw_interfaces.msg import NavigationDecision
from rw_interfaces.srv import GetSegmentedPoints, CalculateCentroid

class NavState(Enum):
    IDLE = auto()
    AWAITING_GLOBAL_WAYPOINTS = auto()
    FOLLOWING_GLOBAL_WAYPOINTS = auto()
    REQUESTING_SEGMENTED_CLOUD = auto()
    REQUESTING_CENTROID = auto()
    FOLLOWING_LOCAL_TARGET = auto()
    WAYPOINT_SEQUENCE_COMPLETE = auto()
    MISSION_FAILED = auto()

class OrchestratorNode(Node):
    def __init__(self):
        super().__init__('orchestrator_node')

        self.declare_parameter('local_target_arrival_threshold', 0.35)
        self.local_arrival_thresh_sq_ = self.get_parameter('local_target_arrival_threshold').get_parameter_value().double_value ** 2

        # State machine
        self.state_ = NavState.IDLE
        self.all_waypoints_ = []
        self.last_completed_global_idx_ = -1
        self.current_correction_target_idx_ = -1
        self.latest_corrected_goal_ = None

        # Action Clients for Nav2
        self.follow_waypoints_client_ = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.navigate_to_pose_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.fw_goal_handle_ = None
        self.ntp_goal_handle_ = None

        # Service Clients
        self.get_segmented_points_client_ = self.create_client(GetSegmentedPoints, 'get_segmented_3d_points')
        self.calc_centroid_client_ = self.create_client(CalculateCentroid, 'calculate_centroid')

        # Subscribers
        self.create_subscription(PoseArray, 'global_waypoints_array', self.waypoints_cb, 10)
        self.create_subscription(NavigationDecision, 'navigation_decision', self.nav_decision_cb, 10)
        
        # Publisher for Visualization
        self.marker_publisher_ = self.create_publisher(MarkerArray, '~/debug_markers', 10)
        self.marker_timer_ = self.create_timer(1.0, self.publish_debug_markers)

        self.get_logger().info('Orchestrator Node started. Waiting for waypoints...')
        self.change_state(NavState.AWAITING_GLOBAL_WAYPOINTS)

    def change_state(self, new_state: NavState):
        if self.state_ == new_state: return
        self.get_logger().info(f"STATE: {self.state_.name} -> {new_state.name}")
        self.state_ = new_state

    # --- Subscriber Callbacks ---
    def waypoints_cb(self, msg: PoseArray):
        if self.state_ == NavState.AWAITING_GLOBAL_WAYPOINTS:
            self.all_waypoints_ = [PoseStamped(header=msg.header, pose=p) for p in msg.poses]
            self.get_logger().info(f"Received {len(self.all_waypoints_)} global waypoints. Starting mission.")
            self.start_global_waypoint_navigation()

    def nav_decision_cb(self, msg: NavigationDecision):
        if self.state_ != NavState.FOLLOWING_GLOBAL_WAYPOINTS: return

        if msg.correction_is_required and msg.active_waypoint_index > self.last_completed_global_idx_:
            self.get_logger().info(f"Decision to activate correction for waypoint {msg.active_waypoint_index}. Requesting segmented cloud.")
            self.current_correction_target_idx_ = msg.active_waypoint_index
            self.cancel_follow_waypoints() # Cancel global nav
            self.change_state(NavState.REQUESTING_SEGMENTED_CLOUD)
            
            # Call the service on the visual fusion node
            if not self.get_segmented_points_client_.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("GetSegmentedPoints service not available.")
                self.resume_global_navigation_after_failure()
                return

            request = GetSegmentedPoints.Request()
            future = self.get_segmented_points_client_.call_async(request)
            future.add_done_callback(self.get_segmented_points_response_cb)

    # --- Service Callbacks ---
    def get_segmented_points_response_cb(self, future):
        try:
            response = future.result()
            if not response.success or response.point_cloud.width == 0:
                self.get_logger().warn(f"Failed to get segmented 3D points: {response.message}. Resuming.")
                self.resume_global_navigation_after_failure()
                return

            self.get_logger().info("Received segmented point cloud. Requesting centroid calculation.")
            self.change_state(NavState.REQUESTING_CENTROID)
            
            request = CalculateCentroid.Request(point_cloud=response.point_cloud)
            if not self.calc_centroid_client_.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("CalculateCentroid service not available.")
                self.resume_global_navigation_after_failure()
                return

            centroid_future = self.calc_centroid_client_.call_async(request)
            centroid_future.add_done_callback(self.calculate_centroid_response_cb)
        except Exception as e:
            self.get_logger().error(f"Exception in GetSegmentedPoints callback: {e}")
            self.resume_global_navigation_after_failure()

    def calculate_centroid_response_cb(self, future):
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warn(f"Failed to calculate centroid: {response.message}. Resuming.")
                self.resume_global_navigation_after_failure()
                return
            
            self.get_logger().info("Centroid calculated. Navigating to corrected local goal.")
            self.latest_corrected_goal_ = response.corrected_goal
            self.start_local_target_navigation(self.latest_corrected_goal_)
        except Exception as e:
            self.get_logger().error(f"Exception in CalculateCentroid callback: {e}")
            self.resume_global_navigation_after_failure()

    # --- Navigation and Marker Methods ---
    def start_global_waypoint_navigation(self):
        next_wp_idx = self.last_completed_global_idx_ + 1
        if next_wp_idx >= len(self.all_waypoints_):
            self.change_state(NavState.WAYPOINT_SEQUENCE_COMPLETE)
            self.get_logger().info("Mission Complete!")
            return
        
        self.change_state(NavState.FOLLOWING_GLOBAL_WAYPOINTS)
        self.current_correction_target_idx_ = -1
        self.latest_corrected_goal_ = None

        goal_msg = FollowWaypoints.Goal(poses=self.all_waypoints_[next_wp_idx:])
        self.follow_waypoints_client_.wait_for_server()
        future = self.follow_waypoints_client_.send_goal_async(goal_msg)
        future.add_done_callback(self.fw_goal_response_cb)

    def start_local_target_navigation(self, goal_pose: PoseStamped):
        self.change_state(NavState.FOLLOWING_LOCAL_TARGET)
        goal_msg = NavigateToPose.Goal(pose=goal_pose)

        self.navigate_to_pose_client_.wait_for_server()
        future = self.navigate_to_pose_client_.send_goal_async(goal_msg)
        future.add_done_callback(self.ntp_goal_response_cb)

    def resume_global_navigation_after_failure(self):
        if self.current_correction_target_idx_ > self.last_completed_global_idx_:
            self.last_completed_global_idx_ = self.current_correction_target_idx_
        self.start_global_waypoint_navigation()

    def publish_debug_markers(self):
        if not self.all_waypoints_: return
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()
        
        # Visualize original global waypoints
        for i, pose_stamped in enumerate(self.all_waypoints_):
            marker = Marker(header=pose_stamped.header, ns="global_waypoints", id=i,
                            type=Marker.ARROW, action=Marker.ADD, pose=pose_stamped.pose)
            marker.header.stamp = now
            marker.scale.x, marker.scale.y, marker.scale.z = 0.5, 0.08, 0.08
            marker.lifetime = RclpyDuration(seconds=2.0).to_msg()
            
            if i <= self.last_completed_global_idx_:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.5, 0.5, 0.5, 0.7 # Gray
            elif i == self.current_correction_target_idx_:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.5, 0.0, 1.0 # Orange
            else:
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 0.0, 1.0, 0.8 # Blue
            marker_array.markers.append(marker)

        # Visualize corrected goal and connecting line
        if self.state_ in [NavState.FOLLOWING_LOCAL_TARGET, NavState.REQUESTING_CENTROID] and self.latest_corrected_goal_:
            # Corrected goal marker
            local_marker = Marker(header=self.latest_corrected_goal_.header, ns="local_corrected_target", id=0,
                                  type=Marker.SPHERE, action=Marker.ADD, pose=self.latest_corrected_goal_.pose)
            local_marker.header.stamp = now
            local_marker.scale.x, local_marker.scale.y, local_marker.scale.z = 0.4, 0.4, 0.4
            local_marker.color.r, local_marker.color.g, local_marker.color.b, local_marker.color.a = 1.0, 0.0, 0.0, 1.0 # Red
            local_marker.lifetime = RclpyDuration(seconds=2.0).to_msg()
            marker_array.markers.append(local_marker)
            
            # Line marker
            if self.current_correction_target_idx_ < len(self.all_waypoints_):
                og_pose = self.all_waypoints_[self.current_correction_target_idx_].pose
                line_marker = Marker(header=self.latest_corrected_goal_.header, ns="correction_line", id=0,
                                     type=Marker.LINE_STRIP, action=Marker.ADD)
                line_marker.header.stamp = now
                line_marker.scale.x = 0.05
                line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.color.a = 1.0, 1.0, 0.0, 0.8 # Yellow
                line_marker.points.extend([og_pose.position, self.latest_corrected_goal_.pose.position])
                line_marker.lifetime = RclpyDuration(seconds=2.0).to_msg()
                marker_array.markers.append(line_marker)

        self.marker_publisher_.publish(marker_array)

    # --- Action Goal/Result Callbacks (abbreviated, no change in logic) ---
    def fw_goal_response_cb(self, future):
        self.fw_goal_handle_ = future.result(); self.fw_goal_handle_.get_result_async().add_done_callback(self.fw_result_cb)
    def fw_result_cb(self, future):
        status = future.result().status; self.fw_goal_handle_ = None
        if status == GoalStatus.STATUS_SUCCEEDED: self.change_state(NavState.WAYPOINT_SEQUENCE_COMPLETE)
        elif status != GoalStatus.STATUS_CANCELED: self.change_state(NavState.MISSION_FAILED)
    def ntp_goal_response_cb(self, future):
        self.ntp_goal_handle_ = future.result(); self.ntp_goal_handle_.get_result_async().add_done_callback(self.ntp_result_cb)
    def ntp_result_cb(self, future):
        status = future.result().status; self.ntp_goal_handle_ = None
        if status == GoalStatus.STATUS_SUCCEEDED: self.last_completed_global_idx_ = self.current_correction_target_idx_; self.start_global_waypoint_navigation()
        else: self.resume_global_navigation_after_failure()
    def cancel_follow_waypoints(self):
        if self.fw_goal_handle_: self.fw_goal_handle_.cancel_goal_async()

def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()