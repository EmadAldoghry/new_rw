#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration as RclpyDuration

from geometry_msgs.msg import PoseStamped, PoseArray
import tf2_ros
import math

from rw_interfaces.msg import NavigationDecision

class NavigationLogicNode(Node):
    """
    Node #5: The decision helper. It monitors the robot's position relative to the
    global waypoints and tells the orchestrator when it's time to switch to
    local correction mode.
    """
    def __init__(self):
        super().__init__('navigation_logic_node')

        self.declare_parameter('correction_activation_distance', 7.0)
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('update_rate_hz', 5.0)

        self.correction_activation_dist_sq_ = self.get_parameter('correction_activation_distance').get_parameter_value().double_value ** 2
        self.robot_base_frame_ = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        self.global_frame_ = self.get_parameter('global_frame').get_parameter_value().string_value
        update_rate = self.get_parameter('update_rate_hz').get_parameter_value().double_value

        self.global_waypoints_ = []
        self.current_robot_pose_ = None
        self.last_decision_ = None

        # TF
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        # Subscribers
        self.create_subscription(PoseArray, 'global_waypoints_array', self.waypoints_callback, 10)

        # Publisher
        self.decision_publisher_ = self.create_publisher(NavigationDecision, 'navigation_decision', 10)

        # Timer
        self.timer_ = self.create_timer(1.0 / update_rate, self.evaluate_decision)
        
        self.get_logger().info('Navigation Logic Node started.')
        
    def waypoints_callback(self, msg: PoseArray):
        self.global_waypoints_ = msg.poses
        self.get_logger().info(f'Received and stored {len(self.global_waypoints_)} global waypoints.')

    def get_robot_pose(self):
        try:
            transform = self.tf_buffer_.lookup_transform(
                self.global_frame_, self.robot_base_frame_, RclpyTime(), timeout=RclpyDuration(seconds=0.1)
            )
            self.current_robot_pose_ = transform.transform.translation
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f"Could not get robot pose: {e}", throttle_duration_sec=2.0)
            return False

    def evaluate_decision(self):
        if not self.global_waypoints_ or not self.get_robot_pose():
            return
        
        # This is a simplified logic. It finds the *closest* waypoint. A more robust
        # implementation would track which waypoint was last completed to only look forward.
        # This is sufficient for this modular design, as the orchestrator holds the true state.
        
        min_dist_sq = float('inf')
        target_idx = -1

        for i, waypoint_pose in enumerate(self.global_waypoints_):
            dist_sq = (self.current_robot_pose_.x - waypoint_pose.position.x)**2 + \
                      (self.current_robot_pose_.y - waypoint_pose.position.y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                target_idx = i
        
        if target_idx == -1:
            return

        decision = NavigationDecision()
        decision.active_waypoint_index = target_idx
        decision.correction_is_required = min_dist_sq < self.correction_activation_dist_sq_

        # Only publish if the decision has changed to avoid spamming the topic
        if self.last_decision_ is None or \
           self.last_decision_.active_waypoint_index != decision.active_waypoint_index or \
           self.last_decision_.correction_is_required != decision.correction_is_required:
            
            self.decision_publisher_.publish(decision)
            self.last_decision_ = decision
            self.get_logger().info(f"Decision update: Waypoint {target_idx}, Correction required: {decision.correction_is_required} (Dist: {math.sqrt(min_dist_sq):.2f}m)")

def main(args=None):
    rclpy.init(args=args)
    node = NavigationLogicNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()