#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import yaml
from pathlib import Path
import os
from ament_index_python.packages import get_package_share_directory

class WaypointPublisherNode(Node):
    """
    Node #1: Reads a YAML file of waypoints and continuously publishes them
    as a PoseArray for logic consumption and a MarkerArray for visualization.
    """
    def __init__(self):
        super().__init__('waypoint_publisher_node')

        self.declare_parameter('waypoints_yaml_path', self._get_default_yaml_path())
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('publish_rate_hz', 1.0)

        self.yaml_path_ = self.get_parameter('waypoints_yaml_path').get_parameter_value().string_value
        self.global_frame_ = self.get_parameter('global_frame').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self.all_waypoints_ = []
        self.pose_array_ = PoseArray()
        self.marker_array_ = MarkerArray()

        # Latched QoS to ensure new subscribers get the last published message
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.pose_array_publisher_ = self.create_publisher(PoseArray, 'global_waypoints_array', latched_qos)
        self.marker_publisher_ = self.create_publisher(MarkerArray, 'global_waypoints_markers', latched_qos)

        if self.load_waypoints_from_yaml():
            self.timer_ = self.create_timer(1.0 / publish_rate, self.publish_data)
        else:
            self.get_logger().error("Node initialization failed due to waypoint loading error. Shutting down.")
            rclpy.shutdown()

    def _get_default_yaml_path(self):
        try:
            return os.path.join(get_package_share_directory('rw'), 'config', 'nav_waypoints.yaml')
        except Exception:
            return ""

    def load_waypoints_from_yaml(self) -> bool:
        if not self.yaml_path_ or not Path(self.yaml_path_).is_file():
            self.get_logger().error(f"Waypoint YAML file not found or path invalid: '{self.yaml_path_}'")
            return False
        
        try:
            with open(self.yaml_path_, 'r') as file:
                yaml_data = yaml.safe_load(file)

            if not yaml_data or 'poses' not in yaml_data:
                self.get_logger().error(f"YAML '{self.yaml_path_}' is empty or has no 'poses' key.")
                return False

            self.pose_array_.header.frame_id = self.global_frame_
            
            for i, pose_entry in enumerate(yaml_data['poses']):
                ps_msg = PoseStamped()
                ps_msg.header.frame_id = self.global_frame_
                ps_msg.header.stamp = self.get_clock().now().to_msg()
                
                pose_block = pose_entry.get('pose', {})
                pos_data = pose_block.get('position', {})
                orient_data = pose_block.get('orientation', {})

                ps_msg.pose.position.x = float(pos_data.get('x', 0.0))
                ps_msg.pose.position.y = float(pos_data.get('y', 0.0))
                ps_msg.pose.position.z = float(pos_data.get('z', 0.0))
                ps_msg.pose.orientation.x = float(orient_data.get('x', 0.0))
                ps_msg.pose.orientation.y = float(orient_data.get('y', 0.0))
                ps_msg.pose.orientation.z = float(orient_data.get('z', 0.0))
                ps_msg.pose.orientation.w = float(orient_data.get('w', 1.0))
                
                self.all_waypoints_.append(ps_msg)
                self.pose_array_.poses.append(ps_msg.pose)

                # Create a marker for this waypoint
                marker = Marker()
                marker.header.frame_id = self.global_frame_
                marker.ns = "initial_global_waypoints"
                marker.id = i
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.pose = ps_msg.pose
                marker.scale.x = 0.8
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0  # Blue
                marker.color.a = 0.8
                self.marker_array_.markers.append(marker)
            
            self.get_logger().info(f"Successfully loaded {len(self.all_waypoints_)} waypoints.")
            return True
        except Exception as e:
            self.get_logger().error(f"Error processing YAML file '{self.yaml_path_}': {e}")
            return False

    def publish_data(self):
        now = self.get_clock().now().to_msg()
        self.pose_array_.header.stamp = now
        for marker in self.marker_array_.markers:
            marker.header.stamp = now

        self.pose_array_publisher_.publish(self.pose_array_)
        self.marker_publisher_.publish(self.marker_array_)
        self.get_logger().debug("Published global waypoints (PoseArray and MarkerArray).")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()