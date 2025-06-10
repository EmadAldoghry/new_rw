#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration as RclpyDuration

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Point, PointStamped, Quaternion
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
import tf_transformations

from rw_interfaces.srv import CalculateCentroid

class CentroidCalculatorNode(Node):
    """
    Node #4: Provides a service to calculate the centroid of a given PointCloud2,
    transform it into the global navigation frame, and return it as a PoseStamped goal.
    """
    def __init__(self):
        super().__init__('centroid_calculator_node')

        self.declare_parameter('navigation_frame', 'map')
        self.navigation_frame_ = self.get_parameter('navigation_frame').value

        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        self.calc_centroid_service_ = self.create_service(
            CalculateCentroid,
            'calculate_centroid',
            self.calculate_centroid_callback
        )
        self.get_logger().info("Centroid Calculator Node started.")

    def calculate_centroid_callback(self, request: CalculateCentroid.Request, response: CalculateCentroid.Response):
        pc_msg = request.point_cloud
        points = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
        
        if not points:
            response.success = False
            response.message = "Received empty point cloud."
            return response

        # Calculate centroid in the point cloud's original frame
        target_points_np = np.array(points, dtype=np.float32)
        centroid_local_frame = np.mean(target_points_np, axis=0)
        
        pt_stamped_local = PointStamped()
        pt_stamped_local.header = pc_msg.header
        pt_stamped_local.point = Point(x=float(centroid_local_frame[0]), y=float(centroid_local_frame[1]), z=float(centroid_local_frame[2]))

        try:
            # Transform the centroid point to the navigation frame
            transform = self.tf_buffer_.lookup_transform(
                self.navigation_frame_,
                pt_stamped_local.header.frame_id,
                RclpyTime(),
                timeout=RclpyDuration(seconds=0.5)
            )
            pt_stamped_nav_frame = do_transform_point(pt_stamped_local, transform)

            # Create the final PoseStamped message
            corrected_goal_pose = PoseStamped()
            corrected_goal_pose.header.stamp = self.get_clock().now().to_msg()
            corrected_goal_pose.header.frame_id = self.navigation_frame_
            corrected_goal_pose.pose.position = pt_stamped_nav_frame.point
            
            # Set a default neutral orientation (pointing forward along X)
            q_identity = tf_transformations.quaternion_from_euler(0, 0, 0)
            corrected_goal_pose.pose.orientation = Quaternion(x=q_identity[0], y=q_identity[1], z=q_identity[2], w=q_identity[3])

            response.corrected_goal = corrected_goal_pose
            response.success = True
            response.message = "Centroid calculated and transformed successfully."
            self.get_logger().info(f"Calculated corrected goal in '{self.navigation_frame_}': P({corrected_goal_pose.pose.position.x:.2f}, {corrected_goal_pose.pose.position.y:.2f})")

        except Exception as ex:
            response.success = False
            response.message = f"Failed to transform centroid: {ex}"
            self.get_logger().warn(f"TF/Goal Calculation Error: {ex}", throttle_duration_sec=2.0)
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = CentroidCalculatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()