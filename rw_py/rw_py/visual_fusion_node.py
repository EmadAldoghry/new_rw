#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from rclpy.duration import Duration as RclpyDuration
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import tf2_ros
import tf_transformations

from rw_interfaces.srv import GetSegmentedPoints

class VisualFusionNode(Node):
    """
    Combined node for segmentation, fusion, and visualization.
    - Subscribes to synchronized Image and PointCloud2.
    - Segments the image to find a black region of interest.
    - Projects LiDAR points onto the image.
    - Displays a debug window showing all projected points, with segmented points highlighted.
    - Provides a service to return the 3D PointCloud of only the segmented points.
    """
    def __init__(self):
        super().__init__('visual_fusion_node')

        # Declare all parameters from both previous nodes
        self.declare_parameter('input_image_topic', 'camera/image')
        self.declare_parameter('input_pc_topic', 'scan_02/points')
        self.declare_parameter('output_window', 'Fused View')
        self.declare_parameter('camera_optical_frame', 'front_camera_link_optical')
        self.declare_parameter('lidar_frame', 'front_lidar_link_optical')
        self.declare_parameter('img_w', 1920)
        self.declare_parameter('img_h', 1200)
        self.declare_parameter('hfov', 1.25)
        self.declare_parameter('roi_x_start', int(1920 * 0 / 100.0))
        self.declare_parameter('roi_y_start', int(1200 * 19 / 100.0))
        self.declare_parameter('roi_x_end', int(1920 * 100 / 100.0))
        self.declare_parameter('roi_y_end', int(1200 * 76 / 100.0))
        for name, default in [('black_h_min', 0), ('black_s_min', 0), ('black_v_min', 0),
                              ('black_h_max', 180), ('black_s_max', 255), ('black_v_max', 50)]:
            self.declare_parameter(name, default)

        # Get all parameters
        self.output_window_ = self.get_parameter('output_window').value
        self.camera_optical_frame_ = self.get_parameter('camera_optical_frame').value
        self.lidar_frame_ = self.get_parameter('lidar_frame').value
        img_w = self.get_parameter('img_w').value
        img_h = self.get_parameter('img_h').value
        hfov = self.get_parameter('hfov').value
        self.roi_ = (self.get_parameter('roi_x_start').value, self.get_parameter('roi_y_start').value, 
                     self.get_parameter('roi_x_end').value, self.get_parameter('roi_y_end').value)
        self.lower_hsv_ = np.array([self.get_parameter(f'black_{c}_min').value for c in 'hsv'])
        self.upper_hsv_ = np.array([self.get_parameter(f'black_{c}_max').value for c in 'hsv'])

        # Setup
        self.bridge_ = CvBridge()
        self.fx_, self.fy_, self.cx_, self.cy_ = img_w / (2*math.tan(hfov/2)), img_w / (2*math.tan(hfov/2)), img_w/2, img_h/2
        self.img_w_, self.img_h_ = img_w, img_h

        # TF
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        self.lidar_to_cam_transform_ = None
        self.get_lidar_to_cam_transform()

        # Data storage for the service
        self.last_segmented_points_3d_ = []
        self.last_pc_header_ = None
        
        # Subscribers
        qos = rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value
        self.image_sub = message_filters.Subscriber(self, Image, self.get_parameter('input_image_topic').value, qos)
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, self.get_parameter('input_pc_topic').value, qos)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.pc_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        # Service
        self.get_points_service_ = self.create_service(GetSegmentedPoints, 'get_segmented_3d_points', self.get_segmented_points_callback)
        
        if self.output_window_:
            cv2.namedWindow(self.output_window_, cv2.WINDOW_NORMAL)

        self.get_logger().info('Visual Fusion Node started.')

    def get_lidar_to_cam_transform(self):
        try:
            trans = self.tf_buffer_.lookup_transform(self.camera_optical_frame_, self.lidar_frame_, RclpyTime(), RclpyDuration(seconds=5.0))
            self.lidar_to_cam_transform_ = trans
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")

    def sync_callback(self, image_msg: Image, pc_msg: PointCloud2):
        if not self.lidar_to_cam_transform_:
            self.get_lidar_to_cam_transform()
            if not self.lidar_to_cam_transform_: return

        try:
            cv_image = self.bridge_.imgmsg_to_cv2(image_msg, 'bgr8')
            fused_display_image = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
            
        # 1. Segmentation
        roi_x_start, roi_y_start, roi_x_end, roi_y_end = self.roi_
        roi_content = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        hsv_roi = cv2.cvtColor(roi_content, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, self.lower_hsv_, self.upper_hsv_)
        
        # 2. Fusion and Projection
        points_raw = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        segmented_points_in_frame = []

        q = self.lidar_to_cam_transform_.transform.rotation
        t = self.lidar_to_cam_transform_.transform.translation
        mat = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        mat[:3, 3] = [t.x, t.y, t.z]

        for p in points_raw:
            point_in_cam = np.dot(mat, [p[0], p[1], p[2], 1.0])
            Z = point_in_cam[0]
            if Z <= 0.01: continue
            
            u = int(self.fx_ * -point_in_cam[1] / Z + self.cx_)
            v = int(self.fy_ * -point_in_cam[2] / Z + self.cy_)

            if 0 <= u < self.img_w_ and 0 <= v < self.img_h_:
                is_on_target = False
                if roi_x_start <= u < roi_x_end and roi_y_start <= v < roi_y_end:
                    if mask[v - roi_y_start, u - roi_x_start] > 0:
                        is_on_target = True
                        segmented_points_in_frame.append([p[0], p[1], p[2]])
                
                if self.output_window_:
                    color = (0, 0, 255) if is_on_target else (255, 255, 0)
                    cv2.circle(fused_display_image, (u, v), 2, color, -1)
        
        # Store results for the service
        self.last_segmented_points_3d_ = segmented_points_in_frame
        self.last_pc_header_ = pc_msg.header

        # 3. Display
        if self.output_window_:
            cv2.rectangle(fused_display_image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
            cv2.imshow(self.output_window_, fused_display_image)
            cv2.waitKey(1)
            
    def get_segmented_points_callback(self, request: GetSegmentedPoints.Request, response: GetSegmentedPoints.Response):
        if not self.last_segmented_points_3d_ or not self.last_pc_header_:
            response.success = False
            response.message = "No segmented points available."
            return response
        
        response.point_cloud = pc2.create_cloud_xyz32(self.last_pc_header_, self.last_segmented_points_3d_)
        response.success = True
        response.message = f"Found {len(self.last_segmented_points_3d_)} segmented 3D points."
        return response

    def destroy_node(self):
        if self.output_window_:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisualFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()