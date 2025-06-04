#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration as rclpyDuration
from rclpy.parameter import Parameter, ParameterType # ParameterType for Descriptor
from rcl_interfaces.msg import ParameterDescriptor # For fallback declaration

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import message_filters
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped, PoseStamped, Point, Quaternion
import math
import threading
import time
import traceback
from std_srvs.srv import SetBool

class ROILidarFusionNode(Node):
    def __init__(self):
        super().__init__('roi_lidar_fusion_node_internal_default_name') # Name will be overridden by launch

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

        self.bridge = CvBridge()

        # Declare other parameters
        self.declare_parameter("input_image_topic", "/camera/image_raw")
        self.declare_parameter("input_camera_info_topic", "/camera/camera_info")
        self.declare_parameter("input_pc_topic", "/lidar/points")
        self.declare_parameter("output_corrected_goal_topic", "/corrected_local_goal")
        self.declare_parameter("navigation_frame", "map")
        self.declare_parameter("camera_optical_frame", "camera_link_optical")
        self.declare_parameter("lidar_optical_frame", "lidar_link_optical")
        self.declare_parameter("output_window", "Fused View")
        self.declare_parameter("img_w", 640)
        self.declare_parameter("img_h", 480)
        self.declare_parameter("hfov", 1.57) # Not used if K from CameraInfo
        self.declare_parameter("enable_black_segmentation", True)
        self.declare_parameter("black_h_max", 180)
        self.declare_parameter("black_s_max", 255)
        self.declare_parameter("black_v_max", 50)
        self.declare_parameter("black_h_min", 0)
        self.declare_parameter("black_s_min", 0)
        self.declare_parameter("black_v_min", 0)
        self.declare_parameter("crack_continuity_threshold_m", 0.5)
        self.declare_parameter("min_crack_pixels_for_valid_target", 50)
        self.declare_parameter("min_dist_colorize", 0.5)
        self.declare_parameter("max_dist_colorize", 10.0)
        self.declare_parameter("point_display_mode", 0)

        # Get Parameter Values
        self.image_topic_ = self.get_parameter("input_image_topic").get_parameter_value().string_value
        self.cam_info_topic_ = self.get_parameter("input_camera_info_topic").get_parameter_value().string_value
        self.pc_topic_ = self.get_parameter("input_pc_topic").get_parameter_value().string_value
        self.output_goal_topic_ = self.get_parameter("output_corrected_goal_topic").get_parameter_value().string_value
        self.navigation_frame_ = self.get_parameter("navigation_frame").get_parameter_value().string_value
        self.camera_optical_frame_ = self.get_parameter("camera_optical_frame").get_parameter_value().string_value
        self.lidar_frame_ = self.get_parameter("lidar_optical_frame").get_parameter_value().string_value
        self.output_window_name_ = self.get_parameter("output_window").get_parameter_value().string_value
        self.img_w_param_ = self.get_parameter("img_w").get_parameter_value().integer_value # Store param value
        self.img_h_param_ = self.get_parameter("img_h").get_parameter_value().integer_value # Store param value
        self.enable_black_segmentation_ = self.get_parameter("enable_black_segmentation").get_parameter_value().bool_value
        self.black_h_max_ = self.get_parameter("black_h_max").get_parameter_value().integer_value
        self.black_s_max_ = self.get_parameter("black_s_max").get_parameter_value().integer_value
        self.black_v_max_ = self.get_parameter("black_v_max").get_parameter_value().integer_value
        self.black_h_min_ = self.get_parameter("black_h_min").get_parameter_value().integer_value
        self.black_s_min_ = self.get_parameter("black_s_min").get_parameter_value().integer_value
        self.black_v_min_ = self.get_parameter("black_v_min").get_parameter_value().integer_value
        self.crack_continuity_thresh_ = self.get_parameter("crack_continuity_threshold_m").get_parameter_value().double_value
        self.min_crack_pixels_ = self.get_parameter("min_crack_pixels_for_valid_target").get_parameter_value().integer_value
        self.min_dist_colorize_ = self.get_parameter("min_dist_colorize").get_parameter_value().double_value
        self.max_dist_colorize_ = self.get_parameter("max_dist_colorize").get_parameter_value().double_value
        self.point_display_mode_ = self.get_parameter("point_display_mode").get_parameter_value().integer_value

        # Internal State
        self.segmentation_active_ = False
        self.last_published_crack_point_map_ = None
        self.K_ = None; self.D_ = None
        self.img_w_ = self.img_w_param_ # Use param for initial default before CameraInfo
        self.img_h_ = self.img_h_param_ # Use param for initial default before CameraInfo
        self.latest_cv_image_ = None
        self.data_lock_ = threading.Lock()

        self.tf_buffer_ = tf2_ros.Buffer(cache_time=rclpyDuration(seconds=10.0))
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        self.get_logger().info(f"TF Buffer and Listener initialized for node '{self.get_name()}'.")

        qos_sensor_data = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_camera_info = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.image_sub_ = message_filters.Subscriber(self, Image, self.image_topic_, qos_profile=qos_sensor_data)
        self.cam_info_sub_ = message_filters.Subscriber(self, CameraInfo, self.cam_info_topic_, qos_profile=qos_camera_info)
        self.pc_sub_ = message_filters.Subscriber(self, PointCloud2, self.pc_topic_, qos_profile=qos_sensor_data)

        self.ts_ = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub_, self.cam_info_sub_, self.pc_sub_], queue_size=15, slop=0.2)
        self.ts_.registerCallback(self.synchronized_callback)
        self.get_logger().info(f"Sync subscribers for: Img='{self.image_topic_}', CamInfo='{self.cam_info_topic_}', LiDAR='{self.pc_topic_}'.")

        self.goal_pub_ = self.create_publisher(PoseStamped, self.output_goal_topic_, 10)
        self.activation_service_ = self.create_service(SetBool, '~/activate_segmentation', self.handle_activation_service)
        self.get_logger().info(f"Activation service '{self.get_name()}/activate_segmentation' created.")

        if self.output_window_name_: cv2.namedWindow(self.output_window_name_, cv2.WINDOW_AUTOSIZE)
        self.get_logger().info(f"Node '{self.get_name()}' initialized successfully.")

    def handle_activation_service(self, request: SetBool.Request, response: SetBool.Response):
        # ... (method content remains the same) ...
        if request.data:
            self.get_logger().info("Segmentation ACTIVATED via service call.")
            self.segmentation_active_ = True
            self.last_published_crack_point_map_ = None 
            response.success = True
            response.message = "Segmentation activated."
        else:
            self.get_logger().info("Segmentation DEACTIVATED via service call.")
            self.segmentation_active_ = False
            self.last_published_crack_point_map_ = None
            self.publish_empty_goal("Deactivated by service")
            response.success = True
            response.message = "Segmentation deactivated."
        return response

    def publish_empty_goal(self, reason: str = "No target found"):
        # ... (method content remains the same) ...
        empty_goal = PoseStamped()
        empty_goal.header.stamp = self.get_clock().now().to_msg()
        empty_goal.header.frame_id = "" 
        self.goal_pub_.publish(empty_goal)
        self.get_logger().debug(f"Published EMPTY goal: {reason}")

    def synchronized_callback(self, image_msg: Image, cam_info_msg: CameraInfo, pc_msg: PointCloud2):
        # ... (method content remains the same, ensure self.img_w_ and self.img_h_ are updated from cam_info_msg) ...
        current_time = self.get_clock().now()
        try:
            # Update Camera Intrinsics from CameraInfo (this will override param defaults)
            if self.K_ is None or self.img_w_ != cam_info_msg.width or self.img_h_ != cam_info_msg.height:
                self.K_ = np.array(cam_info_msg.k).reshape((3, 3))
                self.D_ = np.array(cam_info_msg.d)
                self.img_w_ = cam_info_msg.width # Update from CameraInfo
                self.img_h_ = cam_info_msg.height # Update from CameraInfo
                if self.K_[0,0] == 0:
                    self.get_logger().warn("Camera intrinsic fx is zero. Projection will fail.", throttle_duration_sec=5)
                    return
                self.get_logger().info(f"Camera intrinsics updated from CameraInfo: W={self.img_w_}, H={self.img_h_}, K00={self.K_[0,0]:.2f}")

            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            # Optional undistortion:
            # if self.D_ is not None and np.any(self.D_) and self.K_ is not None:
            #    cv_image = cv2.undistort(cv_image, self.K_, self.D_)
        
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
        except Exception as e:
            self.get_logger().error(f"Error in image processing or TF: {e}\n{traceback.format_exc()}")
            return

        if not self.segmentation_active_:
            if self.output_window_name_:
                display_image = cv_image.copy()
                # Ensure cv_image has been successfully converted before drawing
                if display_image is not None:
                    cv2.putText(display_image, "SEGMENTATION INACTIVE", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                    cv2.imshow(self.output_window_name_, display_image)
                    cv2.waitKey(1)
            return

        best_target_point_map = None
        processed_image_for_display = cv_image.copy()

        if self.enable_black_segmentation_ and self.K_ is not None:
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_black = np.array([self.black_h_min_, self.black_s_min_, self.black_v_min_])
            upper_black = np.array([self.black_h_max_, self.black_s_max_, self.black_v_max_])
            black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
            kernel = np.ones((5,5), np.uint8) # Consider making kernel size a parameter
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
            black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_candidate_points_map = []

            try:
                # Ensure pc_msg.header.frame_id is used as source
                transform_lidar_to_cam = self.tf_buffer_.lookup_transform(
                    self.camera_optical_frame_, 
                    pc_msg.header.frame_id, # Use frame_id from PointCloud2 message
                    pc_msg.header.stamp,
                    timeout=rclpyDuration(seconds=0.1) 
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"TF lookup failed (LiDAR frame '{pc_msg.header.frame_id}' to CamOptical frame '{self.camera_optical_frame_}'): {e}", throttle_duration_sec=1.0)
                self.publish_empty_goal(f"TF L->C error: {pc_msg.header.frame_id} to {self.camera_optical_frame_}")
                if self.output_window_name_: cv2.imshow(self.output_window_name_, processed_image_for_display); cv2.waitKey(1)
                return

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_crack_pixels_: continue
                cv2.drawContours(processed_image_for_display, [contour], -1, (0,255,0), 2)
                M = cv2.moments(contour)
                if M["m00"] == 0: continue
                # img_cx = int(M["m10"] / M["m00"]); img_cy = int(M["m01"] / M["m00"]) # Centroid not directly used for 3D point
                contour_points_3d_cam = []

                for point_data in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
                    p_lidar = PointStamped(); p_lidar.header.frame_id = pc_msg.header.frame_id
                    p_lidar.header.stamp = pc_msg.header.stamp 
                    p_lidar.point.x = float(point_data[0]); p_lidar.point.y = float(point_data[1]); p_lidar.point.z = float(point_data[2])
                    try: p_cam = do_transform_point(p_lidar, transform_lidar_to_cam)
                    except Exception as e_tf_point: self.get_logger().warn(f"TF point transform failed: {e_tf_point}", throttle_duration_sec=5.0); continue
                    
                    Xc, Yc, Zc = p_cam.point.x, p_cam.point.y, p_cam.point.z
                    if Zc <= 0.1: continue # Point is behind or too close to camera plane

                    u_proj = (self.K_[0,0] * Xc / Zc) + self.K_[0,2]
                    v_proj = (self.K_[1,1] * Yc / Zc) + self.K_[1,2]

                    if 0 <= u_proj < self.img_w_ and 0 <= v_proj < self.img_h_: 
                        if cv2.pointPolygonTest(contour, (u_proj, v_proj), False) >= 0:
                            contour_points_3d_cam.append((Xc, Yc, Zc, u_proj, v_proj))
                
                if not contour_points_3d_cam: continue
                contour_points_3d_cam.sort(key=lambda pt: pt[2]) # Sort by Zc (depth in camera frame)
                median_idx = len(contour_points_3d_cam) // 2
                best_3d_point_in_cam = contour_points_3d_cam[median_idx]
                
                target_ps_cam = PointStamped(); target_ps_cam.header.frame_id = self.camera_optical_frame_
                target_ps_cam.header.stamp = pc_msg.header.stamp
                target_ps_cam.point.x = best_3d_point_in_cam[0]; target_ps_cam.point.y = best_3d_point_in_cam[1]; target_ps_cam.point.z = best_3d_point_in_cam[2]

                try:
                    transform_cam_to_nav = self.tf_buffer_.lookup_transform(self.navigation_frame_, self.camera_optical_frame_, pc_msg.header.stamp, timeout=rclpyDuration(seconds=0.1))
                    target_ps_nav = do_transform_point(target_ps_cam, transform_cam_to_nav)
                    valid_candidate_points_map.append( (target_ps_nav.point, area) )
                    if self.point_display_mode_ == 1 or self.point_display_mode_ == 2: # Target only or all valid
                        u_disp, v_disp = int(best_3d_point_in_cam[3]), int(best_3d_point_in_cam[4])
                        cv2.circle(processed_image_for_display, (u_disp,v_disp), 7, (0,0,255),2) # Red circle for chosen point for this contour
                except (LookupException, ConnectivityException, ExtrapolationException) as e: self.get_logger().warn(f"TF lookup failed (CamOptical to NavFrame): {e}", throttle_duration_sec=1.0); continue
            
            if not valid_candidate_points_map:
                if self.segmentation_active_: self.publish_empty_goal("No valid contours with 3D points")
                self.last_published_crack_point_map_ = None
            else:
                if self.last_published_crack_point_map_ is not None:
                    candidates_passing_continuity = []
                    for cand_point_map, cand_area in valid_candidate_points_map:
                        dist_sq = (cand_point_map.x - self.last_published_crack_point_map_.x)**2 + (cand_point_map.y - self.last_published_crack_point_map_.y)**2
                        dist = math.sqrt(dist_sq)
                        if dist <= self.crack_continuity_thresh_: candidates_passing_continuity.append((cand_point_map, cand_area, dist))
                    
                    if not candidates_passing_continuity:
                        if self.segmentation_active_: self.publish_empty_goal("Continuity lost")
                        self.last_published_crack_point_map_ = None
                    else:
                        candidates_passing_continuity.sort(key=lambda item: item[2]) # Closest to last point
                        best_target_point_map = candidates_passing_continuity[0][0]
                else: # First point, or continuity lost previously
                    valid_candidate_points_map.sort(key=lambda item: item[1], reverse=True) # Largest area
                    best_target_point_map = valid_candidate_points_map[0][0]

        if best_target_point_map is not None:
            goal_pose = PoseStamped(); goal_pose.header.stamp = current_time.to_msg(); goal_pose.header.frame_id = self.navigation_frame_
            goal_pose.pose.position = best_target_point_map
            q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) 
            if self.last_published_crack_point_map_ and (self.last_published_crack_point_map_.x != best_target_point_map.x or self.last_published_crack_point_map_.y != best_target_point_map.y) :
                dx = best_target_point_map.x - self.last_published_crack_point_map_.x; dy = best_target_point_map.y - self.last_published_crack_point_map_.y
                yaw = math.atan2(dy, dx)
                cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
                q.w = cy; q.z = sy 
            goal_pose.pose.orientation = q
            self.goal_pub_.publish(goal_pose)
            self.get_logger().info(f"Published VALID goal to map ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})")
            self.last_published_crack_point_map_ = best_target_point_map
        # else: # No new target found, empty goal should have been published inside logic if needed
            # if self.segmentation_active_: self.get_logger().debug("No best_target_point_map.")

        if self.output_window_name_:
            if self.segmentation_active_ and best_target_point_map is None: cv2.putText(processed_image_for_display, "ACTIVE - NO TARGET", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255),2)
            elif self.segmentation_active_ and best_target_point_map is not None: cv2.putText(processed_image_for_display, "ACTIVE - TARGET FOUND", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.imshow(self.output_window_name_, processed_image_for_display); cv2.waitKey(1)

    def destroy_node(self):
        # ... (method content remains the same) ...
        self.get_logger().info(f"Destroying node '{self.get_name()}'...")
        if self.output_window_name_: cv2.destroyAllWindows()
        if self.segmentation_active_: self.publish_empty_goal("Node shutting down")
        super().destroy_node()
        self.get_logger().info(f"Node '{self.get_name()}' destroyed.")

def main(args=None):
    # ... (method content remains the same) ...
    rclpy.init(args=args)
    node = None
    try: node = ROILidarFusionNode(); rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info('Node interrupted by user (KeyboardInterrupt).')
    except SystemExit:
        if node: node.get_logger().info('Node is shutting down (SystemExit).')
    except Exception as e:
        err_node_name = node.get_name() if node else "ROILidarFusionNode (during init)"
        rclpy.logging.get_logger(err_node_name).error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok(): node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        print("ROILidarFusionNode main finally block finished.")

if __name__ == '__main__':
    main()