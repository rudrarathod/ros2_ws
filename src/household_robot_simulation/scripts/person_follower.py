#!/usr/bin/env python3
"""
ROS 2 Node for Person Following with Hybrid Nav2 and Obstacle Avoidance.
Subscribes to /yolo/detections, /camera/image_raw, and /scan (LiDAR).
If Nav2 and Map TF are active, it estimates the person's map position and dispatches Nav2 goals.
If Nav2 is offline, it falls back to the reactive potential field (APF) controller.
If the person is lost in Nav2 mode, the robot continues to the last known position.
Upon arrival at the destination, if the person is still not found, it rotates in place to search.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
import json
import math
import tf2_ros

try:
    import tf2_geometry_msgs
except ImportError:
    pass

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')

        # Declare parameters for easy tuning
        self.declare_parameter('follow_mode', True)
        self.declare_parameter('target_area_fraction', 0.08)  # Desired bbox area fraction of image (e.g., 8%)
        self.declare_parameter('max_linear_speed', 0.25)      # Maximum forward/backward speed (m/s)
        self.declare_parameter('max_angular_speed', 0.6)      # Maximum turning speed (rad/s)
        self.declare_parameter('Kp_linear', 0.6)              # Proportional gain for linear velocity
        self.declare_parameter('Kp_angular', 1.2)             # Proportional gain for angular velocity
        self.declare_parameter('search_angular_speed', 0.3)   # Search rotation speed (rad/s)
        self.declare_parameter('lost_timeout', 2.0)           # Time in seconds before declaring person lost
        self.declare_parameter('avoidance_distance', 0.55)    # Distance threshold to start avoiding obstacles (m)
        self.declare_parameter('Kp_avoidance', 1.0)           # Proportional gain for obstacle repulsion force

        # Get initial parameter values
        self.follow_mode = self.get_parameter('follow_mode').value
        self.target_area_fraction = self.get_parameter('target_area_fraction').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.Kp_linear = self.get_parameter('Kp_linear').value
        self.Kp_angular = self.get_parameter('Kp_angular').value
        self.search_angular_speed = self.get_parameter('search_angular_speed').value
        self.lost_timeout = self.get_parameter('lost_timeout').value
        self.avoidance_distance = self.get_parameter('avoidance_distance').value
        self.Kp_avoidance = self.get_parameter('Kp_avoidance').value

        # State variables
        self.image_width = 640  # Default fallback width
        self.image_height = 480 # Default fallback height
        self.last_person_time = None
        self.state = "DISABLED" if not self.follow_mode else "SEARCHING"
        self.last_scan = None

        # Mode variables
        self.in_backup_mode = True
        self.last_goal_sent_time = None
        self.last_sent_x = None
        self.last_sent_y = None
        self.current_goal_handle = None

        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Nav2 Action Client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Subscriptions
        self.yolo_sub = self.create_subscription(
            String, '/yolo/detections', self.yolo_callback, 10)
        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self.status_pub = self.create_publisher(String, '/person_follower/status', 10)

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)

        self.get_logger().info("Person Follower Node Initialized (Hybrid Nav2 + APF Fallback).")

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'follow_mode':
                self.follow_mode = param.value
                self.state = "SEARCHING" if self.follow_mode else "DISABLED"
                self.get_logger().info(f"Follow mode changed to: {self.follow_mode}")
            elif param.name == 'target_area_fraction':
                self.target_area_fraction = param.value
            elif param.name == 'max_linear_speed':
                self.max_linear_speed = param.value
            elif param.name == 'max_angular_speed':
                self.max_angular_speed = param.value
            elif param.name == 'Kp_linear':
                self.Kp_linear = param.value
            elif param.name == 'Kp_angular':
                self.Kp_angular = param.value
            elif param.name == 'search_angular_speed':
                self.search_angular_speed = param.value
            elif param.name == 'lost_timeout':
                self.lost_timeout = param.value
            elif param.name == 'avoidance_distance':
                self.avoidance_distance = param.value
            elif param.name == 'Kp_avoidance':
                self.Kp_avoidance = param.value
        return rclpy.node.SetParametersResult(successful=True)

    def image_callback(self, msg: Image):
        self.image_width = msg.width
        self.image_height = msg.height

    def scan_callback(self, msg: LaserScan):
        self.last_scan = msg

    def yolo_callback(self, msg: String):
        if not self.follow_mode:
            return

        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse YOLO detections: {e}")
            return

        # Find closest person
        best_person = None
        max_area = 0

        for det in detections:
            if det.get('class') == 'person':
                bbox = det.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_person = det

        if best_person is not None:
            self.last_person_time = self.get_clock().now()
            self.state = "FOLLOWING"
            self.track_person(best_person)
        else:
            now = self.get_clock().now()
            if self.last_person_time is not None:
                elapsed = (now - self.last_person_time).nanoseconds / 1e9
                if elapsed > self.lost_timeout:
                    self.state = "SEARCHING"
            else:
                self.state = "SEARCHING"

    def estimate_target_distance(self, target_angle):
        if self.last_scan is None:
            return None
        
        ranges = self.last_scan.ranges
        angle_min = self.last_scan.angle_min
        angle_increment = self.last_scan.angle_increment

        best_dist = None
        window_half_width = 0.26  # ~15 degrees search window
        
        for i, dist in enumerate(ranges):
            if dist < self.last_scan.range_min or dist > self.last_scan.range_max or math.isinf(dist) or math.isnan(dist):
                continue
            
            angle = angle_min + i * angle_increment
            if abs(angle - target_angle) < window_half_width:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
        
        return best_dist

    def track_person(self, person_det):
        bbox = person_det.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        
        cx = (x1 + x2) / 2.0
        error_x = (cx - (self.image_width / 2.0)) / (self.image_width / 2.0)

        # Estimate bearing angle
        target_angle_camera = -error_x * 0.5

        # Attempt to get distance using LiDAR
        d = self.estimate_target_distance(target_angle_camera)

        # Try to use Nav2 if distance and transforms are available
        nav2_success = False
        if d is not None:
            x_rel = d * math.cos(target_angle_camera)
            y_rel = d * math.sin(target_angle_camera)

            # Create PoseStamped
            pose_local = PoseStamped()
            pose_local.header.frame_id = 'base_footprint'
            pose_local.header.stamp = rclpy.time.Time().to_msg()
            pose_local.pose.position.x = x_rel
            pose_local.pose.position.y = y_rel
            pose_local.pose.orientation.w = 1.0

            try:
                # Transform to map frame
                pose_map = self.tf_buffer.transform(pose_local, 'map', timeout=rclpy.duration.Duration(seconds=0.1))
                
                # Check if Nav2 action server is available
                if self.nav_client.wait_for_server(timeout_sec=0.01):
                    nav2_success = True
                    self.in_backup_mode = False
                    
                    # Rate-limit sending goals to Nav2 (once every 1.0s or if moved > 0.2m)
                    now = self.get_clock().now()
                    time_since_last = (now - self.last_goal_sent_time).nanoseconds / 1e9 if self.last_goal_sent_time else float('inf')
                    
                    dist_moved = float('inf')
                    if self.last_sent_x is not None and self.last_sent_y is not None:
                        dist_moved = math.hypot(pose_map.pose.position.x - self.last_sent_x, pose_map.pose.position.y - self.last_sent_y)

                    if time_since_last > 1.0 or dist_moved > 0.2:
                        goal_msg = NavigateToPose.Goal()
                        goal_msg.pose = pose_map
                        self.get_logger().info(f"[Nav2 Mode] Routing to x={pose_map.pose.position.x:.2f}, y={pose_map.pose.position.y:.2f}")
                        send_goal_future = self.nav_client.send_goal_async(goal_msg)
                        send_goal_future.add_done_callback(self.goal_response_callback)
                        self.last_goal_sent_time = now
                        self.last_sent_x = pose_map.pose.position.x
                        self.last_sent_y = pose_map.pose.position.y
            except Exception as e:
                # TF transform or other Nav2 step failed, will fall back to backup
                self.get_logger().warn(f"Nav2 transform/goal step failed: {e}", throttle_duration_sec=2.0)

        if not nav2_success:
            # Fall back to Backup Mode: reactive APF direct velocity control
            if not self.in_backup_mode:
                self.get_logger().info("Nav2 or Map TF unavailable. Entering Backup Mode (reactive APF).")
                self.in_backup_mode = True

            # Calculate area fraction for local controller distance error
            bbox_area = float((x2 - x1) * (y2 - y1))
            total_area = float(self.image_width * self.image_height)
            area_fraction = bbox_area / total_area
            error_dist = 1.0 - (area_fraction / self.target_area_fraction)

            # Potential Field Calculation
            Fx_attract = math.cos(target_angle_camera)
            Fy_attract = math.sin(target_angle_camera)

            Fx_repulse = 0.0
            Fy_repulse = 0.0

            if self.last_scan is not None:
                ranges = self.last_scan.ranges
                angle_min = self.last_scan.angle_min
                angle_increment = self.last_scan.angle_increment

                for i, dist in enumerate(ranges):
                    if dist < self.last_scan.range_min or dist > self.last_scan.range_max or math.isinf(dist) or math.isnan(dist):
                        continue
                    
                    angle = angle_min + i * angle_increment
                    if -0.8 < angle < 0.8:  # ±45 degrees
                        if dist < self.avoidance_distance:
                            weight = (self.avoidance_distance - dist) / self.avoidance_distance
                            Fx_repulse -= (weight / (dist * dist)) * math.cos(angle)
                            Fy_repulse -= (weight / (dist * dist)) * math.sin(angle)

            Fx_total = Fx_attract + self.Kp_avoidance * Fx_repulse
            Fy_total = Fy_attract + self.Kp_avoidance * Fy_repulse
            target_angle = math.atan2(Fy_total, Fx_total)

            # Steering
            angular_vel = self.Kp_angular * target_angle

            # Critical Front Stop Check
            is_front_blocked = False
            if self.last_scan is not None:
                angle_min = self.last_scan.angle_min
                angle_increment = self.last_scan.angle_increment
                for i, dist in enumerate(self.last_scan.ranges):
                    if dist < self.last_scan.range_min or dist > self.last_scan.range_max or math.isinf(dist) or math.isnan(dist):
                        continue
                    angle = angle_min + i * angle_increment
                    if -0.26 < angle < 0.26 and dist < 0.28:
                        is_front_blocked = True
                        break

            if is_front_blocked:
                linear_vel = 0.0
            else:
                alignment_factor = math.cos(target_angle)
                speed_multiplier = max(0.25, alignment_factor)
                linear_vel = self.Kp_linear * error_dist * speed_multiplier

            # Clamp velocities
            angular_vel = max(-self.max_angular_speed, min(self.max_angular_speed, angular_vel))
            linear_vel = max(-0.10, min(self.max_linear_speed, linear_vel))

            # Publish backup cmd_vel
            cmd_msg = Twist()
            cmd_msg.linear.x = linear_vel
            cmd_msg.angular.z = angular_vel
            self.cmd_pub.publish(cmd_msg)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().info("Goal rejected by Nav2 server")
                return
            self.current_goal_handle = goal_handle
        except Exception as e:
            self.get_logger().error(f"Goal response callback failed: {e}")

    def cancel_nav2_goal(self):
        if self.current_goal_handle is not None:
            self.get_logger().info("Canceling active Nav2 goal to rotate and search")
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None

    def control_loop(self):
        if not self.follow_mode:
            cmd_msg = Twist()
            self.cmd_pub.publish(cmd_msg)
            self.state = "DISABLED"
            return

        if self.state == "SEARCHING":
            should_rotate_search = False
            
            if self.in_backup_mode:
                should_rotate_search = True
            else:
                # Nav2 Mode: Check if we have reached the last known target location
                if self.last_sent_x is not None and self.last_sent_y is not None:
                    try:
                        trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
                        rx = trans.transform.translation.x
                        ry = trans.transform.translation.y
                        
                        dist_to_goal = math.hypot(self.last_sent_x - rx, self.last_sent_y - ry)
                        if dist_to_goal < 0.45:
                            # Reached destination, cancel goal and start search rotation
                            self.cancel_nav2_goal()
                            should_rotate_search = True
                    except Exception:
                        should_rotate_search = True
                else:
                    should_rotate_search = True

            if should_rotate_search:
                cmd_msg = Twist()
                cmd_msg.linear.x = 0.0
                cmd_msg.angular.z = self.search_angular_speed
                self.cmd_pub.publish(cmd_msg)

    def publish_status(self):
        status_msg = String()
        mode_str = "NAV2" if not self.in_backup_mode else "BACKUP_APF"
        status_msg.data = f"{self.state} ({mode_str})"
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
