#!/usr/bin/env python3
"""
ROS 2 Node for Module 9: AI Person Following.
Subscribes to:
  - /yolo/detections (String): Detected bounding boxes from camera.
  - /scan (LaserScan): 2D LiDAR for distance estimation and obstacle avoidance.
  - /camera/image_raw (Image): Camera resolution metadata.

Publishes:
  - /cmd_vel_raw (Twist): Commanded robot velocities.
  - /person_follower/status (String): Status string ("DISABLED", "FOLLOWING", "SEARCHING").

Features:
  - Dynamically enabled/disabled via ROS 2 parameter 'follow_mode'.
  - Computes target bearing from YOLO bounding box centroid.
  - Computes target distance using LiDAR scan in the target bearing cone.
  - Smoothly drives toward target while repelling away from nearby LiDAR obstacles.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
import json
import math


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')

        # Parameters
        self.declare_parameter('follow_mode', False)
        self.declare_parameter('target_distance', 0.9)        # Desired following distance (m)
        self.declare_parameter('max_linear_speed', 0.25)      # Maximum linear speed (m/s)
        self.declare_parameter('max_angular_speed', 0.6)      # Maximum turning speed (rad/s)
        self.declare_parameter('Kp_linear', 0.5)              # Proportional gain for linear velocity
        self.declare_parameter('Kp_angular', 1.2)             # Proportional gain for angular velocity
        self.declare_parameter('search_angular_speed', 0.3)   # Search rotation speed (rad/s)
        self.declare_parameter('lost_timeout', 2.0)           # Time before switching to SEARCHING

        self.follow_mode = self.get_parameter('follow_mode').value
        self.target_distance = self.get_parameter('target_distance').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.Kp_linear = self.get_parameter('Kp_linear').value
        self.Kp_angular = self.get_parameter('Kp_angular').value
        self.search_angular_speed = self.get_parameter('search_angular_speed').value
        self.lost_timeout = self.get_parameter('lost_timeout').value

        # States
        self.image_width = 640
        self.image_height = 480
        self.last_person_time = None
        self.state = "DISABLED" if not self.follow_mode else "SEARCHING"
        self.last_scan = None

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

        # 10Hz Control Loop & 2Hz Status Timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)

        self.get_logger().info("Person Follower Node Initialized (Module 9).")

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'follow_mode':
                self.follow_mode = param.value
                self.state = "SEARCHING" if self.follow_mode else "DISABLED"
                self.get_logger().info(f"Follow mode set to: {self.follow_mode}")
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
        except Exception:
            return

        # Find largest person detection
        best_person = None
        max_area = 0

        for det in detections:
            if det.get('class') in ['person', 'human', 'pedestrian']:
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

    def estimate_distance(self, target_angle):
        if self.last_scan is None:
            return None

        ranges = self.last_scan.ranges
        angle_min = self.last_scan.angle_min
        angle_inc = self.last_scan.angle_increment
        window = 0.12  # ~7 degree cone

        valid = []
        for i, dist in enumerate(ranges):
            if dist < self.last_scan.range_min or dist > self.last_scan.range_max or math.isinf(dist) or math.isnan(dist):
                continue
            angle = angle_min + i * angle_inc
            if abs(angle - target_angle) < window:
                valid.append(dist)

        if not valid:
            return None
        valid.sort()
        return valid[len(valid) // 2]  # Median

    def track_person(self, person_det):
        bbox = person_det.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox

        # Centroid horizontal error normalized to [-1, 1]
        cx = (x1 + x2) / 2.0
        error_x = (cx - (self.image_width / 2.0)) / (self.image_width / 2.0)
        target_bearing = -error_x * 0.5  # radians

        # Estimate distance
        dist = self.estimate_distance(target_bearing)
        if dist is None:
            # Fallback estimation based on bounding box height
            h = y2 - y1
            dist = max(0.5, 300.0 / max(1.0, float(h)))

        # Compute angular velocity (P control)
        w = self.Kp_angular * target_bearing
        w = max(-self.max_angular_speed, min(self.max_angular_speed, w))

        # Compute linear velocity (P control on distance error)
        dist_error = dist - self.target_distance
        if abs(dist_error) < 0.1:
            v = 0.0
        else:
            v = self.Kp_linear * dist_error
            v = max(-0.08, min(self.max_linear_speed, v))

        # Local obstacle avoidance repulsion
        if self.last_scan is not None and v > 0:
            ranges = self.last_scan.ranges
            angle_min = self.last_scan.angle_min
            angle_inc = self.last_scan.angle_increment
            front_min = float('inf')

            for i, r in enumerate(ranges):
                if r < 0.1 or r > 2.0 or math.isinf(r) or math.isnan(r):
                    continue
                angle = angle_min + i * angle_inc
                if abs(angle) < 0.35:  # Front cone
                    if r < front_min:
                        front_min = r

            # If front obstacle is closer than 0.45m, stop or slow down
            if front_min < 0.45:
                v = min(v, 0.0)

        cmd_msg = Twist()
        cmd_msg.linear.x = v
        cmd_msg.angular.z = w
        self.cmd_pub.publish(cmd_msg)

    def control_loop(self):
        if not self.follow_mode:
            cmd_msg = Twist()
            self.cmd_pub.publish(cmd_msg)
            self.state = "DISABLED"
            return

        if self.state == "SEARCHING":
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.angular.z = self.search_angular_speed
            self.cmd_pub.publish(cmd_msg)

    def publish_status(self):
        status_msg = String()
        status_msg.data = self.state
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
