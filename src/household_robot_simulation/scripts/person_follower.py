#!/usr/bin/env python3
"""
ROS 2 Node for Person Following.
Subscribes to /yolo/detections and /camera/image_raw.
Controls the robot's velocity to follow the closest detected person.
Publishes velocity commands to /cmd_vel_raw (filtered by safety filter).
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import json

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

        # Get initial parameter values
        self.follow_mode = self.get_parameter('follow_mode').value
        self.target_area_fraction = self.get_parameter('target_area_fraction').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.Kp_linear = self.get_parameter('Kp_linear').value
        self.Kp_angular = self.get_parameter('Kp_angular').value
        self.search_angular_speed = self.get_parameter('search_angular_speed').value
        self.lost_timeout = self.get_parameter('lost_timeout').value

        # State variables
        self.image_width = 640  # Default fallback width
        self.image_height = 480 # Default fallback height
        self.last_person_time = None
        self.state = "DISABLED" if not self.follow_mode else "SEARCHING"

        # Parameter callback
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Subscriptions
        self.yolo_sub = self.create_subscription(
            String, '/yolo/detections', self.yolo_callback, 10)
        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self.status_pub = self.create_publisher(String, '/person_follower/status', 10)

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)

        self.get_logger().info("Person Follower node initialized.")

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
        return rclpy.node.SetParametersResult(successful=True)

    def image_callback(self, msg: Image):
        # Update image dimensions dynamically
        self.image_width = msg.width
        self.image_height = msg.height

    def yolo_callback(self, msg: String):
        if not self.follow_mode:
            return

        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse YOLO detections: {e}")
            return

        # Find the person detection with the largest bounding box area (assumed closest)
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
            # Check if we timed out waiting for a detection
            now = self.get_clock().now()
            if self.last_person_time is not None:
                elapsed = (now - self.last_person_time).nanoseconds / 1e9
                if elapsed > self.lost_timeout:
                    self.state = "SEARCHING"
            else:
                self.state = "SEARCHING"

    def track_person(self, person_det):
        bbox = person_det.get('bbox', [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        
        # Calculate bounding box centroid
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Calculate tracking errors
        # X error: deviation from center of frame (-1.0 to 1.0)
        error_x = (cx - (self.image_width / 2.0)) / (self.image_width / 2.0)
        
        # Area fraction: target distance indicator (0.0 to 1.0)
        bbox_area = float((x2 - x1) * (y2 - y1))
        total_area = float(self.image_width * self.image_height)
        area_fraction = bbox_area / total_area

        # Distance error: proportional to the difference in area fraction
        error_dist = 1.0 - (area_fraction / self.target_area_fraction)

        # Proportional controller output
        angular_vel = -self.Kp_angular * error_x
        linear_vel = self.Kp_linear * error_dist

        # Apply maximum speed limits (clamping)
        angular_vel = max(-self.max_angular_speed, min(self.max_angular_speed, angular_vel))
        
        # Allow backing off slowly if too close (clamped at -0.1 m/s)
        min_linear = -0.10
        linear_vel = max(min_linear, min(self.max_linear_speed, linear_vel))

        # Command publication message
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_vel
        cmd_msg.angular.z = angular_vel
        self.cmd_pub.publish(cmd_msg)

    def control_loop(self):
        # Handle state actions when no active detections are coming in
        if not self.follow_mode:
            cmd_msg = Twist()
            self.cmd_pub.publish(cmd_msg)
            self.state = "DISABLED"
            return

        if self.state == "SEARCHING":
            # Slowly rotate in place to scan for person
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
