#!/usr/bin/env python3
"""
Corridor-Aware Safety Stop & Intelligent Autonomous Backtracking Recovery Node.

Features:
1. Multi-Path & Corridor Safety Verification:
   - Uses Cartesian bounding-box projection (X: [0.15m, 0.28m], |Y| <= 0.16m) matching physical
     robot geometry to avoid false stops against side doorframes in narrow passages.
2. Intelligent 3-Second Stall Detection & Automatic Backtrack Recovery:
   - Monitors odometry progress while movement is actively commanded.
   - If robot is stalled (< 3cm displacement) for 3.0 seconds, automatically triggers backtrack recovery:
     reverses safely with angular diversion for 2.0 seconds to escape tight spots and prompts Nav2 replanning.
3. Publishes clean, safe commands to /cmd_vel and safety zone visualization to RViz.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


class CmdVelSafetyFilter(Node):
    def __init__(self):
        super().__init__('cmd_vel_safety_filter')

        # Parameters
        self.declare_parameter('corridor_forward_dist', 0.28)  # Front stop distance (m)
        self.declare_parameter('corridor_half_width', 0.16)     # Half-width matching 0.18m radius (m)
        self.declare_parameter('stall_duration', 3.0)           # 3-second stall timeout
        self.declare_parameter('stall_threshold', 0.03)         # 3cm minimum expected movement

        self.corridor_forward_dist = self.get_parameter('corridor_forward_dist').value
        self.corridor_half_width = self.get_parameter('corridor_half_width').value
        self.stall_duration = self.get_parameter('stall_duration').value
        self.stall_threshold = self.get_parameter('stall_threshold').value

        # Sensor & State Variables
        self.latest_scan = None
        self.safety_active = False

        # Odometry & Stall Detection States
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_pose_time = self.get_clock().now()
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0

        self.stuck_state = False
        self.stuck_start_time = None
        self.stuck_direction = 1.0
        self.backtrack_sign = 1.0
        self.is_commanding_movement = False

        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_raw', self.cmd_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/safety_zone_marker', 10)

        # Timers
        self.marker_timer = self.create_timer(0.1, self.publish_marker)
        self.stuck_timer = self.create_timer(0.5, self.check_stall_timer_callback)

        self.get_logger().info("Intelligent Corridor Safety Filter & 3s Backtrack Recovery Initialized.")

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def check_stall_timer_callback(self):
        """Monitors robot odometry. If stalled for 3.0 seconds, initiates autonomous backtrack."""
        now = self.get_clock().now()

        if not self.is_commanding_movement or self.stuck_state:
            self.last_pose_x = self.robot_x
            self.last_pose_y = self.robot_y
            self.last_pose_time = now
            return

        dx = self.robot_x - self.last_pose_x
        dy = self.robot_y - self.last_pose_y
        dist = math.sqrt(dx * dx + dy * dy)
        dt = (now - self.last_pose_time).nanoseconds / 1e9

        if dt >= self.stall_duration:
            if dist < self.stall_threshold:
                # 3-Second Stall Detected!
                self.stuck_state = True
                self.stuck_start_time = now
                self.backtrack_sign = 1.0 if (int(now.nanoseconds / 1e9) % 2 == 0) else -1.0
                self.get_logger().warn(
                    f"⚠️ [STALL DETECTED] Robot has made < {dist*100:.1f}cm progress over {dt:.1f}s! "
                    f"Executing autonomous backtrack recovery to clear path for Nav2 replanning..."
                )
            # Reset checkpoint
            self.last_pose_x = self.robot_x
            self.last_pose_y = self.robot_y
            self.last_pose_time = now

    def cmd_callback(self, msg: Twist):
        self.is_commanding_movement = (abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.05)
        if abs(msg.linear.x) > 0.01:
            self.stuck_direction = 1.0 if msg.linear.x > 0.0 else -1.0

        safe_msg = Twist()
        self.safety_active = False

        # 1. Autonomous Backtracking Recovery Routine (Runs for 2.0s upon stall)
        if self.stuck_state:
            now = self.get_clock().now()
            elapsed = (now - self.stuck_start_time).nanoseconds / 1e9

            if elapsed < 2.0:
                # Backtrack in reverse direction with a subtle pivot angle to break alignment traps
                safe_msg.linear.x = -0.16 * self.stuck_direction
                safe_msg.angular.z = 0.25 * self.backtrack_sign

                # Safety check behind before moving backwards
                if safe_msg.linear.x < 0.0 and self.is_obstacle_in_corridor(forward=False):
                    safe_msg.linear.x = 0.0
                    safe_msg.angular.z = 0.35 * self.backtrack_sign  # Spin only if reverse blocked

                self.cmd_pub.publish(safe_msg)
                return
            else:
                self.stuck_state = False
                self.get_logger().info("✅ Backtrack recovery complete! Handing control back to Nav2 for path replan.")

        # 2. Corridor-Aware Safety Filter
        safe_msg.linear = msg.linear
        safe_msg.angular = msg.angular

        if self.latest_scan is not None:
            if msg.linear.x > 0.01:
                # Forward motion corridor check
                if self.is_obstacle_in_corridor(forward=True):
                    safe_msg.linear.x = 0.0
                    self.safety_active = True
                    self.get_logger().warn("Obstacle in immediate forward path! Safety brake applied.", throttle_duration_sec=1.5)

            elif msg.linear.x < -0.01:
                # Reverse motion corridor check
                if self.is_obstacle_in_corridor(forward=False):
                    safe_msg.linear.x = 0.0
                    self.safety_active = True
                    self.get_logger().warn("Obstacle in immediate rear path! Safety brake applied.", throttle_duration_sec=1.5)

        self.cmd_pub.publish(safe_msg)

    def is_obstacle_in_corridor(self, forward=True):
        """
        Cartesian corridor collision check.
        Checks if any point from LaserScan falls inside the vehicle width corridor.
        """
        if self.latest_scan is None:
            return False

        scan = self.latest_scan
        angle = scan.angle_min

        for r in scan.ranges:
            if not math.isnan(r) and not math.isinf(r) and r > 0.05:
                # Convert polar (r, theta) to Cartesian (x, y) relative to robot center
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                # Check bounding corridor
                if forward:
                    # Forward box: X in [0.15m, 0.28m], |Y| <= 0.16m
                    if 0.15 <= x <= self.corridor_forward_dist and abs(y) <= self.corridor_half_width:
                        return True
                else:
                    # Reverse box: X in [-0.28m, -0.15m], |Y| <= 0.16m
                    if -self.corridor_forward_dist <= x <= -0.15 and abs(y) <= self.corridor_half_width:
                        return True

            angle += scan.angle_increment

        return False

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "corridor_safety_zone"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Corridor box visualization (0.28m long, 0.32m wide)
        marker.pose.position.x = self.corridor_forward_dist / 2.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.02
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.corridor_forward_dist
        marker.scale.y = 2.0 * self.corridor_half_width
        marker.scale.z = 0.02

        if self.stuck_state:
            # Orange for active backtracking recovery
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.6
        elif self.safety_active:
            # Red for obstacle brake
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
        else:
            # Soft green for normal clear corridor
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.2
            marker.color.a = 0.18

        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSafetyFilter()
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
