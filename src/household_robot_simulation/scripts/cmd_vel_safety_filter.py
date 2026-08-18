#!/usr/bin/env python3
"""
Laser Safety Stop / Obstacle Padding node for household service robot.

Subscribes to /cmd_vel_raw (teleop/nav commands) and /scan (LiDAR).
Overrides linear velocity to 0.0 if an obstacle is within the safety padding
in the direction of travel, while allowing rotation to escape corners.
Publishes safe commands to /cmd_vel.
Publishes a visualization marker representing the safety zone in RViz.
"""
import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker


class CmdVelSafetyFilter(Node):
    def __init__(self):
        super().__init__('cmd_vel_safety_filter')
        
        # Parameters
        self.declare_parameter('safety_padding', 0.35)  # 0.20m robot radius + 0.15m buffer
        self.safety_padding = self.get_parameter('safety_padding').value
        
        # State
        self.latest_scan = None
        self.safety_active = False
        
        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_raw', self.cmd_callback, 10)
            
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/safety_zone_marker', 10)
        
        # Timer to publish marker periodically
        self.marker_timer = self.create_timer(0.1, self.publish_marker)
        
        self.get_logger().info(f"Safety Filter started with padding: {self.safety_padding}m")

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def cmd_callback(self, msg: Twist):
        safe_msg = Twist()
        safe_msg.linear = msg.linear
        safe_msg.angular = msg.angular
        
        self.safety_active = False
        
        if self.latest_scan is None:
            self.cmd_pub.publish(safe_msg)
            return

        # Check front sector (-45 to +45 deg) if moving forward
        if msg.linear.x > 0.0:
            if self.is_obstacle_close(min_angle=-math.pi/4.0, max_angle=math.pi/4.0):
                safe_msg.linear.x = 0.0
                self.safety_active = True
                self.get_logger().warn("Forward obstacle detected! Safety stop active.", throttle_duration_sec=1.0)
                
        # Check rear sector (135 to 225 deg / -135 to -225 deg) if moving backward
        elif msg.linear.x < 0.0:
            if self.is_obstacle_close(min_angle=3.0*math.pi/4.0, max_angle=5.0*math.pi/4.0) or \
               self.is_obstacle_close(min_angle=-5.0*math.pi/4.0, max_angle=-3.0*math.pi/4.0):
                safe_msg.linear.x = 0.0
                self.safety_active = True
                self.get_logger().warn("Rear obstacle detected! Safety stop active.", throttle_duration_sec=1.0)

        self.cmd_pub.publish(safe_msg)

    def is_obstacle_close(self, min_angle, max_angle):
        scan = self.latest_scan
        angle = scan.angle_min
        
        for r in scan.ranges:
            if not math.isnan(r) and not math.isinf(r):
                norm_angle = math.atan2(math.sin(angle), math.cos(angle))
                
                in_bounds = False
                if min_angle <= max_angle:
                    in_bounds = min_angle <= norm_angle <= max_angle
                else:
                    in_bounds = norm_angle >= min_angle or norm_angle <= max_angle
                    
                if in_bounds and r < self.safety_padding:
                    return True
            angle += scan.angle_increment
            
        return False

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "safety_zone"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position flat on the ground
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.005
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Diameter is 2 * safety_padding
        diameter = 2.0 * self.safety_padding
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = 0.01  # Very thin cylinder disk
        
        # Red if safety active, else Green
        if self.safety_active:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.4  # Slightly more solid when active
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.15 # Soft alpha for idle state
            
        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSafetyFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
