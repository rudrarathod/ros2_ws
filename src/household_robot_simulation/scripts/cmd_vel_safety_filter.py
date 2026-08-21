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
import struct
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
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
        
        # Stuck detection states
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_pose_time = self.get_clock().now()
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_yaw = 0.0
        self.stuck_state = False
        self.stuck_start_time = None
        self.stuck_direction = 1.0  # 1.0 for forward, -1.0 for backward
        self.stuck_obstacles = []  # list of tuples: (x, y, timestamp)
        self.stuck_duration = 3.0  # duration to check if stuck
        self.stuck_threshold = 0.02  # distance threshold in meters
        self.is_commanding_movement = False
        self.has_valid_path = True
        self.last_plan_time = self.get_clock().now()
        
        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel_raw', self.cmd_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.plan_sub = self.create_subscription(
            Path, '/plan', self.plan_callback, 10)
            
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stuck_pub = self.create_publisher(PointCloud2, '/stuck_obstacles', 10)
        self.marker_pub = self.create_publisher(Marker, '/safety_zone_marker', 10)
        
        # Timers
        self.marker_timer = self.create_timer(0.1, self.publish_marker)
        self.stuck_timer = self.create_timer(1.0, self.check_stuck_timer_callback)
        
        self.get_logger().info(f"Safety Filter started with padding: {self.safety_padding}m")

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def plan_callback(self, msg: Path):
        self.last_plan_time = self.get_clock().now()
        if len(msg.poses) == 0:
            self.has_valid_path = False
            return

        # Check if the path intersects with any active stuck obstacles
        has_intersection = False
        now = self.get_clock().now()
        active_obstacles = [obs for obs in self.stuck_obstacles 
                            if (now - obs[2]).nanoseconds / 1e9 < 20.0]
        
        for pose in msg.poses:
            px = pose.pose.position.x
            py = pose.pose.position.y
            for ox, oy, _ in active_obstacles:
                dist = math.sqrt((px - ox)**2 + (py - oy)**2)
                if dist < 0.30:  # 30cm intersection threshold
                    has_intersection = True
                    break
            if has_intersection:
                break
                
        self.has_valid_path = not has_intersection
        
        # If we have a valid path and were backing up, stop backing up immediately!
        if self.has_valid_path and self.stuck_state:
            self.stuck_state = False
            self.get_logger().info("Found a valid path avoiding the stuck obstacle! Resuming navigation.")

    def check_stuck_timer_callback(self):
        now = self.get_clock().now()
        
        # Age out old stuck obstacles (keep for 20 seconds)
        self.stuck_obstacles = [obs for obs in self.stuck_obstacles 
                                if (now - obs[2]).nanoseconds / 1e9 < 20.0]
        
        if not self.is_commanding_movement or self.stuck_state:
            # Not commanding movement or already executing backup recovery
            self.last_pose_x = self.robot_x
            self.last_pose_y = self.robot_y
            self.last_pose_yaw = self.robot_yaw
            self.last_pose_time = now
            self.publish_stuck_obstacles()
            return
            
        # Check if stuck
        dx = self.robot_x - self.last_pose_x
        dy = self.robot_y - self.last_pose_y
        dist = math.sqrt(dx*dx + dy*dy)
        dt = (now - self.last_pose_time).nanoseconds / 1e9
        
        if dt >= self.stuck_duration:
            if dist < self.stuck_threshold:
                # Detected stuck!
                self.stuck_state = True
                self.stuck_start_time = now
                self.has_valid_path = False  # Reset path state
                
                # Stuck position (0.35m in front of or behind the robot)
                obs_x = self.robot_x + self.stuck_direction * 0.35 * math.cos(self.robot_yaw)
                obs_y = self.robot_y + self.stuck_direction * 0.35 * math.sin(self.robot_yaw)
                
                self.stuck_obstacles.append((obs_x, obs_y, now))
                self.get_logger().warn(
                    f"Robot is stuck! Marking obstacle at ({obs_x:.2f}, {obs_y:.2f}) and backing up..."
                )
                
            # Reset tracking
            self.last_pose_x = self.robot_x
            self.last_pose_y = self.robot_y
            self.last_pose_yaw = self.robot_yaw
            self.last_pose_time = now
            
        self.publish_stuck_obstacles()

    def publish_stuck_obstacles(self):
        msg = PointCloud2()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = 1
        msg.is_bigendian = False
        msg.point_step = 12
        msg.is_dense = True
        
        points = []
        for x, y, _ in self.stuck_obstacles:
            points.append((x, y, 0.1))
            points.append((x + 0.05, y, 0.1))
            points.append((x - 0.05, y, 0.1))
            points.append((x, y + 0.05, 0.1))
            points.append((x, y - 0.05, 0.1))
            
        msg.width = len(points)
        msg.row_step = msg.point_step * msg.width
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        buffer = bytearray()
        for p in points:
            buffer.extend(struct.pack('fff', p[0], p[1], p[2]))
        msg.data = bytes(buffer)
        
        self.stuck_pub.publish(msg)

    def cmd_callback(self, msg: Twist):
        # Track whether Nav2 is commanding translation
        self.is_commanding_movement = (abs(msg.linear.x) > 0.01)
        if self.is_commanding_movement:
            self.stuck_direction = 1.0 if msg.linear.x > 0.0 else -1.0

        safe_msg = Twist()
        self.safety_active = False
        
        # 1. Stuck recovery state: Override command to back up safely
        if self.stuck_state:
            now = self.get_clock().now()
            elapsed = (now - self.stuck_start_time).nanoseconds / 1e9
            
            # Back up for at least 2.0 seconds, and continue up to 15.0 seconds if no valid path is found yet
            time_to_backup = (elapsed < 2.0) or (not self.has_valid_path and elapsed < 15.0)
            
            if time_to_backup:
                # Back up in the opposite direction of what got us stuck
                safe_msg.linear.x = -0.15 * self.stuck_direction
                safe_msg.angular.z = 0.0
                
                # Check for safety zone in the backup direction
                if safe_msg.linear.x > 0.0:
                    if self.latest_scan is not None and self.is_obstacle_close(min_angle=-math.pi/4.0, max_angle=math.pi/4.0):
                        safe_msg.linear.x = 0.0
                        self.safety_active = True
                elif safe_msg.linear.x < 0.0:
                    if self.latest_scan is not None and (self.is_obstacle_close(min_angle=3.0*math.pi/4.0, max_angle=5.0*math.pi/4.0) or \
                       self.is_obstacle_close(min_angle=-5.0*math.pi/4.0, max_angle=-3.0*math.pi/4.0)):
                        safe_msg.linear.x = 0.0
                        self.safety_active = True
                        
                self.cmd_pub.publish(safe_msg)
                return
            else:
                self.stuck_state = False
                self.get_logger().info("Stuck recovery complete or timed out, handing control back to Nav2.")

        # 2. Normal safety filter logic
        safe_msg.linear = msg.linear
        safe_msg.angular = msg.angular
        
        if self.latest_scan is not None:
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
