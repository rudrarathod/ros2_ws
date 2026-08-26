#!/usr/bin/env python3
"""
ROS 2 Node for Module 10: Household Robot Behaviors & Module 11: Monitoring.
Subscribes to:
  - /voice/command (String): High-level voice commands.
  - /cmd_vel (Twist): Monitor robot movement for battery simulation.
  - /yolo/detections (String): Vision detections for patrol intruder / hazard alerts.

Behaviors:
  1. Room Navigation: "go to <room>" (kitchen, bedroom, living room, etc.)
  2. Virtual Item Delivery: "bring water" / "bring medicine" (retrieves from source and delivers to room)
  3. Home Security Patrol: "start patrol" / "stop patrol" (loops through waypoints, triggers intruder alarm)
  4. Person Following: "follow me" / "stop" (toggles /person_follower node)
  5. Charging Station Return: "dock" / "go charge" / auto-dock on low battery (<= 20%)
  6. Emergency Alert: sounds /emergency/alarm and displays 3D RViz alert marker
  7. Battery Simulation: publishes /battery/percentage (Float32) and 3D status marker
"""

import math
import os
import json
import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
import tf2_ros
from ament_index_python.packages import get_package_share_directory


class VoiceInterpreter(Node):
    def __init__(self):
        super().__init__('voice_interpreter')

        # TF2 Setup to query current robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Load semantic locations & patrol waypoints from YAML database
        self.locations = {}
        self.patrol_waypoints = []
        self.load_database()

        # Navigation & Patrol States
        self.active_nav_target = None
        self.patrol_mode = False
        self.current_patrol_index = 0
        self.patrol_dwell_timer = None
        self.is_dwelling = False

        # Delivery Task States
        self.delivery_item = None
        self.delivery_source = None
        self.delivery_destination = None
        self.delivery_stage = None  # "GO_TO_SOURCE", "PICKING_UP", "GO_TO_DESTINATION"
        self.pickup_timer = None

        # Inventory
        self.inventory = {
            'water bottle': 3,
            'medicine': 2
        }

        # Battery Simulation States
        self.battery_level = 100.0
        self.is_charging = False
        self.low_battery_triggered = False
        self.docking_active = False
        self.robot_is_moving = False

        # Emergency Alert States
        self.emergency_active = False
        self.emergency_reason = ""

        # Service client for person follower
        self.param_client = self.create_client(SetParameters, '/person_follower/set_parameters')

        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_goal_handle = None

        # Subscribers
        self.cmd_sub = self.create_subscription(
            String, '/voice/command', self.voice_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.yolo_sub = self.create_subscription(
            String, '/yolo/detections', self.yolo_callback, 10)

        # Publishers
        self.battery_pub = self.create_publisher(Float32, '/battery/percentage', 10)
        self.marker_pub = self.create_publisher(Marker, '/battery/marker', 10)
        self.alarm_pub = self.create_publisher(Bool, '/emergency/alarm', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)

        # 1Hz Battery simulation timer
        self.battery_timer = self.create_timer(1.0, self.simulate_battery)

        self.get_logger().info("Voice Command Interpreter & Behavior Coordinator Initialized.")

    def load_database(self):
        """Load semantic rooms and patrol waypoints from YAML."""
        try:
            pkg_share = get_package_share_directory('household_robot_simulation')
            yaml_path = os.path.join(pkg_share, 'config', 'semantic_locations.yaml')
            self.yaml_path = os.path.realpath(yaml_path)
            with open(self.yaml_path, 'r') as f:
                config = yaml.safe_load(f) or {}

            for name, coords in config.get('locations', {}).items():
                self.locations[name.strip().lower()] = (float(coords['x']), float(coords['y']), float(coords['yaw']))

            for wp in config.get('patrol_waypoints', []):
                self.patrol_waypoints.append((float(wp['x']), float(wp['y']), float(wp['yaw'])))

            self.get_logger().info(
                f"Loaded {len(self.locations)} rooms and {len(self.patrol_waypoints)} patrol waypoints."
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to load semantic database ({e}). Using default coordinates.")
            self.locations = {
                'kitchen': (2.16, 2.71, 1.56),
                'bedroom': (-3.10, 0.48, 3.09),
                'living_room': (3.76, -2.58, 0.29),
                'kitchen_counter': (4.0, 2.5, 0.0),
                'medicine_cabinet': (1.0, -3.0, -1.57),
                'charging_station': (0.0, 0.0, 0.0),
                'start': (0.0, 0.0, 0.0)
            }
            self.patrol_waypoints = [
                (3.76, -2.58, 0.29),
                (2.16, 2.71, 1.56),
                (-3.10, 0.48, 3.09)
            ]

    def cmd_vel_callback(self, msg: Twist):
        self.robot_is_moving = (abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01)

    def yolo_callback(self, msg: String):
        """Monitor vision detections for hazards or intruders during security patrol."""
        if self.emergency_active:
            return

        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        for d in detections:
            cls = d.get('class', '').lower()
            conf = d.get('confidence', 0.0)

            # Security Patrol Feature: Intruder detection
            if self.patrol_mode and cls in ["person", "human", "pedestrian"]:
                self.stop_patrol(reason="Intruder Person Detected")
                self.trigger_emergency_alarm(f"INTRUDER DETECTED! Unknown person spotted during patrol (Conf: {conf:.2f})")
                return

            # Safety Hazard Detection: Dangerous objects
            if cls in ["knife", "scissors"]:
                self.stop_patrol(reason=f"Hazardous Item ({cls})")
                self.trigger_emergency_alarm(f"Hazardous Item Detected: {cls}!")
                return

    def trigger_emergency_alarm(self, reason: str):
        if self.emergency_active:
            return

        self.emergency_active = True
        self.emergency_reason = reason
        self.get_logger().error(f"🚨 EMERGENCY ALERT: {reason}")

        # Stop all tasks and brake robot
        self.stop_patrol(reason="Emergency Alarm")
        self.delivery_stage = None
        self.active_nav_target = None
        self.set_follower_mode(False)
        self.cancel_nav2_goal()

        brake_msg = Twist()
        self.cmd_pub.publish(brake_msg)

        alarm_msg = Bool()
        alarm_msg.data = True
        self.alarm_pub.publish(alarm_msg)

        self.publish_emergency_marker()

    def publish_emergency_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "emergency"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 0.95
        marker.pose.orientation.w = 1.0
        marker.text = f"🚨 {self.emergency_reason.upper()} 🚨"
        marker.scale.z = 0.16
        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.1
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def simulate_battery(self):
        if self.is_charging:
            self.battery_level = min(100.0, self.battery_level + 5.0)
            if self.battery_level >= 100.0:
                self.is_charging = False
                self.low_battery_triggered = False
                self.get_logger().info("Battery fully charged (100%)! Ready for tasks.")
        else:
            decay = 0.3 if self.robot_is_moving else 0.08
            self.battery_level = max(0.0, self.battery_level - decay)

            if self.battery_level <= 20.0 and not self.low_battery_triggered:
                self.low_battery_triggered = True
                self.trigger_autonomous_docking()

        battery_msg = Float32()
        battery_msg.data = self.battery_level
        self.battery_pub.publish(battery_msg)
        self.publish_battery_marker()

    def publish_battery_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "battery"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 0.6
        marker.pose.orientation.w = 1.0

        state = "Charging" if self.is_charging else "Discharging"
        marker.text = f"Battery: {self.battery_level:.1f}% ({state})"
        marker.scale.z = 0.12

        if self.battery_level > 50.0:
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif self.battery_level > 20.0:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
        else:
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def trigger_autonomous_docking(self):
        self.get_logger().warn(f"Battery Critical ({self.battery_level:.1f}%)! Returning to charging station.")
        self.stop_patrol(reason="Low Battery")
        self.delivery_stage = None
        self.set_follower_mode(False)
        self.cancel_nav2_goal()

        self.docking_active = True
        self.active_nav_target = "charging_station"
        x, y, yaw = self.locations.get('charging_station', (0.0, 0.0, 0.0))
        self.send_nav2_goal(x, y, yaw)

    def set_follower_mode(self, enable: bool):
        if not self.param_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("Service /person_follower/set_parameters not available.")
            return

        req = SetParameters.Request()
        param = Parameter()
        param.name = 'follow_mode'
        param.value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=enable)
        req.parameters = [param]
        self.param_client.call_async(req)

    # =========================================================================
    # NAVIGATION & BEHAVIORS
    # =========================================================================
    def navigate_to_room(self, room_name: str):
        """Dispatch Nav2 goal to named room location."""
        room_key = room_name.strip().lower().replace(" ", "_")
        matched = None

        if room_key in self.locations:
            matched = room_key
        else:
            for key in self.locations:
                if key in room_key or room_key in key:
                    matched = key
                    break

        if matched is None:
            rooms = ", ".join(self.locations.keys())
            self.get_logger().warn(f"Room '{room_name}' not found. Available rooms: {rooms}")
            return False

        self.stop_patrol(reason="Room Navigation Requested")
        self.set_follower_mode(False)
        self.delivery_stage = None

        x, y, yaw = self.locations[matched]
        self.active_nav_target = matched
        self.get_logger().info(f"Navigating to room '{matched}' at x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad")
        self.send_nav2_goal(x, y, yaw)
        return True

    def start_patrol(self):
        """Start security patrol loop across waypoints."""
        if not self.patrol_waypoints:
            self.get_logger().warn("No patrol waypoints found.")
            return False

        self.set_follower_mode(False)
        self.delivery_stage = None
        self.patrol_mode = True
        self.current_patrol_index = 0
        self.get_logger().info(f"🛡️ Security Patrol activated across {len(self.patrol_waypoints)} waypoints.")
        self.send_next_patrol_waypoint()
        return True

    def stop_patrol(self, reason: str = "User Command"):
        if self.patrol_mode or self.is_dwelling:
            self.patrol_mode = False
            self.is_dwelling = False
            if self.patrol_dwell_timer is not None:
                self.patrol_dwell_timer.cancel()
                self.patrol_dwell_timer = None
            self.get_logger().info(f"🛡️ Security Patrol stopped ({reason}).")

    def send_next_patrol_waypoint(self):
        if not self.patrol_mode or not self.patrol_waypoints:
            return
        total = len(self.patrol_waypoints)
        x, y, yaw = self.patrol_waypoints[self.current_patrol_index]
        self.active_nav_target = f"Patrol Waypoint #{self.current_patrol_index + 1}"
        self.get_logger().info(f"🛡️ Patrol moving to Waypoint [{self.current_patrol_index + 1}/{total}] (x={x:.2f}, y={y:.2f})")
        self.send_nav2_goal(x, y, yaw)

    def patrol_dwell_complete(self):
        if self.patrol_dwell_timer is not None:
            self.patrol_dwell_timer.cancel()
            self.patrol_dwell_timer = None
        self.is_dwelling = False

        if self.patrol_mode:
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_waypoints)
            self.send_next_patrol_waypoint()

    # =========================================================================
    # VOICE COMMAND HANDLER
    # =========================================================================
    def voice_callback(self, msg: String):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received voice command: '{command}'")

        # 1. Emergency Alarm Clear
        if any(kw in command for kw in ["clear alarm", "reset alarm", "all clear", "cancel alarm"]):
            if self.emergency_active:
                self.emergency_active = False
                self.emergency_reason = ""
                alarm_msg = Bool()
                alarm_msg.data = False
                self.alarm_pub.publish(alarm_msg)

                # Remove marker
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "emergency"
                marker.id = 1
                marker.action = Marker.DELETE
                self.marker_pub.publish(marker)
                self.get_logger().info("Emergency alarm cleared. Resuming normal operation.")
            return

        if self.emergency_active:
            self.get_logger().warn(f"Action refused: Emergency Alarm is ACTIVE ({self.emergency_reason}). Clear alarm first.")
            return

        # 2. Stop / Brake
        if any(kw in command for kw in ["stop", "halt", "stay", "brake"]):
            self.stop_patrol(reason="Stop Command")
            self.cancel_nav2_goal()
            self.set_follower_mode(False)
            self.active_nav_target = None
            self.delivery_stage = None
            brake_msg = Twist()
            self.cmd_pub.publish(brake_msg)
            self.get_logger().info("Robot stopped.")
            return

        # 3. Person Follow
        elif any(kw in command for kw in ["follow me", "come here", "track me", "start following"]):
            self.stop_patrol(reason="Follow Mode Requested")
            self.cancel_nav2_goal()
            self.delivery_stage = None
            self.set_follower_mode(True)
            self.get_logger().info("Person Following Mode enabled.")
            return

        # 4. Security Patrol
        elif any(kw in command for kw in ["stop patrol", "cancel patrol", "end patrol"]):
            self.stop_patrol(reason="Voice Command")
            self.cancel_nav2_goal()
            return

        elif any(kw in command for kw in ["start patrol", "patrol house", "patrol mode", "patrol"]):
            self.start_patrol()
            return

        # 5. Charging Station Return
        elif any(kw in command for kw in ["go charge", "dock", "return to charger", "recharge"]):
            self.get_logger().info("Returning to charging station.")
            self.stop_patrol(reason="Docking Requested")
            self.delivery_stage = None
            self.set_follower_mode(False)
            self.cancel_nav2_goal()
            self.docking_active = True
            self.active_nav_target = "charging_station"
            x, y, yaw = self.locations.get('charging_station', (0.0, 0.0, 0.0))
            self.send_nav2_goal(x, y, yaw)
            return

        # 6. Battery Status
        elif any(kw in command for kw in ["battery status", "battery level", "check battery"]):
            state_str = "Charging" if self.is_charging else "Discharging"
            self.get_logger().info(f"Battery Status: {self.battery_level:.1f}% ({state_str})")
            return

        # 7. Virtual Item Delivery (Medicine / Water)
        elif any(kw in command for kw in ["deliver", "bring", "get", "fetch"]):
            self.stop_patrol(reason="Delivery Task")
            self.set_follower_mode(False)
            self.cancel_nav2_goal()

            item = "water bottle" if "water" in command else "medicine"
            source = "kitchen_counter" if item == "water bottle" else "medicine_cabinet"

            if self.inventory.get(item, 0) <= 0:
                self.get_logger().error(f"Item '{item}' is out of stock.")
                return

            destination = "bedroom" if item == "medicine" else "living_room"
            for room in ["kitchen", "bedroom", "living_room", "living room"]:
                if room in command:
                    destination = room.replace(" ", "_")
                    break

            self.delivery_item = item
            self.delivery_source = source
            self.delivery_destination = destination
            self.delivery_stage = "GO_TO_SOURCE"
            self.active_nav_target = source

            x, y, yaw = self.locations[source]
            self.get_logger().info(f"Delivery: Fetching {item} from {source} and delivering to {destination}.")
            self.send_nav2_goal(x, y, yaw)
            return

        # 8. Room Navigation
        elif any(command.startswith(prefix) for prefix in [
            "go to", "navigate to", "travel to", "move to", "head to", "drive to"
        ]) or any(r.replace("_", " ") in command or r in command for r in self.locations):
            # Extract target room
            target = command
            for prefix in ["go to the ", "go to ", "navigate to the ", "navigate to ", "move to the ", "move to ", "head to the ", "head to "]:
                if target.startswith(prefix):
                    target = target[len(prefix):].strip()
                    break
            self.navigate_to_room(target)
            return

        else:
            self.get_logger().warn(f"Unrecognized voice command: '{command}'")

    # =========================================================================
    # NAV2 ACTION DISPATCH & RESULTS
    # =========================================================================
    def send_nav2_goal(self, x: float, y: float, yaw: float):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 action server ('navigate_to_pose') not available.")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Nav2 Dispatch: Sending goal to ({x:.2f}, {y:.2f})")
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("Nav2 Goal rejected by server.")
                return
            self.current_goal_handle = goal_handle
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(lambda f: self.get_result_callback(f, goal_handle))
        except Exception as e:
            self.get_logger().error(f"Nav2 Goal response error: {e}")

    def get_result_callback(self, future, goal_handle):
        try:
            if goal_handle != self.current_goal_handle:
                return
            result = future.result()
            status = result.status
            self.current_goal_handle = None

            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"Arrived at destination: '{self.active_nav_target}'!")

                # Handle Delivery Stages
                if self.delivery_stage == "GO_TO_SOURCE":
                    self.delivery_stage = "PICKING_UP"
                    self.get_logger().info(f"Picking up {self.delivery_item}... (Waiting 3s)")
                    self.pickup_timer = self.create_timer(3.0, self.pickup_complete)
                    return

                elif self.delivery_stage == "GO_TO_DESTINATION":
                    self.delivery_stage = None
                    self.inventory[self.delivery_item] = max(0, self.inventory[self.delivery_item] - 1)
                    self.get_logger().info(
                        f"Delivered {self.delivery_item} to {self.delivery_destination}! "
                        f"Remaining stock: {self.inventory[self.delivery_item]}"
                    )
                    return

                # Handle Patrol Dwell
                if self.patrol_mode:
                    self.is_dwelling = True
                    self.get_logger().info(f"🛡️ Scanning waypoint {self.current_patrol_index + 1}... (Scanning 4s)")
                    self.patrol_dwell_timer = self.create_timer(4.0, self.patrol_dwell_complete)
                    return

                # Handle Charging Docking
                if self.docking_active or self.active_nav_target == "charging_station":
                    self.is_charging = True
                    self.docking_active = False
                    self.get_logger().info("Docked at charging station! Battery charging initiated.")
                    return

            else:
                self.get_logger().warn(f"Navigation to '{self.active_nav_target}' finished with status: {status}")
        except Exception as e:
            self.get_logger().error(f"Result callback error: {e}")

    def pickup_complete(self):
        if self.pickup_timer is not None:
            self.pickup_timer.cancel()
            self.pickup_timer = None

        if self.delivery_stage == "PICKING_UP":
            self.delivery_stage = "GO_TO_DESTINATION"
            self.active_nav_target = self.delivery_destination
            x, y, yaw = self.locations[self.delivery_destination]
            self.get_logger().info(f"Item retrieved! Delivering to {self.delivery_destination}...")
            self.send_nav2_goal(x, y, yaw)

    def cancel_nav2_goal(self):
        if self.current_goal_handle is not None:
            self.get_logger().info("Canceling active Nav2 goal.")
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = VoiceInterpreter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
