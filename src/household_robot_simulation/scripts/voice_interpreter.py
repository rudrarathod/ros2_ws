#!/usr/bin/env python3
"""
ROS 2 Node for Voice Command Interpretation, Task Dispatching, Semantic Navigation, and Security Patrol.
Subscribes to /voice/command.
Translates voice/text commands to autonomous robot actions:
- "go to <room>" / "navigate to <room>" -> Navigate to saved semantic location.
- "patrol" / "start patrol" / "watch the house" -> Start security patrol loop through saved waypoints.
- "stop patrol" -> Stop patrol loop.
- "follow me" / "come here" -> Enable person-following mode.
- "stop" / "halt" / "stay" -> Disable all active tasks and brake robot.
- "save location <name>" / "delete location <name>" / "list locations" -> Manage locations.
- Vision System: When in Patrol Mode, detects person/intruder and sounds emergency alarm.
"""

import math
import os
import re
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

        # TF2 Setup to query current robot position for saving locations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Load semantic locations and patrol waypoints from YAML configuration database
        self.locations = {}
        self.patrol_waypoints = []
        try:
            pkg_share = get_package_share_directory('household_robot_simulation')
            yaml_path = os.path.join(pkg_share, 'config', 'semantic_locations.yaml')
            self.yaml_path = os.path.realpath(yaml_path)
            with open(self.yaml_path, 'r') as f:
                config = yaml.safe_load(f) or {}

            # Parse target locations
            for name, coords in config.get('locations', {}).items():
                self.locations[name.strip().lower()] = (float(coords['x']), float(coords['y']), float(coords['yaw']))

            # Parse saved patrol waypoints
            for wp in config.get('patrol_waypoints', []):
                self.patrol_waypoints.append((float(wp['x']), float(wp['y']), float(wp['yaw'])))

            self.get_logger().info(
                f"Successfully loaded {len(self.locations)} semantic locations "
                f"and {len(self.patrol_waypoints)} patrol waypoints from database."
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load semantic database: {e}")
            self.locations = {
                'kitchen': (2.16, 2.71, 1.56),
                'bedroom': (-3.10, 0.48, 3.09),
                'living_room': (3.76, -2.58, 0.29),
                'charging_station': (0.0, 0.0, 0.0),
                'start': (0.0, 0.0, 0.0)
            }
            self.patrol_waypoints = [
                (3.76, -2.58, 0.29),
                (2.12, 2.69, 1.58),
                (-3.10, 0.48, 3.09)
            ]

        # Ensure charging station exists in target locations database
        if 'charging_station' not in self.locations:
            self.locations['charging_station'] = (0.0, 0.0, 0.0)

        # Navigation & Patrol States
        self.active_nav_target = None
        self.patrol_mode = False
        self.current_patrol_index = 0
        self.patrol_dwell_timer = None
        self.is_dwelling = False

        # Delivery task states
        self.delivery_item = None
        self.delivery_source = None
        self.delivery_destination = None
        self.delivery_stage = None  # "GO_TO_SOURCE", "PICKING_UP", "GO_TO_DESTINATION"
        self.pickup_timer = None

        # Item counter / inventory database
        self.inventory = {
            'water bottle': 3,
            'medicine': 2
        }

        # Battery Simulation & Docking States
        self.battery_level = 100.0
        self.is_charging = False
        self.low_battery_triggered = False
        self.docking_active = False
        self.robot_is_moving = False

        # Emergency Alert States
        self.emergency_active = False
        self.emergency_reason = ""

        # Service client for setting follower parameters
        self.param_client = self.create_client(SetParameters, '/person_follower/set_parameters')

        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_goal_handle = None
        self.get_result_future = None

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

        self.get_logger().info("Voice Command Interpreter & Security Navigation Node Initialized.")

    def cmd_vel_callback(self, msg: Twist):
        self.robot_is_moving = (abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01)

    def yolo_callback(self, msg: String):
        """Monitor detections from vision processor for hazards and patrol intruders."""
        if self.is_charging or self.emergency_active:
            return

        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        for d in detections:
            cls = d.get('class', '').lower()
            conf = d.get('confidence', 0.0)

            # Security Patrol Feature: Intruder alert when person detected in patrol mode
            if self.patrol_mode and cls in ["person", "human", "pedestrian"]:
                self.get_logger().error(
                    f"🚨 [SECURITY PATROL] INTRUDER DETECTED! Class: '{cls}', Confidence: {conf:.2f}"
                )
                self.stop_patrol(reason="Intruder Person Detected")
                self.trigger_emergency_alarm(f"INTRUDER DETECTED: Unknown Person Spotted during Patrol! (Conf: {conf:.2f})")
                return

            # Safety Hazard Detection: Dangerous items (knife, scissors)
            if cls in ["knife", "scissors"]:
                self.stop_patrol(reason=f"Hazardous Item Detected ({cls})")
                self.trigger_emergency_alarm(f"Hazardous Item Detected: {cls}!")
                return

    def trigger_emergency_alarm(self, reason: str):
        if self.emergency_active:
            return

        self.emergency_active = True
        self.emergency_reason = reason

        self.get_logger().error(f"!!! EMERGENCY ALERT !!! {reason} - Sounding alarms and halting robot.")

        # Stop the robot immediately
        self.stop_patrol(reason="Emergency Alarm")
        self.delivery_stage = None
        self.active_nav_target = None
        self.set_follower_mode(False)
        self.cancel_nav2_goal()

        # Send active zero velocity brake
        brake_msg = Twist()
        self.cmd_pub.publish(brake_msg)

        # Sound alarm topic
        alarm_msg = Bool()
        alarm_msg.data = True
        self.alarm_pub.publish(alarm_msg)

        # Publish 3D visual warning marker
        self.publish_emergency_marker()

    def publish_emergency_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "emergency"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # Position floating 0.95m above robot base
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.95
        marker.pose.orientation.w = 1.0

        marker.text = f"🚨 {self.emergency_reason.upper()} 🚨"
        marker.scale.z = 0.16

        # Red text
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
                self.get_logger().info("Battery fully charged (100.0%)! Ready for new commands.")
        else:
            decay = 0.4 if self.robot_is_moving else 0.1
            self.battery_level = max(0.0, self.battery_level - decay)

            if self.battery_level <= 30.0 and int(self.battery_level) % 5 == 0 and abs(self.battery_level - int(self.battery_level)) < 0.1:
                self.get_logger().warn(f"Low Battery Alert: {self.battery_level:.1f}% Remaining.")

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

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.6
        marker.pose.orientation.w = 1.0

        state = "Charging" if self.is_charging else "Discharging"
        marker.text = f"Battery: {self.battery_level:.1f}% ({state})"
        marker.scale.z = 0.12

        if self.battery_level > 50.0:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif self.battery_level > 20.0:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    def trigger_autonomous_docking(self):
        self.get_logger().warn(f"Battery Critical ({self.battery_level:.1f}%)! Aborting tasks and returning to charging station.")
        self.stop_patrol(reason="Low Battery")
        self.delivery_stage = None
        self.active_nav_target = "charging_station"
        self.set_follower_mode(False)
        self.cancel_nav2_goal()

        self.docking_active = True
        x, y, yaw = self.locations['charging_station']
        self.send_nav2_goal(x, y, yaw)

    def set_follower_mode(self, enable: bool):
        if not self.param_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("SetParameters service for /person_follower not available.")
            return

        req = SetParameters.Request()
        param = Parameter()
        param.name = 'follow_mode'
        param.value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=enable)
        req.parameters = [param]

        self.get_logger().info(f"Sending request to set follow_mode = {enable}")
        self.param_client.call_async(req)

    def get_current_pose(self, target_frame='map', base_frame='base_footprint'):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                base_frame,
                rclpy.time.Time()
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation

            siny_cosp = 2.0 * (rot.w * rot.z + rot.x * rot.y)
            cosy_cosp = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return round(trans.x, 3), round(trans.y, 3), round(yaw, 3)
        except Exception:
            if target_frame == 'map':
                return self.get_current_pose(target_frame='odom', base_frame=base_frame)
            return None

    def persist_database(self):
        """Save in-memory locations back to YAML, preserving existing patrol_waypoints."""
        try:
            data = {}
            if os.path.exists(self.yaml_path):
                with open(self.yaml_path, 'r') as f:
                    data = yaml.safe_load(f) or {}

            data['locations'] = {
                name: {'x': float(c[0]), 'y': float(c[1]), 'yaw': float(c[2])}
                for name, c in self.locations.items()
            }
            with open(self.yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f"Database successfully updated on disk ({self.yaml_path}).")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to write database: {e}")
            return False

    def handle_save_location(self, raw_name: str):
        name = raw_name.strip()
        for prefix in ['location as ', 'location ', 'room as ', 'room ', 'spot as ', 'spot ', 'here as ']:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
                break
        name = name.strip().lower().replace(" ", "_")
        if not name:
            self.get_logger().warn("Save Location Failed: No room name provided.")
            return

        pose = self.get_current_pose()
        if pose is None:
            self.get_logger().error("Save Location Failed: Could not get robot pose from TF.")
            return

        x, y, yaw = pose
        self.locations[name] = (x, y, yaw)
        self.persist_database()
        deg = round(math.degrees(yaw), 1)
        self.get_logger().info(f"Action: Saved new room location '{name}' at x={x}, y={y}, yaw={yaw} rad ({deg}°)")

    def handle_delete_location(self, raw_name: str):
        name = raw_name.strip().lower()
        for prefix in ['location ', 'room ', 'spot ']:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
                break
        name_key = name.replace(" ", "_")
        if name_key in self.locations:
            del self.locations[name_key]
            self.persist_database()
            self.get_logger().info(f"Action: Successfully deleted location '{name_key}'.")
        elif name in self.locations:
            del self.locations[name]
            self.persist_database()
            self.get_logger().info(f"Action: Successfully deleted location '{name}'.")
        else:
            self.get_logger().warn(f"Delete Location Failed: Location '{name}' not found in database.")

    def handle_list_locations(self):
        self.get_logger().info(f"--- Mapped Semantic Rooms ({len(self.locations)}) ---")
        for name, coords in self.locations.items():
            deg = round(math.degrees(coords[2]), 1)
            self.get_logger().info(f"  • {name:20s}: x={coords[0]:6.2f}, y={coords[1]:6.2f}, yaw={coords[2]:5.2f} rad ({deg:5.1f}°)")
        self.get_logger().info(f"--- Saved Patrol Waypoints ({len(self.patrol_waypoints)}) ---")
        for i, wp in enumerate(self.patrol_waypoints, start=1):
            deg = round(math.degrees(wp[2]), 1)
            self.get_logger().info(f"  [{i}] Waypoint #{i:02d}        : x={wp[0]:6.2f}, y={wp[1]:6.2f}, yaw={wp[2]:5.2f} rad ({deg:5.1f}°)")

    # =========================================================================
    # NAVIGATION SYSTEM
    # =========================================================================
    def navigate_to_room(self, raw_destination: str):
        """Intelligently parse destination room name and dispatch Nav2 goal."""
        dest = raw_destination.strip().lower()
        # Strip common action prefixes
        for prefix in [
            "go to room ", "go to the room ", "navigate to room ", "navigate to the room ",
            "travel to the room ", "travel to room ", "move to the room ", "move to room ",
            "take me to the room ", "take me to room ", "go to the ", "go to ", "navigate to the ",
            "navigate to ", "travel to the ", "travel to ", "move to the ", "move to ", "head to the ",
            "head to ", "drive to the ", "drive to ", "take me to the ", "take me to "
        ]:
            if dest.startswith(prefix):
                dest = dest[len(prefix):].strip()
                break

        # Remove leading "the " if still present
        if dest.startswith("the "):
            dest = dest[4:].strip()

        # Normalize spaces/underscores
        clean_dest = dest.replace(" ", "_")
        space_dest = dest.replace("_", " ")

        matched_room = None
        if clean_dest in self.locations:
            matched_room = clean_dest
        elif space_dest in self.locations:
            matched_room = space_dest
        else:
            # Check normalized keys
            for key in self.locations.keys():
                key_clean = key.replace("_", " ")
                if (key == clean_dest or key == space_dest or
                    key_clean == clean_dest or key_clean == space_dest or
                    key in clean_dest or key_clean in space_dest or
                    clean_dest in key or space_dest in key_clean):
                    matched_room = key
                    break

        if matched_room is None:
            available = ", ".join([f"'{k}'" for k in self.locations.keys()])
            self.get_logger().warn(f"Navigation target '{raw_destination}' not recognized. Available rooms: {available}")
            return False

        # Stop patrol mode and following
        self.stop_patrol(reason="Room Navigation Requested")
        self.set_follower_mode(False)
        self.delivery_stage = None

        x, y, yaw = self.locations[matched_room]
        deg = round(math.degrees(yaw), 1)
        self.active_nav_target = matched_room
        self.get_logger().info(f"Action: Navigating to room '{matched_room}' at (x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad / {deg}°)...")
        self.send_nav2_goal(x, y, yaw)
        return True

    # =========================================================================
    # SECURITY PATROL SYSTEM
    # =========================================================================
    def start_patrol(self):
        """Initiate automated security patrol through saved waypoints."""
        if not self.patrol_waypoints:
            self.get_logger().warn("Patrol Start Failed: No patrol waypoints found in database! Save waypoints first using save_location.py.")
            return False

        self.set_follower_mode(False)
        self.delivery_stage = None
        self.patrol_mode = True
        self.current_patrol_index = 0

        self.get_logger().info(
            f"🛡️ [SECURITY PATROL] Activated! Initiating patrol cycle across {len(self.patrol_waypoints)} waypoints. "
            f"Live vision intruder detection is ACTIVE."
        )
        self.send_next_patrol_waypoint()
        return True

    def stop_patrol(self, reason: str = "User command"):
        """Stop active patrol loop."""
        if self.patrol_mode or self.is_dwelling:
            self.patrol_mode = False
            self.is_dwelling = False
            if self.patrol_dwell_timer is not None:
                self.patrol_dwell_timer.cancel()
                self.patrol_dwell_timer = None
            self.get_logger().info(f"🛡️ [SECURITY PATROL] Deactivated ({reason}).")

    def send_next_patrol_waypoint(self):
        """Dispatch navigation goal to next waypoint in patrol cycle."""
        if not self.patrol_mode or not self.patrol_waypoints:
            return

        total_wps = len(self.patrol_waypoints)
        x, y, yaw = self.patrol_waypoints[self.current_patrol_index]
        deg = round(math.degrees(yaw), 1)
        self.active_nav_target = f"Patrol Waypoint #{self.current_patrol_index + 1}"

        self.get_logger().info(
            f"🛡️ [SECURITY PATROL] Moving to Waypoint [{self.current_patrol_index + 1}/{total_wps}]: "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad ({deg}°)"
        )
        self.send_nav2_goal(x, y, yaw)

    def patrol_dwell_complete(self):
        """Called when dwell/scan period at a waypoint finishes."""
        if self.patrol_dwell_timer is not None:
            self.patrol_dwell_timer.cancel()
            self.patrol_dwell_timer = None
        self.is_dwelling = False

        if self.patrol_mode:
            # Advance to next waypoint in loop
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_waypoints)
            self.send_next_patrol_waypoint()

    # =========================================================================
    # VOICE COMMAND HANDLER
    # =========================================================================
    def voice_callback(self, msg: String):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received voice command: '{command}'")

        # 1. Location Management Commands
        if any(command.startswith(prefix) for prefix in ["save location", "save room", "record room", "record location", "set location", "set room"]):
            for prefix in ["save location", "save room", "record room", "record location", "set location", "set room"]:
                if command.startswith(prefix):
                    room_name = command[len(prefix):].strip()
                    self.handle_save_location(room_name)
                    return

        elif any(command.startswith(prefix) for prefix in ["delete location", "delete room", "remove location", "remove room"]):
            for prefix in ["delete location", "delete room", "remove location", "remove room"]:
                if command.startswith(prefix):
                    room_name = command[len(prefix):].strip()
                    self.handle_delete_location(room_name)
                    return

        elif any(kw in command for kw in ["list locations", "show locations", "list rooms", "show rooms", "list waypoints"]):
            self.handle_list_locations()
            return

        # 2. Emergency Alarm Clear Commands
        is_clear_cmd = any(kw in command for kw in ["clear alarm", "reset alarm", "cancel alarm", "all clear", "clear emergency"])
        if is_clear_cmd:
            if self.emergency_active:
                self.emergency_active = False
                self.emergency_reason = ""

                alarm_msg = Bool()
                alarm_msg.data = False
                self.alarm_pub.publish(alarm_msg)

                # Delete emergency visual marker
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "emergency"
                marker.id = 1
                marker.action = Marker.DELETE
                self.marker_pub.publish(marker)

                self.get_logger().info("Action: Emergency alarm successfully cleared. Resuming normal operations.")
            else:
                self.get_logger().info("No active emergency alarm to clear.")
            return

        # Refuse execution if emergency alarm is active
        if self.emergency_active:
            self.get_logger().warn(f"Action Refused: Emergency Alarm is ACTIVE ({self.emergency_reason}). Please clear the alarm first!")
            return

        # Battery low guard
        is_battery_cmd = any(kw in command for kw in ["battery", "charge", "dock", "recharge"])
        if self.battery_level < 10.0 and not is_battery_cmd:
            self.get_logger().error(f"Action Refused: Battery level too low ({self.battery_level:.1f}%). Please charge the robot.")
            return

        # Interrupt charging session if new active command given
        if self.is_charging and not is_battery_cmd:
            self.is_charging = False
            self.low_battery_triggered = False
            self.get_logger().info(f"Undocking: Interrupting charging session at {self.battery_level:.1f}% to execute new command.")

        # 3. Stop / Brake Commands
        if any(kw in command for kw in ["stop", "halt", "stay", "brake"]):
            self.stop_patrol(reason="Stop Command Received")
            self.cancel_nav2_goal()
            self.set_follower_mode(False)
            self.active_nav_target = None
            self.delivery_stage = None

            brake_msg = Twist()
            self.cmd_pub.publish(brake_msg)
            self.get_logger().info("Action: Stopping robot movement immediately.")
            return

        # 4. Person Follow Commands
        elif any(kw in command for kw in ["follow me", "come here", "track me", "start following"]):
            self.stop_patrol(reason="Follow Mode Requested")
            self.cancel_nav2_goal()
            self.set_follower_mode(True)
            self.get_logger().info("Action: Enabling Person Following Mode.")
            return

        # 5. Security Patrol Commands
        elif any(kw in command for kw in ["stop patrol", "cancel patrol", "end patrol", "pause patrol", "halt patrol"]):
            self.stop_patrol(reason="User voice command")
            self.cancel_nav2_goal()
            return

        elif any(kw in command for kw in ["start patrol", "patrol house", "begin patrol", "security patrol", "patrol mode", "patrol"]):
            self.start_patrol()
            return

        # 6. Battery Status Commands
        elif any(kw in command for kw in ["battery status", "battery level", "check battery"]):
            state_str = "Charging" if self.is_charging else "Discharging"
            self.get_logger().info(f"Battery Status: {self.battery_level:.1f}% ({state_str})")
            return

        # 7. Manual Charging / Docking
        elif any(kw in command for kw in ["go charge", "dock", "return to charger", "recharge"]):
            self.get_logger().info("Action: Manual command received. Returning to charging station.")
            self.stop_patrol(reason="Docking Requested")
            self.delivery_stage = None
            self.set_follower_mode(False)
            self.cancel_nav2_goal()

            self.docking_active = True
            self.active_nav_target = "charging_station"
            x, y, yaw = self.locations['charging_station']
            self.send_nav2_goal(x, y, yaw)
            return

        # 8. Delivery Tasks
        elif any(kw in command for kw in ["deliver", "bring", "get", "fetch"]):
            self.stop_patrol(reason="Delivery Task Requested")
            self.set_follower_mode(False)
            self.cancel_nav2_goal()

            item = "water bottle" if "water" in command else "medicine"
            source = "kitchen_counter" if item == "water bottle" else "medicine_cabinet"

            if self.inventory[item] <= 0:
                self.get_logger().error(f"Action Aborted: '{item}' is OUT OF STOCK! Please restock it first.")
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
            self.get_logger().info(f"Action: Starting delivery. Retrieving {item} from {source} and delivering to {destination}.")
            self.send_nav2_goal(x, y, yaw)
            return

        # 9. Restocking Tasks
        elif any(kw in command for kw in ["restock", "refill"]):
            self.stop_patrol(reason="Restock Task Requested")
            self.set_follower_mode(False)
            self.cancel_nav2_goal()

            if "water" in command:
                self.inventory['water bottle'] = 3
                self.get_logger().info("Action: Kitchen counter restocked to 3 water bottles.")
            elif "medicine" in command:
                self.inventory['medicine'] = 2
                self.get_logger().info("Action: Medicine cabinet restocked to 2 units.")
            else:
                self.inventory['water bottle'] = 3
                self.inventory['medicine'] = 2
                self.get_logger().info("Action: All item storage counters successfully restocked.")
            return

        # 10. Room / Point Navigation Commands
        elif any(command.startswith(prefix) for prefix in [
            "go to", "navigate to", "travel to", "move to", "head to", "drive to", "take me to"
        ]) or any(room.replace("_", " ") in command or room in command for room in self.locations.keys()):
            self.navigate_to_room(command)
            return

        else:
            self.get_logger().warn(f"Unrecognized voice command phrase: '{command}'")

    # =========================================================================
    # NAV2 ACTION CLIENT & DISPATCH
    # =========================================================================
    def send_nav2_goal(self, x: float, y: float, yaw: float):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Nav2 action server ('navigate_to_pose') is not available.")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        # Calculate quaternion from yaw
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Nav2 Dispatch: Sending goal x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad")
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn("Nav2 Dispatch: Goal rejected by server.")
                return
            self.current_goal_handle = goal_handle

            def result_cb(result_future):
                self.get_result_callback(result_future, goal_handle)

            self.get_result_future = goal_handle.get_result_async()
            self.get_result_future.add_done_callback(result_cb)
        except Exception as e:
            self.get_logger().error(f"Goal response callback error: {e}")

    def get_result_callback(self, future, goal_handle):
        try:
            if goal_handle != self.current_goal_handle:
                return

            result = future.result()
            status = result.status
            self.get_logger().info(f"Nav2 Action Finished with status: {status}")
            self.current_goal_handle = None

            if status == GoalStatus.STATUS_SUCCEEDED:
                # Docking completion
                if self.docking_active:
                    self.docking_active = False
                    self.is_charging = True
                    self.active_nav_target = None
                    self.get_logger().info("✅ Arrived at Charging Station. Successfully docked! Charging started...")

                # Delivery Stage 1: Pickup
                elif self.delivery_stage == "GO_TO_SOURCE":
                    self.get_logger().info(f"✅ Arrived at source '{self.delivery_source}'. Loading {self.delivery_item}... Please wait 3 seconds.")
                    self.delivery_stage = "PICKING_UP"
                    self.pickup_timer = self.create_timer(3.0, self.pickup_complete_callback)

                # Delivery Stage 2: Destination
                elif self.delivery_stage == "GO_TO_DESTINATION":
                    self.get_logger().info(f"✅ Arrived at destination '{self.delivery_destination}'. Successfully delivered {self.delivery_item}!")
                    self.delivery_stage = None
                    self.active_nav_target = None

                # Security Patrol Waypoint Arrival
                elif self.patrol_mode:
                    total_wps = len(self.patrol_waypoints)
                    self.get_logger().info(
                        f"✅ [SECURITY PATROL] Arrived at Waypoint [{self.current_patrol_index + 1}/{total_wps}]. "
                        f"Surveying area for 2.5 seconds..."
                    )
                    self.is_dwelling = True
                    self.patrol_dwell_timer = self.create_timer(2.5, self.patrol_dwell_complete)

                # General Room / Point Navigation Arrival
                elif self.active_nav_target:
                    self.get_logger().info(f"✅ Successfully arrived at destination: '{self.active_nav_target}'!")
                    self.active_nav_target = None

            else:
                # Failure or cancellation
                if self.docking_active:
                    self.get_logger().error("Failed to navigate to Charging Station!")
                    self.docking_active = False
                elif self.delivery_stage is not None:
                    self.get_logger().warn(f"Delivery navigation was interrupted or failed during stage: {self.delivery_stage}")
                    self.delivery_stage = None
                elif self.patrol_mode:
                    self.get_logger().warn("Patrol navigation goal was cancelled or interrupted.")

                self.active_nav_target = None

        except Exception as e:
            self.get_logger().error(f"Navigation result callback failed: {e}")

    def pickup_complete_callback(self):
        if self.pickup_timer is not None:
            self.pickup_timer.cancel()
            self.pickup_timer = None

        if self.delivery_stage == "PICKING_UP":
            self.inventory[self.delivery_item] -= 1
            self.get_logger().info(f"Item acquired: Grabbed {self.delivery_item}! (Remaining stock: {self.inventory[self.delivery_item]})")

            self.delivery_stage = "GO_TO_DESTINATION"
            self.active_nav_target = self.delivery_destination
            x, y, yaw = self.locations[self.delivery_destination]
            self.get_logger().info(f"Navigating to final delivery destination: '{self.delivery_destination}'...")
            self.send_nav2_goal(x, y, yaw)

    def cancel_nav2_goal(self):
        if self.current_goal_handle is not None:
            self.get_logger().info("Canceling active Nav2 navigation goal.")
            try:
                self.current_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.current_goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = VoiceInterpreter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
