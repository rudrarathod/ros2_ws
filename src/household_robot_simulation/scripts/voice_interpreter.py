#!/usr/bin/env python3
"""
ROS 2 Node for Voice Command Interpretation and Task Dispatching.
Subscribes to /voice/command.
Translates commands to robot actions:
- "follow me" / "come here" -> Enable follow_mode on person_follower.
- "stop" / "halt" -> Disable follow_mode on person_follower and brake.
- "save location <name>" / "delete location <name>" / "list locations" -> Manage semantic rooms.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from visualization_msgs.msg import Marker
import json
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
import math
import os
import re
import yaml
import tf2_ros
from ament_index_python.packages import get_package_share_directory

class VoiceInterpreter(Node):
    def __init__(self):
        super().__init__('voice_interpreter')

        # TF2 Setup to query current robot position for saving locations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Load semantic locations from YAML configuration database
        self.locations = {}
        try:
            pkg_share = get_package_share_directory('household_robot_simulation')
            yaml_path = os.path.join(pkg_share, 'config', 'semantic_locations.yaml')
            self.yaml_path = os.path.realpath(yaml_path)
            with open(self.yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Parse target locations
            for name, coords in config.get('locations', {}).items():
                self.locations[name] = (coords['x'], coords['y'], coords['yaw'])
                
            self.get_logger().info(f"Successfully loaded {len(self.locations)} semantic locations from database.")
        except Exception as e:
            self.get_logger().error(f"Failed to load semantic locations database: {e}")
            # Fallback to hardcoded defaults in case database loading fails
            self.locations = {
                'kitchen': (3.5, -1.0, 0.0),
                'bedroom': (0.0, 3.5, 1.57),
                'living room': (0.0, 0.0, 0.0),
                'start': (0.0, 0.0, 0.0)
            }

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

        # Ensure charging station exists in target locations database
        if 'charging_station' not in self.locations:
            self.locations['charging_station'] = (0.0, 0.0, 0.0)

        # Service client for setting follower parameters
        self.param_client = self.create_client(SetParameters, '/person_follower/set_parameters')
        
        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Emergency Alert States
        self.emergency_active = False
        self.emergency_reason = ""

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

        # Publisher for braking
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)

        self.current_goal_handle = None

        # 1Hz Battery simulation timer
        self.battery_timer = self.create_timer(1.0, self.simulate_battery)

        self.get_logger().info("Voice Command Interpreter Node Initialized.")

    def cmd_vel_callback(self, msg: Twist):
        # Determine if the robot is actively moving
        self.robot_is_moving = (abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01)

    def yolo_callback(self, msg: String):
        if self.is_charging or self.emergency_active:
            return  # Skip checks while charging or if alarm is already active

        try:
            detections = json.loads(msg.data)
        except Exception:
            return

        # Check for emergency conditions:
        # Hazard: 'knife' or 'scissors' seen
        for d in detections:
            cls = d['class'].lower()
            if cls in ["knife", "scissors"]:
                self.trigger_emergency_alarm(f"Hazardous Item Detected: {cls}!")
                break

    def trigger_emergency_alarm(self, reason: str):
        if self.emergency_active:
            return  # Already in emergency state
            
        self.emergency_active = True
        self.emergency_reason = reason
        
        self.get_logger().error(f"!!! EMERGENCY ALERT !!! {reason} - Sounding alarms and halting robot.")
        
        # Stop the robot immediately
        self.delivery_stage = None
        self.set_follower_mode(False)
        self.cancel_nav2_goal()
        
        # Publish active braking commands
        brake_msg = Twist()
        self.cmd_pub.publish(brake_msg)
        
        # Publish Alarm topic
        alarm_msg = Bool()
        alarm_msg.data = True
        self.alarm_pub.publish(alarm_msg)
        
        # Update markers
        self.publish_emergency_marker()

    def publish_emergency_marker(self):
        # Publish a red warning marker in RViz
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "emergency"
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position floating 0.85m above the robot base (higher than battery marker)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.85
        marker.pose.orientation.w = 1.0
        
        marker.text = f"!!! EMERGENCY: {self.emergency_reason.upper()} !!!"
        marker.scale.z = 0.15
        
        # Flashing red text
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)

    def simulate_battery(self):
        if self.is_charging:
            # Charge the battery by 5.0% per second for fast simulation testing
            self.battery_level = min(100.0, self.battery_level + 5.0)
            if self.battery_level >= 100.0:
                self.is_charging = False
                self.low_battery_triggered = False
                self.get_logger().info("Battery fully charged (100.0%)! Ready for new commands.")
        else:
            # Discharge: 0.1% per second idle, 0.4% per second when moving
            decay = 0.4 if self.robot_is_moving else 0.1
            self.battery_level = max(0.0, self.battery_level - decay)
            
            # Warn at 30%
            if self.battery_level <= 30.0 and int(self.battery_level) % 5 == 0 and abs(self.battery_level - int(self.battery_level)) < 0.1:
                self.get_logger().warn(f"Low Battery Alert: {self.battery_level:.1f}% Remaining.")
                
            # Trigger autonomous docking at 20%
            if self.battery_level <= 20.0 and not self.low_battery_triggered:
                self.low_battery_triggered = True
                self.trigger_autonomous_docking()

        # Publish current battery level
        battery_msg = Float32()
        battery_msg.data = self.battery_level
        self.battery_pub.publish(battery_msg)

        # Publish RViz visualization marker
        self.publish_battery_marker()

    def publish_battery_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "battery"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position floating 0.6m above the robot base
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.6
        marker.pose.orientation.w = 1.0
        
        # Formulate text representation
        state = "Charging" if self.is_charging else "Discharging"
        marker.text = f"Battery: {self.battery_level:.1f}% ({state})"
        
        # Size of the text font
        marker.scale.z = 0.12
        
        # Color based on battery level
        if self.battery_level > 50.0:
            marker.color.r = 0.0
            marker.color.g = 1.0  # Green
            marker.color.b = 0.0
        elif self.battery_level > 20.0:
            marker.color.r = 1.0
            marker.color.g = 1.0  # Yellow
            marker.color.b = 0.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0  # Red
            marker.color.b = 0.0
            
        marker.color.a = 1.0  # Opaque
        
        # Publisher
        self.marker_pub.publish(marker)

    def trigger_autonomous_docking(self):
        self.get_logger().warn(f"Battery Critical ({self.battery_level:.1f}%)! Aborting tasks and returning to charging station.")
        self.delivery_stage = None
        self.set_follower_mode(False)
        self.cancel_nav2_goal()
        
        # Dispatch docking goal
        self.docking_active = True
        x, y, yaw = self.locations['charging_station']
        self.send_nav2_goal(x, y, yaw)

    def set_follower_mode(self, enable: bool):
        """Set the follow_mode parameter on person_follower node."""
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
        """Get the robot's current pose using TF."""
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
        """Save in-memory locations back to YAML, preserving existing other keys like patrol_waypoints."""
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
        name = name.strip().lower()
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
        self.get_logger().info(f"Action: Saved new location '{name}' at x={x}, y={y}, yaw={yaw} rad ({deg}°)")

    def handle_delete_location(self, raw_name: str):
        name = raw_name.strip()
        for prefix in ['location ', 'room ', 'spot ']:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
                break
        name = name.strip().lower()
        if name in self.locations:
            del self.locations[name]
            self.persist_database()
            self.get_logger().info(f"Action: Successfully deleted location '{name}'.")
        else:
            self.get_logger().warn(f"Delete Location Failed: Location '{name}' not found in database.")

    def handle_list_locations(self):
        self.get_logger().info(f"--- Mapped Locations ({len(self.locations)}) ---")
        for name, coords in self.locations.items():
            deg = round(math.degrees(coords[2]), 1)
            self.get_logger().info(f"  • {name:20s}: x={coords[0]:6.2f}, y={coords[1]:6.2f}, yaw={coords[2]:5.2f} rad ({deg:5.1f}°)")

    def voice_callback(self, msg: String):
        command = msg.data.lower().strip()
        self.get_logger().info(f"Received voice command: '{command}'")

        # Location Management Commands
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

        elif any(kw in command for kw in ["list locations", "show locations", "list rooms", "show rooms"]):
            self.handle_list_locations()
            return

        # Check if emergency alarm is active
        is_clear_cmd = any(kw in command for kw in ["clear", "reset", "cancel", "all clear"])
        if self.emergency_active and not is_clear_cmd:
            self.get_logger().warn(f"Action Refused: Emergency Alarm is ACTIVE ({self.emergency_reason}). Please clear the alarm first!")
            return

        # Check if battery is critically low
        is_battery_cmd = any(kw in command for kw in ["battery", "charge", "dock", "recharge", "refill", "restock"])
        if self.battery_level < 10.0 and not is_battery_cmd and not is_clear_cmd:
            self.get_logger().error(f"Action Refused: Battery level too low ({self.battery_level:.1f}%). Please allow the robot to charge.")
            return

        # Interrupt charging if executing a new command
        if self.is_charging and not is_battery_cmd and not is_clear_cmd:
            self.is_charging = False
            self.low_battery_triggered = False
            self.get_logger().info(f"Undocking: Interrupting charging session at {self.battery_level:.1f}% to execute new command.")

        # 1. Follow Commands
        if any(kw in command for kw in ["follow me", "come here", "track me", "start following"]):
            self.cancel_nav2_goal()
            self.set_follower_mode(True)
            self.get_logger().info("Action: Enabling Person Following Mode.")

        # 2. Stop Commands
        elif any(kw in command for kw in ["stop", "halt", "stay", "brake"]):
            self.cancel_nav2_goal()
            self.set_follower_mode(False)
            # Send immediate active braking command
            brake_msg = Twist()
            self.cmd_pub.publish(brake_msg)
            self.get_logger().info("Action: Stopping robot movement.")

        # 3. Battery Status Commands
        elif any(kw in command for kw in ["battery status", "battery level", "check battery"]):
            state_str = "Charging" if self.is_charging else "Discharging"
            self.get_logger().info(f"Battery Status: {self.battery_level:.1f}% ({state_str})")

        # 4. Manual Charging/Docking Commands
        elif any(kw in command for kw in ["go charge", "dock", "return to charger", "recharge"]):
            self.get_logger().info("Action: Manual command received. Returning to charging station.")
            self.delivery_stage = None
            self.set_follower_mode(False)
            self.cancel_nav2_goal()
            
            self.docking_active = True
            x, y, yaw = self.locations['charging_station']
            self.send_nav2_goal(x, y, yaw)

        # 5. Alarm Reset/Clear Commands
        elif is_clear_cmd:
            if self.emergency_active:
                self.emergency_active = False
                self.emergency_reason = ""
                
                # Publish Alarm topic false
                alarm_msg = Bool()
                alarm_msg.data = False
                self.alarm_pub.publish(alarm_msg)
                
                # Delete the emergency marker
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "emergency"
                marker.id = 1
                marker.action = Marker.DELETE
                self.marker_pub.publish(marker)
                
                self.get_logger().info("Action: Emergency alarm successfully cleared. Resuming normal operations.")
            else:
                self.get_logger().info("No active alarms to clear.")

        # 5. Delivery Commands
        elif any(kw in command for kw in ["deliver", "bring", "get", "fetch"]):
            self.set_follower_mode(False)
            self.cancel_nav2_goal()
            
            # Determine item type and pickup location
            item = "water bottle" if "water" in command else "medicine"
            source = "kitchen_counter" if item == "water bottle" else "medicine_cabinet"
            
            # Check counter inventory database
            if self.inventory[item] <= 0:
                self.get_logger().error(f"Action Aborted: '{item}' is OUT OF STOCK on the counter/storage! Please restock it first.")
                return
            
            # Determine destination location
            destination = "bedroom" if item == "medicine" else "living room"
            for room in ["kitchen", "bedroom", "living room"]:
                if room in command:
                    destination = room
                    break
                    
            self.delivery_item = item
            self.delivery_source = source
            self.delivery_destination = destination
            self.delivery_stage = "GO_TO_SOURCE"
            
            x, y, yaw = self.locations[source]
            self.get_logger().info(f"Action: Starting delivery workflow. Retrieve {item} from {source} (Stock: {self.inventory[item]}) and deliver to {destination}.")
            self.send_nav2_goal(x, y, yaw)

        # 6. Restocking Commands
        elif any(kw in command for kw in ["restock", "refill"]):
            self.set_follower_mode(False)
            self.cancel_nav2_goal()
            
            if "water" in command:
                self.inventory['water bottle'] = 3
                self.get_logger().info("Action: Kitchen counter successfully restocked to 3 water bottles.")
            elif "medicine" in command:
                self.inventory['medicine'] = 2
                self.get_logger().info("Action: Medicine cabinet successfully restocked to 2 units.")
            else:
                self.inventory['water bottle'] = 3
                self.inventory['medicine'] = 2
                self.get_logger().info("Action: All item storage counters successfully restocked.")

        else:
            self.get_logger().warn(f"Unrecognized voice command phrase: '{command}'")

    def send_nav2_goal(self, x, y, yaw):
        # We do NOT call self.cancel_nav2_goal() here.
        # ROS 2 Nav2 Action Server naturally supports preemption. Sending a new goal
        # will automatically preempt (cancel) the active navigation task on the server,
        # avoiding race conditions on goal status updates.

        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Nav2 action server not available.")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        
        # Calculate quaternion from yaw
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Sending NavigateToPose goal to x={x:.2f}, y={y:.2f}")
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().info("Goal rejected by Nav2 server")
                return
            self.current_goal_handle = goal_handle
            
            # Watch the result
            def result_cb(result_future):
                self.get_result_callback(result_future, goal_handle)
            
            self.get_result_future = goal_handle.get_result_async()
            self.get_result_future.add_done_callback(result_cb)
        except Exception as e:
            self.get_logger().error(f"Goal response callback failed: {e}")

    def get_result_callback(self, future, goal_handle):
        try:
            if goal_handle != self.current_goal_handle:
                return

            result = future.result()
            status = result.status
            self.get_logger().info(f"Nav2 navigation finished with status: {status}")
            self.current_goal_handle = None

            if status == GoalStatus.STATUS_SUCCEEDED:
                if self.docking_active:
                    self.docking_active = False
                    self.is_charging = True
                    self.get_logger().info("Arrived at Charging Station. Successfully docked! Charging started...")
                elif self.delivery_stage == "GO_TO_SOURCE":
                    self.get_logger().info(f"Arrived at source '{self.delivery_source}'. Loading {self.delivery_item}... Please wait 3 seconds.")
                    self.delivery_stage = "PICKING_UP"
                    self.pickup_timer = self.create_timer(3.0, self.pickup_complete_callback)
                elif self.delivery_stage == "GO_TO_DESTINATION":
                    self.get_logger().info(f"Arrived at destination '{self.delivery_destination}'. Successfully delivered {self.delivery_item}!")
                    self.delivery_stage = None
            else:
                if self.docking_active:
                    self.get_logger().error("Failed to navigate to Charging Station!")
                    self.docking_active = False
                elif self.delivery_stage is not None:
                    self.get_logger().warn(f"Delivery navigation failed or was canceled during stage: {self.delivery_stage}")
                    self.delivery_stage = None

        except Exception as e:
            self.get_logger().error(f"Navigation result callback failed: {e}")

    def pickup_complete_callback(self):
        if self.pickup_timer is not None:
            self.pickup_timer.cancel()
            self.pickup_timer = None
            
        if self.delivery_stage == "PICKING_UP":
            # Decrement inventory upon successful pickup
            self.inventory[self.delivery_item] -= 1
            self.get_logger().info(f"Successfully grabbed {self.delivery_item}! (Remaining stock on counter: {self.inventory[self.delivery_item]})")
            
            self.delivery_stage = "GO_TO_DESTINATION"
            x, y, yaw = self.locations[self.delivery_destination]
            self.get_logger().info(f"Item loaded. Navigating to destination: '{self.delivery_destination}'...")
            self.send_nav2_goal(x, y, yaw)

    def cancel_nav2_goal(self):
        if self.current_goal_handle is not None:
            self.get_logger().info("Canceling active Nav2 navigation goal.")
            self.current_goal_handle.cancel_goal_async()
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
