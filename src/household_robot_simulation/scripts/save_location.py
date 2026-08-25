#!/usr/bin/env python3
"""
Interactive CLI and command-line utility for managing semantic rooms, locations, and saved patrol waypoints.
Features:
- Save current robot pose (from TF) as a named room or patrol waypoint.
- List all saved locations and patrol waypoints with coordinates and orientations.
- Edit/update coordinates of existing rooms.
- Delete rooms or patrol waypoints from the database.
- Immediately updates config/semantic_locations.yaml.
"""

import os
import sys
import math
import argparse
import yaml
import rclpy
from rclpy.node import Node
import tf2_ros
from ament_index_python.packages import get_package_share_directory


class WaypointManager(Node):
    def __init__(self):
        super().__init__('waypoint_manager')
        
        # TF2 Setup to query current robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Determine YAML database path (resolve symlink to modify actual source file)
        try:
            pkg_share = get_package_share_directory('household_robot_simulation')
            install_yaml = os.path.join(pkg_share, 'config', 'semantic_locations.yaml')
            self.yaml_path = os.path.realpath(install_yaml)
        except Exception:
            ws_root = os.path.expanduser('~/ros2_ws')
            self.yaml_path = os.path.join(ws_root, 'src', 'household_robot_simulation', 'config', 'semantic_locations.yaml')

        self.get_logger().info(f"Using semantic database: {self.yaml_path}")

    def load_database(self):
        """Load YAML configuration from disk."""
        if not os.path.exists(self.yaml_path):
            return {'locations': {}, 'patrol_waypoints': []}
        try:
            with open(self.yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if not data:
                    data = {}
                if 'locations' not in data:
                    data['locations'] = {}
                if 'patrol_waypoints' not in data:
                    data['patrol_waypoints'] = []
                return data
        except Exception as e:
            self.get_logger().error(f"Error loading YAML: {e}")
            return {'locations': {}, 'patrol_waypoints': []}

    def save_database(self, data):
        """Save YAML configuration to disk."""
        try:
            os.makedirs(os.path.dirname(self.yaml_path), exist_ok=True)
            with open(self.yaml_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            print(f"-> Successfully saved database to: {self.yaml_path}")
            return True
        except Exception as e:
            print(f"-> Error saving database: {e}")
            return False

    def get_current_pose(self, target_frame='map', base_frame='base_footprint', timeout=2.0):
        """Look up the robot's current pose using TF."""
        start_time = self.get_clock().now().nanoseconds / 1e9
        while (self.get_clock().now().nanoseconds / 1e9 - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    base_frame,
                    rclpy.time.Time()
                )
                trans = transform.transform.translation
                rot = transform.transform.rotation

                # Convert quaternion to yaw
                siny_cosp = 2.0 * (rot.w * rot.z + rot.x * rot.y)
                cosy_cosp = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)

                return round(trans.x, 3), round(trans.y, 3), round(yaw, 3)
            except Exception:
                pass
        
        # Fallback to odom frame if map frame is unavailable
        if target_frame == 'map':
            self.get_logger().warn("Map frame not found, attempting fallback to 'odom' frame...")
            return self.get_current_pose(target_frame='odom', base_frame=base_frame, timeout=1.0)
            
        return None

    def save_room(self, name, x=None, y=None, yaw=None):
        """Save a new room or overwrite existing coordinates."""
        name = name.strip().lower()
        if x is None or y is None or yaw is None:
            pose = self.get_current_pose()
            if pose is None:
                print("-> Error: Could not get robot pose from TF. Make sure simulation is running.")
                return False
            x, y, yaw = pose

        data = self.load_database()
        data['locations'][name] = {
            'x': float(x),
            'y': float(y),
            'yaw': float(yaw)
        }
        success = self.save_database(data)
        if success:
            deg = round(math.degrees(yaw), 1)
            print(f"-> Room '{name}' saved at: x={x}, y={y}, yaw={yaw} rad ({deg}°)")
        return success

    def add_patrol_waypoint(self, x=None, y=None, yaw=None):
        """Add a waypoint to the saved patrol waypoints list."""
        if x is None or y is None or yaw is None:
            pose = self.get_current_pose()
            if pose is None:
                print("-> Error: Could not get robot pose from TF. Make sure simulation is running.")
                return False
            x, y, yaw = pose

        data = self.load_database()
        new_wp = {
            'x': float(x),
            'y': float(y),
            'yaw': float(yaw)
        }
        data['patrol_waypoints'].append(new_wp)
        success = self.save_database(data)
        if success:
            deg = round(math.degrees(yaw), 1)
            idx = len(data['patrol_waypoints'])
            print(f"-> Patrol Waypoint #{idx} saved: x={x}, y={y}, yaw={yaw} rad ({deg}°)")
        return success

    def delete_room(self, name):
        """Delete a room by name."""
        name = name.strip().lower()
        data = self.load_database()
        if name in data['locations']:
            del data['locations'][name]
            self.save_database(data)
            print(f"-> Room '{name}' deleted successfully.")
            return True
        else:
            print(f"-> Error: Room '{name}' not found in database.")
            return False

    def delete_patrol_waypoint(self, index):
        """Delete a patrol waypoint by 1-based index."""
        data = self.load_database()
        waypoints = data.get('patrol_waypoints', [])
        if 1 <= index <= len(waypoints):
            removed = waypoints.pop(index - 1)
            self.save_database(data)
            print(f"-> Patrol Waypoint #{index} ({removed['x']}, {removed['y']}) removed.")
            return True
        else:
            print(f"-> Error: Invalid waypoint index #{index}. Valid range: 1 to {len(waypoints)}.")
            return False

    def list_all(self):
        """Print all saved rooms and patrol waypoints."""
        data = self.load_database()
        locations = data.get('locations', {})
        patrol = data.get('patrol_waypoints', [])

        print("\n===========================================================")
        print("          SAVED ROOMS & SEMANTIC LOCATIONS                 ")
        print("===========================================================")
        if not locations:
            print("  (No rooms saved yet)")
        else:
            for name, coords in locations.items():
                deg = round(math.degrees(coords['yaw']), 1)
                print(f"  • {name:20s} : x={coords['x']:6.2f}, y={coords['y']:6.2f}, yaw={coords['yaw']:5.2f} rad ({deg:5.1f}°)")

        print("\n===========================================================")
        print("               SAVED PATROL WAYPOINTS                      ")
        print("===========================================================")
        if not patrol:
            print("  (No patrol waypoints saved yet)")
        else:
            for i, wp in enumerate(patrol, start=1):
                deg = round(math.degrees(wp['yaw']), 1)
                print(f"  [{i}] Waypoint #{i:02d}        : x={wp['x']:6.2f}, y={wp['y']:6.2f}, yaw={wp['yaw']:5.2f} rad ({deg:5.1f}°)")
        print("===========================================================\n")


def interactive_menu(manager):
    """Run interactive text UI menu."""
    while True:
        print("\n===========================================================")
        print("        ROBOT LOCATION & WAYPOINT MANAGER                  ")
        print("===========================================================")
        print("  1. Save Current Pose as Room / Location")
        print("  2. Save Current Pose as Patrol Waypoint")
        print("  3. List All Rooms & Patrol Waypoints")
        print("  4. Edit Room Coordinates Manually")
        print("  5. Delete a Room / Location")
        print("  6. Delete a Patrol Waypoint")
        print("  7. Exit")
        print("===========================================================")
        
        choice = input("Select an option (1-7): ").strip()
        
        if choice == '1':
            name = input("Enter room / location name (e.g. 'dining_table', 'balcony'): ").strip()
            if name:
                manager.save_room(name)
        elif choice == '2':
            manager.add_patrol_waypoint()
        elif choice == '3':
            manager.list_all()
        elif choice == '4':
            name = input("Enter name of room to edit: ").strip().lower()
            data = manager.load_database()
            if name in data['locations']:
                curr = data['locations'][name]
                print(f"Current values: x={curr['x']}, y={curr['y']}, yaw={curr['yaw']}")
                use_current = input("Overwrite with current robot pose? [Y/n]: ").strip().lower()
                if use_current != 'n':
                    manager.save_room(name)
                else:
                    try:
                        x = float(input(f"Enter x [{curr['x']}]: ") or curr['x'])
                        y = float(input(f"Enter y [{curr['y']}]: ") or curr['y'])
                        yaw = float(input(f"Enter yaw [{curr['yaw']}]: ") or curr['yaw'])
                        manager.save_room(name, x, y, yaw)
                    except ValueError:
                        print("-> Invalid input numbers.")
            else:
                print(f"-> Room '{name}' not found.")
        elif choice == '5':
            name = input("Enter room name to delete: ").strip().lower()
            if name:
                confirm = input(f"Are you sure you want to delete '{name}'? [y/N]: ").strip().lower()
                if confirm == 'y':
                    manager.delete_room(name)
        elif choice == '6':
            manager.list_all()
            try:
                idx = int(input("Enter waypoint number to delete: ").strip())
                manager.delete_patrol_waypoint(idx)
            except ValueError:
                print("-> Invalid number.")
        elif choice == '7' or choice.lower() in ['q', 'exit', 'quit']:
            print("Exiting Manager.")
            break
        else:
            print("-> Invalid choice. Please enter a number 1 to 7.")


def main(args=None):
    rclpy.init(args=args)
    manager = WaypointManager()

    parser = argparse.ArgumentParser(description="Household Robot Location & Waypoint Manager")
    parser.add_argument('--room', type=str, help="Save current pose as specified room name")
    parser.add_argument('--patrol', '--waypoint', action='store_true', dest='patrol', help="Save current pose as patrol waypoint")
    parser.add_argument('--list', action='store_true', help="List all saved rooms and waypoints")
    parser.add_argument('--delete-room', type=str, help="Delete a room by name")
    parser.add_argument('--delete-patrol', '--delete-waypoint', type=int, dest='delete_patrol', help="Delete a patrol waypoint by 1-based index")
    parser.add_argument('--x', type=float, help="Manual X coordinate (optional)")
    parser.add_argument('--y', type=float, help="Manual Y coordinate (optional)")
    parser.add_argument('--yaw', type=float, help="Manual Yaw in rad (optional)")

    # Filter out ROS args
    filtered_args = [a for a in sys.argv[1:] if not a.startswith('--ros-args')]
    parsed = parser.parse_args(filtered_args)

    if parsed.list:
        manager.list_all()
    elif parsed.room:
        manager.save_room(parsed.room, parsed.x, parsed.y, parsed.yaw)
    elif parsed.patrol:
        manager.add_patrol_waypoint(parsed.x, parsed.y, parsed.yaw)
    elif parsed.delete_room:
        manager.delete_room(parsed.delete_room)
    elif parsed.delete_patrol is not None:
        manager.delete_patrol_waypoint(parsed.delete_patrol)
    else:
        interactive_menu(manager)

    manager.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
