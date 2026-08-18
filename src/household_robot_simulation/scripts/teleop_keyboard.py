#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty

msg = """
Control Your Custom Robot!
---------------------------
Moving around:
   W : Move Forward
   S : Move Backward
   A : Turn Left
   D : Turn Right

Space : Stop

Adjust speed:
   + or = : Increase speed (linear +1.0 m/s, angular +2.0 rad/s)
   -      : Decrease speed (linear -1.0 m/s, angular -2.0 rad/s)

Q : Quit

Keep holding WASD keys to drive. Releasing them will automatically stop the robot.
"""

class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.publisher = self.create_publisher(Twist, 'cmd_vel_raw', 10)
        self.settings = termios.tcgetattr(sys.stdin)
        self.linear_speed = 1.0   # Default linear velocity (m/s)
        self.angular_speed = 2.0  # Default angular velocity (rad/s)

    def getKey(self):
        # Set terminal to raw mode to read single character immediately
        tty.setraw(sys.stdin.fileno())
        # Use select to wait up to 0.15s for key input
        rlist, _, _ = select.select([sys.stdin], [], [], 0.15)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def print_speeds(self):
        print(f"Current speeds: Linear = {self.linear_speed:.1f} m/s | Angular = {self.angular_speed:.1f} rad/s          ", end='\r')

    def run(self):
        print(msg)
        self.print_speeds()
        print("\n")
        try:
            while rclpy.ok():
                key = self.getKey()
                twist = Twist()

                if key == 'w' or key == 'W':
                    twist.linear.x = self.linear_speed
                    twist.angular.z = 0.0
                    print("Cmd: Forward  ", end='\r')
                elif key == 's' or key == 'S':
                    twist.linear.x = -self.linear_speed
                    twist.angular.z = 0.0
                    print("Cmd: Backward ", end='\r')
                elif key == 'a' or key == 'A':
                    twist.linear.x = 0.0
                    twist.angular.z = self.angular_speed
                    print("Cmd: Turn Left ", end='\r')
                elif key == 'd' or key == 'D':
                    twist.linear.x = 0.0
                    twist.angular.z = -self.angular_speed
                    print("Cmd: Turn Right", end='\r')
                elif key == ' ':
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    print("Cmd: Stop      ", end='\r')
                elif key == '+' or key == '=':
                    self.linear_speed = min(150.0, self.linear_speed + 1.0)
                    self.angular_speed = min(300.0, self.angular_speed + 2.0)
                    print(f"\nSpeed updated: Linear = {self.linear_speed:.1f} m/s | Angular = {self.angular_speed:.1f} rad/s")
                    continue
                elif key == '-':
                    self.linear_speed = max(1.0, self.linear_speed - 1.0)
                    self.angular_speed = max(2.0, self.angular_speed - 2.0)
                    print(f"\nSpeed updated: Linear = {self.linear_speed:.1f} m/s | Angular = {self.angular_speed:.1f} rad/s")
                    continue
                elif key == 'q' or key == 'Q':
                    print("\nQuitting...")
                    break
                else:
                    # If no key was received in the last 0.15s, stop the robot for safety
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

                self.publisher.publish(twist)

        except Exception as e:
            print(f"\nError: {e}")
        finally:
            # Publish stop command before exiting
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboard()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
