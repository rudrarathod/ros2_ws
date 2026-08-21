#!/usr/bin/env python3
"""
Interactive Command-line Interface to Simulate Voice Commands.
Publishes String messages to /voice/command.
Allows typing commands or speaking into the microphone (if SpeechRecognition is installed).
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import threading

class VoiceSimulatorCLI(Node):
    def __init__(self):
        super().__init__('voice_simulator_cli')
        self.pub = self.create_publisher(String, '/voice/command', 10)
        
        # Try importing SpeechRecognition for microphone input support
        self.sr_available = False
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.sr_available = True
            self.get_logger().info("Speech Recognition (Mic Input) is available.")
        except ImportError:
            self.get_logger().info("Speech Recognition not installed. Falling back to TEXT input only.")
        except Exception as e:
            self.get_logger().info(f"Microphone init failed ({e}). Falling back to TEXT input only.")

    def publish_command(self, cmd_text):
        msg = String()
        msg.data = cmd_text
        self.pub.publish(msg)
        print(f"-> Published voice command: '{cmd_text}'")

    def run_cli(self):
        print("\n==============================================")
        print("    AI Robot Voice Command Simulator CLI      ")
        print("==============================================")
        print("Available Commands:")
        print("  - 'follow me' / 'come here'  (Start following you)")
        print("  - 'stop' / 'halt' / 'stay'   (Brake and stop robot)")
        print("  - 'go to kitchen'            (Send Nav2 to kitchen)")
        print("  - 'go to bedroom'            (Send Nav2 to bedroom)")
        print("  - 'go to living room'        (Send Nav2 to living room)")
        print("  - 'patrol'                   (Start waypoints patrol loop)")
        print("  - 'quit' / 'exit'            (Close CLI)")
        print("==============================================\n")

        while rclpy.ok():
            mode_prompt = "[T]ype command" + (" or [S]peak command" if self.sr_available else "") + " (or type 'exit'): "
            user_choice = input(mode_prompt).strip().lower()

            if user_choice == 'exit' or user_choice == 'quit':
                break

            if user_choice == 's' and self.sr_available:
                import speech_recognition as sr
                print("Listening... (Speak your command now)")
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    try:
                        audio = self.recognizer.listen(source, timeout=3.0, phrase_time_limit=3.0)
                        print("Processing audio...")
                        command = self.recognizer.recognize_google(audio)
                        self.publish_command(command)
                    except sr.WaitTimeoutError:
                        print("Timeout: No speech detected.")
                    except sr.UnknownValueError:
                        print("Error: Could not understand audio.")
                    except sr.RequestError as e:
                        print(f"Service Error: Google Speech Recognition failed; {e}")
                    except Exception as e:
                        print(f"Error: {e}")
            else:
                # If they typed 't', or if they entered the command directly:
                if user_choice == 't':
                    cmd = input("Enter voice command text: ").strip()
                else:
                    cmd = user_choice

                if cmd == 'exit' or cmd == 'quit':
                    break
                if cmd:
                    self.publish_command(cmd)

        print("Exiting CLI...")
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceSimulatorCLI()
    
    # Run the CLI in a separate thread so ROS can spin
    cli_thread = threading.Thread(target=node.run_cli)
    cli_thread.daemon = True
    cli_thread.start()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
