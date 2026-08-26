#!/usr/bin/env python3
"""
ROS 2 Node for Voice Command Detection, Recognition, and Interaction.
Features:
  1. Live Microphone Speech Recognition (using arecord + SpeechRecognition / Google STT).
  2. Continuous Voice Detection Mode (continuous hands-free command listening).
  3. Interactive Text Command Input (fallback for noisy/silent environments).
  4. Spoken Voice Feedback / Audio Acknowledgment via Text-to-Speech (spd-say).
  5. Publishes recognized voice commands to /voice/command.
"""

import os
import sys
import time
import subprocess
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Try importing speech_recognition
try:
    import speech_recognition as sr
    # Ensure recognize_google is available across all versions of speech_recognition
    if not hasattr(sr.Recognizer, 'recognize_google'):
        try:
            from speech_recognition.recognizers import google as google_recognizer
            sr.Recognizer.recognize_google = google_recognizer.recognize_legacy
        except Exception:
            pass
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False


class VoiceDetectorCLI(Node):
    def __init__(self):
        super().__init__('voice_simulator_cli')
        self.pub = self.create_publisher(String, '/voice/command', 10)
        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None
        self.sr_available = SR_AVAILABLE
        self.continuous_mode = False

        self.get_logger().info("Voice Command Detection Node Initialized.")

    def speak(self, text: str):
        """Play audible vocal response using system TTS."""
        try:
            subprocess.Popen(
                ["spd-say", "-t", "female3", "-r", "10", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    def get_vocal_response(self, cmd_text: str) -> str:
        """Map recognized voice commands to vocal confirmation messages."""
        cmd = cmd_text.lower()
        if "kitchen" in cmd:
            return "Navigating to the kitchen."
        elif "bedroom" in cmd:
            return "Navigating to the bedroom."
        elif "living room" in cmd:
            return "Navigating to the living room."
        elif "patrol" in cmd:
            return "Starting home security patrol mode."
        elif "follow" in cmd:
            return "Following you now."
        elif "water" in cmd:
            return "Fetching water bottle from kitchen counter."
        elif "medicine" in cmd:
            return "Fetching medicine from cabinet."
        elif "charge" in cmd or "dock" in cmd:
            return "Returning to charging station."
        elif "stop" in cmd or "halt" in cmd:
            return "Stopping all movement."
        elif "battery" in cmd:
            return "Checking battery level."
        elif "alarm" in cmd:
            return "Emergency alarm reset."
        else:
            return f"Executing command: {cmd_text}"

    def publish_command(self, cmd_text: str):
        """Publish recognized text command to ROS 2 topic /voice/command."""
        clean_text = cmd_text.strip()
        if not clean_text:
            return

        msg = String()
        msg.data = clean_text
        self.pub.publish(msg)

        # Vocal feedback
        response_text = self.get_vocal_response(clean_text)
        self.speak(response_text)

        print(f"\n🤖 [ROBOT ACTION]: '{clean_text}'")
        print(f"🔊 [ROBOT VOICE] : \"{response_text}\"\n")

    def capture_and_recognize(self, duration=4.0) -> str:
        """Capture audio from system microphone and transcribe with SpeechRecognition."""
        if not self.sr_available:
            print("Speech recognition module not available.")
            return ""

        tmp_wav = "/tmp/voice_command.wav"
        try:
            print(f"\n🎙️  Listening for {duration:.0f} seconds... (Speak your command now!)")
            # Record 16kHz 16-bit mono PCM from default microphone
            subprocess.run(
                ["arecord", "-q", "-d", str(int(duration)), "-f", "S16_LE", "-r", "16000", "-c", "1", tmp_wav],
                check=True
            )

            print("🧠 Processing voice recognition...")
            with sr.AudioFile(tmp_wav) as source:
                audio = self.recognizer.record(source)

            # Recognize speech using Google Speech Recognition
            if hasattr(self.recognizer, 'recognize_google'):
                recognized_text = self.recognizer.recognize_google(audio)
            else:
                from speech_recognition.recognizers import google as google_recognizer
                recognized_text = google_recognizer.recognize_legacy(self.recognizer, audio)
            print(f"✨ Recognized Speech: \"{recognized_text}\"")
            return recognized_text.strip()

        except sr.UnknownValueError:
            print("❌ Speech Recognition: Could not understand audio (try speaking closer to the mic).")
            return ""
        except sr.RequestError as e:
            print(f"⚠️ Speech Recognition Service Error: {e}")
            return ""
        except Exception as e:
            print(f"❌ Microphone Capture Error: {e}")
            return ""
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    def run_cli(self):
        time.sleep(0.5)
        print("\n" + "=" * 55)
        print("    🤖 AI HOUSEHOLD ROBOT - VOICE INTERACTION CLI     ")
        print("=" * 55)
        print("Available Voice Commands:")
        print("  📍 Navigation : 'go to kitchen' | 'navigate to bedroom' | 'go to living room'")
        print("  📦 Delivery   : 'bring water to bedroom' | 'bring medicine to living room'")
        print("  🛡️ Security   : 'start patrol' | 'stop patrol' | 'clear alarm'")
        print("  🚶 Follow Me  : 'follow me' | 'stop following' | 'stop'")
        print("  ⚡ Battery    : 'dock' | 'go charge' | 'battery status'")
        print("  🚪 Manage Map : 'save location <name>' | 'list locations'")
        print("=" * 55 + "\n")

        while rclpy.ok():
            prompt = (
                "Select Mode: [S]peak into Mic | [C]ontinuous Voice Mode | [T]ype text | [Q]uit: "
            )
            try:
                choice = input(prompt).strip().lower()
            except (KeyboardInterrupt, EOFError):
                break

            if choice in ['q', 'quit', 'exit']:
                break

            # Single voice capture mode
            elif choice == 's':
                text = self.capture_and_recognize(duration=4.0)
                if text:
                    self.publish_command(text)

            # Continuous hands-free listening loop
            elif choice == 'c':
                print("\n🎙️ [CONTINUOUS VOICE MODE ACTIVE]")
                print("Press Ctrl+C anytime to return to main menu.\n")
                try:
                    while rclpy.ok():
                        text = self.capture_and_recognize(duration=4.0)
                        if text:
                            self.publish_command(text)
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    print("\n[Stopped Continuous Voice Mode]\n")

            # Direct text input mode
            elif choice == 't':
                cmd_text = input("Enter command text: ").strip()
                if cmd_text:
                    self.publish_command(cmd_text)

            # Direct text command fallback
            elif choice:
                self.publish_command(choice)

        print("\nExiting Voice Interaction CLI...")
        sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceDetectorCLI()

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
