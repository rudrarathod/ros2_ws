#!/usr/bin/env python3
"""
AI Household Service Robot - Web Dashboard & Control Center (Module 11)
Features:
  - Default Live Raw Camera Video Stream (with toggle for AI Processed / YOLO stream)
  - Real-Time Telemetry: Battery radial gauge, Robot State, Alarm status, Odometry (X, Y, Yaw, Velocity)
  - Interactive Control Center: Room Navigation (Kitchen, Bedroom, Living Room, Dock), Item Delivery (Water, Medicine)
  - Behavior Controls: Security Patrol, Person Following, Return to Charger, Emergency Stop
  - Virtual Teleop Controller: On-screen D-Pad + Keyboard WASD Driving (/cmd_vel_raw)
  - In-Browser Voice Command Interface: Web Speech API / Mic input + Text Command Bar (/voice/command)
  - Real-time 2D House Map Floorplan with live robot pose marker
  - Real-time activity and detection event logs
"""

import os
import sys
import time
import math
import json
import threading
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan

import subprocess
from flask import Flask, Response, jsonify, request, render_template_string

# Try importing speech_recognition
try:
    import speech_recognition as sr
    if not hasattr(sr.Recognizer, 'recognize_google'):
        try:
            from speech_recognition.recognizers import google as google_recognizer
            sr.Recognizer.recognize_google = google_recognizer.recognize_legacy
        except Exception:
            pass
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# Initialize Flask App
app = Flask(__name__)

# Shared Robot State Store
robot_state = {
    "battery": 100.0,
    "battery_state": "DISCHARGING",
    "status_text": "System Ready",
    "current_mode": "IDLE",
    "emergency_alarm": False,
    "x": 0.0,
    "y": 0.0,
    "yaw": 0.0,
    "linear_speed": 0.0,
    "angular_speed": 0.0,
    "min_laser_dist": 5.0,
    "person_detected": False,
    "last_command": "None",
    "logs": []
}

raw_frame_lock = threading.Lock()
latest_raw_frame = None

processed_frame_lock = threading.Lock()
latest_processed_frame = None


def add_log(message: str, category: str = "INFO"):
    timestamp = time.strftime("%H:%M:%S")
    entry = {"time": timestamp, "msg": message, "category": category}
    robot_state["logs"].append(entry)
    if len(robot_state["logs"]) > 50:
        robot_state["logs"].pop(0)


class WebDashboardNode(Node):
    def __init__(self):
        super().__init__('web_dashboard_node')

        # Publishers
        self.cmd_voice_pub = self.create_publisher(String, '/voice/command', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)

        # Subscriptions
        self.create_subscription(Image, '/camera/image_raw', self.raw_image_callback, 10)
        self.create_subscription(Image, '/camera/image_processed', self.processed_image_callback, 10)
        self.create_subscription(Float32, '/battery/percentage', self.battery_callback, 10)
        self.create_subscription(Bool, '/emergency/alarm', self.alarm_callback, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(String, '/voice/command', self.voice_command_callback, 10)
        self.create_subscription(String, '/person_follower/status', self.follower_status_callback, 10)

        add_log("ROS 2 Web Dashboard Node Connected", "SYSTEM")
        self.get_logger().info("Web Dashboard Node initialized successfully.")

    def raw_image_callback(self, msg: Image):
        global latest_raw_frame
        try:
            # Decode raw camera ROS Image to OpenCV BGR
            if msg.encoding in ['rgb8', 'bgr8']:
                channels = 3
                dtype = np.uint8
            elif msg.encoding == 'mono8':
                channels = 1
                dtype = np.uint8
            else:
                channels = 3
                dtype = np.uint8

            img = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width, channels))
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            with raw_frame_lock:
                latest_raw_frame = img
        except Exception as e:
            pass

    def processed_image_callback(self, msg: Image):
        global latest_processed_frame
        try:
            if msg.encoding in ['rgb8', 'bgr8']:
                channels = 3
                dtype = np.uint8
            else:
                channels = 3
                dtype = np.uint8

            img = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, msg.width, channels))
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            with processed_frame_lock:
                latest_processed_frame = img
        except Exception as e:
            pass

    def battery_callback(self, msg: Float32):
        robot_state["battery"] = round(float(msg.data), 1)
        if robot_state["battery"] >= 99.0 and robot_state["current_mode"] == "DOCKED":
            robot_state["battery_state"] = "CHARGED"
        elif robot_state["current_mode"] == "DOCKED":
            robot_state["battery_state"] = "CHARGING"
        else:
            robot_state["battery_state"] = "DISCHARGING"

    def alarm_callback(self, msg: Bool):
        if msg.data != robot_state["emergency_alarm"]:
            robot_state["emergency_alarm"] = msg.data
            if msg.data:
                add_log("🚨 EMERGENCY INTRUDER ALARM TRIGGERED!", "ALARM")
            else:
                add_log("✅ Emergency alarm reset.", "SYSTEM")

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        robot_state["x"] = round(pos.x, 2)
        robot_state["y"] = round(pos.y, 2)

        # Yaw from Quaternion
        siny_cosp = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        robot_state["yaw"] = round(math.degrees(yaw), 1)

        twist = msg.twist.twist
        robot_state["linear_speed"] = round(math.hypot(twist.linear.x, twist.linear.y), 2)
        robot_state["angular_speed"] = round(twist.angular.z, 2)

    def scan_callback(self, msg: LaserScan):
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if valid:
            robot_state["min_laser_dist"] = round(min(valid), 2)

    def voice_command_callback(self, msg: String):
        robot_state["last_command"] = msg.data
        add_log(f"Voice Command Dispatched: '{msg.data}'", "COMMAND")

        cmd = msg.data.lower()
        if "kitchen" in cmd:
            robot_state["current_mode"] = "NAVIGATING TO KITCHEN"
        elif "bedroom" in cmd:
            robot_state["current_mode"] = "NAVIGATING TO BEDROOM"
        elif "living" in cmd:
            robot_state["current_mode"] = "NAVIGATING TO LIVING ROOM"
        elif "patrol" in cmd and "stop" not in cmd:
            robot_state["current_mode"] = "SECURITY PATROL"
        elif "follow" in cmd and "stop" not in cmd:
            robot_state["current_mode"] = "FOLLOWING PERSON"
        elif "water" in cmd:
            robot_state["current_mode"] = "DELIVERING WATER"
        elif "medicine" in cmd:
            robot_state["current_mode"] = "DELIVERING MEDICINE"
        elif "dock" in cmd or "charge" in cmd:
            robot_state["current_mode"] = "DOCKING"
        elif "stop" in cmd or "halt" in cmd:
            robot_state["current_mode"] = "STOPPED"
        elif "clear" in cmd:
            robot_state["current_mode"] = "IDLE"

    def follower_status_callback(self, msg: String):
        if "FOLLOWING" in msg.data:
            robot_state["person_detected"] = True
        elif "LOST" in msg.data or "STOPPED" in msg.data:
            robot_state["person_detected"] = False

    def publish_voice_command(self, cmd_text: str):
        msg = String()
        msg.data = cmd_text
        self.cmd_voice_pub.publish(msg)

    def publish_teleop(self, linear: float, angular: float):
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_vel_pub.publish(twist)


# Global reference to ROS node
dashboard_node = None


# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def get_status():
    return jsonify(robot_state)


@app.route('/api/command', methods=['POST'])
def send_command():
    data = request.get_json(force=True)
    cmd = data.get('command', '').strip()
    if cmd and dashboard_node:
        dashboard_node.publish_voice_command(cmd)
        return jsonify({"status": "success", "command": cmd})
    return jsonify({"status": "error", "message": "No command provided"}), 400


@app.route('/api/teleop', methods=['POST'])
def teleop():
    data = request.get_json(force=True)
    linear = data.get('linear', 0.0)
    angular = data.get('angular', 0.0)
    if dashboard_node:
        dashboard_node.publish_teleop(linear, angular)
        return jsonify({"status": "success", "linear": linear, "angular": angular})
    return jsonify({"status": "error"}), 400


@app.route('/api/voice_record', methods=['POST'])
def voice_record():
    """Fallback voice recorder for Firefox / Zen Browser / Non-Chromium browsers."""
    if not SR_AVAILABLE:
        return jsonify({"status": "error", "message": "Speech Recognition module not available on server"}), 500

    tmp_wav = "/tmp/web_voice_command.wav"
    try:
        duration = 4
        # Record 16kHz 16-bit mono PCM from default system microphone
        subprocess.run(
            ["arecord", "-q", "-d", str(int(duration)), "-f", "S16_LE", "-r", "16000", "-c", "1", tmp_wav],
            check=True
        )

        r = sr.Recognizer()
        with sr.AudioFile(tmp_wav) as source:
            audio = r.record(source)

        if hasattr(r, 'recognize_google'):
            text = r.recognize_google(audio)
        else:
            from speech_recognition.recognizers import google as google_recognizer
            text = google_recognizer.recognize_legacy(r, audio)

        clean_text = text.strip()
        if clean_text and dashboard_node:
            dashboard_node.publish_voice_command(clean_text)

        return jsonify({"status": "success", "transcript": clean_text})
    except sr.UnknownValueError:
        return jsonify({"status": "error", "message": "Could not understand audio. Please speak closer to the mic."})
    except sr.RequestError as e:
        return jsonify({"status": "error", "message": f"Speech service error: {e}"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Recording error: {e}"})
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass


def generate_frames(stream_type="raw"):
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "Waiting for Camera Stream...", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    _, blank_encoded = cv2.imencode('.jpg', blank_frame)
    blank_bytes = blank_encoded.tobytes()

    while True:
        frame_bytes = None
        if stream_type == "processed":
            with processed_frame_lock:
                if latest_processed_frame is not None:
                    _, encoded = cv2.imencode('.jpg', latest_processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = encoded.tobytes()
        else:
            with raw_frame_lock:
                if latest_raw_frame is not None:
                    _, encoded = cv2.imencode('.jpg', latest_raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = encoded.tobytes()

        if frame_bytes is None:
            frame_bytes = blank_bytes

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # ~25 FPS


@app.route('/video_feed')
def video_feed():
    stream_type = request.args.get('type', 'raw')
    return Response(generate_frames(stream_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- HTML / CSS / JS EMBEDDED DASHBOARD ---

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Household Robot Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #090d16;
      --bg-card: rgba(17, 24, 39, 0.7);
      --bg-card-hover: rgba(31, 41, 55, 0.8);
      --border-card: rgba(255, 255, 255, 0.08);
      --border-accent: rgba(6, 182, 212, 0.4);
      --cyan: #06b6d4;
      --cyan-glow: rgba(6, 182, 212, 0.25);
      --emerald: #10b981;
      --emerald-glow: rgba(16, 185, 129, 0.25);
      --amber: #f59e0b;
      --rose: #f43f5e;
      --rose-glow: rgba(244, 63, 94, 0.4);
      --purple: #a855f7;
      --text-primary: #f9fafb;
      --text-secondary: #9ca3af;
      --text-muted: #6b7280;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-main);
      background-image: radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.05), transparent 40%),
                        radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.05), transparent 40%);
      color: var(--text-primary);
      font-family: 'Inter', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* HEADER */
    header {
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-card);
      padding: 1rem 1.75rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 50;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .brand-icon {
      width: 40px;
      height: 40px;
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.4rem;
      box-shadow: 0 0 16px var(--cyan-glow);
    }

    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .brand-subtitle {
      font-size: 0.75rem;
      color: var(--cyan);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 600;
    }

    .header-badges {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.4rem 0.85rem;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--emerald);
    }

    .status-dot {
      width: 8px;
      height: 8px;
      background: var(--emerald);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--emerald);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* EMERGENCY BANNER */
    #alarmBanner {
      display: none;
      background: linear-gradient(90deg, #991b1b, #dc2626, #991b1b);
      color: white;
      padding: 0.75rem 1.5rem;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      font-size: 0.95rem;
      letter-spacing: 0.02em;
      animation: alertPulse 1.5s infinite;
      border-bottom: 2px solid #f87171;
    }

    @keyframes alertPulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.85; }
    }

    .alarm-btn {
      background: white;
      color: #991b1b;
      border: none;
      padding: 0.35rem 1rem;
      border-radius: 6px;
      font-weight: 700;
      cursor: pointer;
      font-size: 0.8rem;
      transition: all 0.2s;
    }

    .alarm-btn:hover {
      background: #fee2e2;
      transform: scale(1.05);
    }

    /* MAIN CONTAINER GRID */
    main {
      flex: 1;
      padding: 1.5rem 1.75rem;
      display: grid;
      grid-template-columns: 1.45fr 1fr 1fr;
      gap: 1.25rem;
      max-width: 1750px;
      margin: 0 auto;
      width: 100%;
    }

    @media (max-width: 1280px) {
      main {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media (max-width: 860px) {
      main {
        grid-template-columns: 1fr;
      }
    }

    /* GLASS CARD */
    .card {
      background: var(--bg-card);
      backdrop-filter: blur(14px);
      border: 1px solid var(--border-card);
      border-radius: 18px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      transition: border-color 0.3s, box-shadow 0.3s;
    }

    .card:hover {
      border-color: rgba(255, 255, 255, 0.14);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .card-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* VIDEO STREAM PANEL */
    .video-container {
      position: relative;
      width: 100%;
      border-radius: 12px;
      overflow: hidden;
      background: #000;
      aspect-ratio: 4 / 3;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .video-container img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .video-overlay-badge {
      position: absolute;
      top: 12px;
      left: 12px;
      background: rgba(0, 0, 0, 0.65);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .stream-toggle-group {
      display: flex;
      background: rgba(0, 0, 0, 0.4);
      padding: 3px;
      border-radius: 10px;
      border: 1px solid var(--border-card);
    }

    .stream-toggle-btn {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      padding: 0.35rem 0.75rem;
      border-radius: 8px;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .stream-toggle-btn.active {
      background: var(--cyan);
      color: #000;
      box-shadow: 0 0 10px var(--cyan-glow);
    }

    /* METRICS / GAUGES ROW */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.85rem;
    }

    .metric-card {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border-card);
      border-radius: 14px;
      padding: 0.85rem 1rem;
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }

    .metric-icon {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
    }

    .metric-val {
      font-family: 'Outfit', sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-primary);
    }

    .metric-lbl {
      font-size: 0.7rem;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    /* BATTERY BAR */
    .battery-wrapper {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .battery-bar-outer {
      width: 100%;
      height: 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }

    .battery-bar-inner {
      height: 100%;
      width: 100%;
      background: linear-gradient(90deg, var(--emerald), #34d399);
      border-radius: 999px;
      transition: width 0.4s, background 0.4s;
    }

    /* ACTION BUTTONS */
    .btn-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.65rem;
    }

    .action-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-card);
      color: var(--text-primary);
      padding: 0.75rem 0.85rem;
      border-radius: 12px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.6rem;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      user-select: none;
    }

    .action-btn:hover {
      background: rgba(6, 182, 212, 0.15);
      border-color: var(--cyan);
      transform: translateY(-2px);
      box-shadow: 0 4px 14px var(--cyan-glow);
    }

    .action-btn:active {
      transform: translateY(0);
    }

    .action-btn.danger {
      background: rgba(244, 63, 94, 0.1);
      border-color: rgba(244, 63, 94, 0.3);
      color: #fca5a5;
    }

    .action-btn.danger:hover {
      background: var(--rose);
      color: white;
      box-shadow: 0 0 16px var(--rose-glow);
    }

    .action-btn.primary {
      background: rgba(6, 182, 212, 0.15);
      border-color: var(--cyan);
      color: var(--cyan);
    }

    /* MINIMAL TELEOP CONTROLLER */
    .teleop-minimal {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.65rem;
      padding: 0.3rem 0;
    }

    .dpad-grid {
      display: grid;
      grid-template-columns: repeat(3, 46px);
      grid-template-rows: repeat(3, 46px);
      gap: 6px;
      justify-content: center;
      align-items: center;
    }

    .dpad-cell {
      width: 46px;
      height: 46px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border-card);
      color: var(--text-primary);
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      user-select: none;
      transition: all 0.15s ease;
    }

    .dpad-cell .dpad-sym {
      font-size: 1rem;
      line-height: 1;
      color: var(--cyan);
    }

    .dpad-cell .dpad-sub {
      font-size: 0.55rem;
      font-weight: 700;
      color: var(--text-muted);
      margin-top: 2px;
      font-family: 'JetBrains Mono', monospace;
    }

    .dpad-cell:hover {
      background: rgba(6, 182, 212, 0.15);
      border-color: var(--cyan);
      transform: translateY(-1px);
    }

    .dpad-cell:active, .dpad-cell.active-key {
      background: var(--cyan);
      border-color: var(--cyan);
      transform: scale(0.93);
    }

    .dpad-cell:active .dpad-sym, .dpad-cell.active-key .dpad-sym,
    .dpad-cell:active .dpad-sub, .dpad-cell.active-key .dpad-sub {
      color: #000;
    }

    .dpad-cell.stop-cell {
      background: rgba(244, 63, 94, 0.08);
      border-color: rgba(244, 63, 94, 0.25);
    }

    .dpad-cell.stop-cell .dpad-sym {
      color: var(--rose);
      font-size: 0.85rem;
    }

    .dpad-cell.stop-cell:hover {
      background: var(--rose);
      border-color: var(--rose);
    }

    .dpad-cell.stop-cell:hover .dpad-sym,
    .dpad-cell.stop-cell:hover .dpad-sub,
    .dpad-cell.stop-cell:active .dpad-sym,
    .dpad-cell.stop-cell:active .dpad-sub,
    .dpad-cell.stop-cell.active-key .dpad-sym,
    .dpad-cell.stop-cell.active-key .dpad-sub {
      color: #fff;
    }

    .dpad-cell.stop-cell:active, .dpad-cell.stop-cell.active-key {
      background: var(--rose);
      border-color: var(--rose);
      transform: scale(0.92);
    }

    .speed-pills-minimal {
      display: flex;
      gap: 0.35rem;
      width: 100%;
      max-width: 220px;
    }

    .speed-pill-min {
      flex: 1;
      padding: 0.3rem 0;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-card);
      border-radius: 6px;
      font-size: 0.68rem;
      font-weight: 600;
      color: var(--text-secondary);
      cursor: pointer;
      text-align: center;
      transition: all 0.15s;
    }

    .speed-pill-min:hover {
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-primary);
    }

    .speed-pill-min.active {
      background: rgba(6, 182, 212, 0.12);
      border-color: var(--cyan);
      color: var(--cyan);
    }

    /* VOICE COMMAND BAR */
    .command-bar {
      display: flex;
      gap: 0.5rem;
    }

    .command-input {
      flex: 1;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-card);
      border-radius: 10px;
      padding: 0.65rem 1rem;
      color: white;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.2s;
    }

    .command-input:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 10px var(--cyan-glow);
    }

    .mic-btn {
      background: linear-gradient(135deg, var(--cyan), var(--purple));
      border: none;
      width: 42px;
      height: 42px;
      border-radius: 10px;
      color: white;
      font-size: 1.1rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.2s;
    }

    .mic-btn.listening {
      background: var(--rose);
      animation: micPulse 1s infinite;
    }

    @keyframes micPulse {
      0%, 100% { box-shadow: 0 0 12px var(--rose); }
      50% { box-shadow: 0 0 24px var(--rose); }
    }

    /* 2D MAP CANVAS */
    #mapCanvas {
      width: 100%;
      height: 180px;
      background: #060911;
      border-radius: 12px;
      border: 1px solid var(--border-card);
    }

    /* LOGS CONSOLE */
    .log-terminal {
      background: #05070d;
      border: 1px solid var(--border-card);
      border-radius: 12px;
      padding: 0.75rem;
      height: 140px;
      overflow-y: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      display: flex;
      flex-direction: column-reverse;
      gap: 0.35rem;
    }

    .log-item {
      display: flex;
      gap: 0.5rem;
      line-height: 1.3;
    }

    .log-time {
      color: var(--text-muted);
    }

    .log-item.ALARM { color: #f87171; font-weight: bold; }
    .log-item.COMMAND { color: var(--cyan); }
    .log-item.SYSTEM { color: var(--emerald); }
    .log-item.INFO { color: var(--text-secondary); }
  </style>
</head>
<body>

  <!-- EMERGENCY ALARM BANNER -->
  <div id="alarmBanner">
    <span>🚨 WARNING: INTRUDER / OBSTACLE ALARM DETECTED!</span>
    <button class="alarm-btn" onclick="sendCommand('clear alarm')">RESET ALARM</button>
  </div>

  <!-- HEADER -->
  <header>
    <div class="brand">
      <div class="brand-icon">🤖</div>
      <div>
        <div class="brand-title">AI Household Service Robot</div>
        <div class="brand-subtitle">Real-time Simulation & Telemetry</div>
      </div>
    </div>
    <div class="header-badges">
      <div class="status-badge">
        <div class="status-dot"></div>
        <span id="headerMode">IDLE</span>
      </div>
    </div>
  </header>

  <!-- MAIN DASHBOARD -->
  <main>

    <!-- COLUMN 1: LIVE VIDEO STREAM & METRICS -->
    <div class="card" style="grid-row: span 2;">
      <div class="card-header">
        <div class="card-title">📹 Live Camera Feed</div>
        <div class="stream-toggle-group">
          <button id="btnRaw" class="stream-toggle-btn active" onclick="switchStream('raw')">Raw Camera</button>
          <button id="btnProcessed" class="stream-toggle-btn" onclick="switchStream('processed')">AI Vision</button>
        </div>
      </div>

      <div class="video-container">
        <img id="videoStream" src="/video_feed?type=raw" alt="Live Camera Feed">
        <div class="video-overlay-badge" id="streamTypeBadge">
          <span>📹</span> RAW STREAM
        </div>
      </div>

      <!-- METRICS GRID -->
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-icon" style="background: rgba(6, 182, 212, 0.15); color: var(--cyan);">📍</div>
          <div>
            <div class="metric-val" id="valPos">0.0, 0.0</div>
            <div class="metric-lbl">Position (X, Y) m</div>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-icon" style="background: rgba(168, 85, 247, 0.15); color: var(--purple);">🧭</div>
          <div>
            <div class="metric-val" id="valYaw">0.0°</div>
            <div class="metric-lbl">Robot Heading (Yaw)</div>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-icon" style="background: rgba(16, 185, 129, 0.15); color: var(--emerald);">⚡</div>
          <div>
            <div class="metric-val" id="valSpeed">0.0 m/s</div>
            <div class="metric-lbl">Linear Velocity</div>
          </div>
        </div>

        <div class="metric-card">
          <div class="metric-icon" style="background: rgba(245, 158, 11, 0.15); color: var(--amber);">🛡️</div>
          <div>
            <div class="metric-val" id="valObstacle">5.0 m</div>
            <div class="metric-lbl">Safety Distance</div>
          </div>
        </div>
      </div>

      <!-- BATTERY LEVEL -->
      <div class="battery-wrapper">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 600;">
          <span style="color: var(--text-secondary);">🔋 Battery Level</span>
          <span id="valBattery" style="color: var(--emerald);">100% (DISCHARGING)</span>
        </div>
        <div class="battery-bar-outer">
          <div class="battery-bar-inner" id="batteryBar"></div>
        </div>
      </div>
    </div>

    <!-- COLUMN 2: ROOM NAVIGATION & HOUSEHOLD TASKS -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">📍 Room Navigation</div>
      </div>
      <div class="btn-grid">
        <button class="action-btn" onclick="sendCommand('go to kitchen')">🍳 Kitchen</button>
        <button class="action-btn" onclick="sendCommand('go to bedroom')">🛏️ Bedroom</button>
        <button class="action-btn" onclick="sendCommand('go to living room')">🛋️ Living Room</button>
        <button class="action-btn" onclick="sendCommand('go to charging station')">⚡ Charger</button>
      </div>

      <div class="card-header" style="margin-top: 0.5rem;">
        <div class="card-title">📦 Service Tasks</div>
      </div>
      <div class="btn-grid">
        <button class="action-btn primary" onclick="sendCommand('bring water to bedroom')">💧 Deliver Water</button>
        <button class="action-btn primary" onclick="sendCommand('bring medicine to living room')">💊 Deliver Medicine</button>
      </div>

      <div class="card-header" style="margin-top: 0.5rem;">
        <div class="card-title">🛡️ Behaviors</div>
      </div>
      <div class="btn-grid">
        <button class="action-btn" onclick="sendCommand('start patrol')">🛡️ Start Patrol</button>
        <button class="action-btn" onclick="sendCommand('stop patrol')">⏹️ Stop Patrol</button>
        <button class="action-btn" onclick="sendCommand('follow me')">🚶 Follow Me</button>
        <button class="action-btn danger" onclick="sendCommand('stop')">🚨 EMERGENCY STOP</button>
      </div>
    </div>

    <!-- COLUMN 3: TELEOP CONTROLLER & VOICE COMMANDS -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">🕹️ Manual Teleop</div>
        <span style="font-size: 0.68rem; color: var(--text-muted); font-weight: 600;">WASD</span>
      </div>

      <div class="teleop-minimal">
        <div class="dpad-grid">
          <div></div>
          <button id="btnUp" class="dpad-cell" 
            onmousedown="startDrive(1, 0, 'btnUp')" onmouseup="stopDrive()" 
            ontouchstart="startDrive(1, 0, 'btnUp')" ontouchend="stopDrive()" title="Forward (W / ↑)">
            <span class="dpad-sym">▲</span>
            <span class="dpad-sub">W</span>
          </button>
          <div></div>

          <button id="btnLeft" class="dpad-cell" 
            onmousedown="startDrive(0, 1, 'btnLeft')" onmouseup="stopDrive()" 
            ontouchstart="startDrive(0, 1, 'btnLeft')" ontouchend="stopDrive()" title="Left (A / ←)">
            <span class="dpad-sym">◀</span>
            <span class="dpad-sub">A</span>
          </button>
          
          <button id="btnStop" class="dpad-cell stop-cell" 
            onclick="stopDrive()" title="Stop (Space)">
            <span class="dpad-sym">■</span>
            <span class="dpad-sub">SPACE</span>
          </button>

          <button id="btnRight" class="dpad-cell" 
            onmousedown="startDrive(0, -1, 'btnRight')" onmouseup="stopDrive()" 
            ontouchstart="startDrive(0, -1, 'btnRight')" ontouchend="stopDrive()" title="Right (D / →)">
            <span class="dpad-sym">▶</span>
            <span class="dpad-sub">D</span>
          </button>

          <div></div>
          <button id="btnDown" class="dpad-cell" 
            onmousedown="startDrive(-1, 0, 'btnDown')" onmouseup="stopDrive()" 
            ontouchstart="startDrive(-1, 0, 'btnDown')" ontouchend="stopDrive()" title="Reverse (S / ↓)">
            <span class="dpad-sym">▼</span>
            <span class="dpad-sub">S</span>
          </button>
          <div></div>
        </div>

        <div class="speed-pills-minimal">
          <div class="speed-pill-min" onclick="setSpeedPreset(0.20, 0.45, this)">0.20 m/s</div>
          <div class="speed-pill-min active" onclick="setSpeedPreset(0.35, 0.65, this)">0.35 m/s</div>
          <div class="speed-pill-min" onclick="setSpeedPreset(0.60, 1.00, this)">0.60 m/s</div>
        </div>
      </div>

      <div class="card-header" style="margin-top: 0.5rem;">
        <div class="card-title">🎙️ Voice & Text Command</div>
      </div>

      <div class="command-bar">
        <input type="text" id="cmdInput" class="command-input" placeholder="e.g. 'go to kitchen', 'start patrol'" onkeydown="if(event.key==='Enter') sendInputCmd()">
        <button id="micBtn" class="mic-btn" onclick="toggleMic()" title="Click to Speak">🎙️</button>
        <button class="action-btn primary" onclick="sendInputCmd()" style="padding: 0 1rem;">Send</button>
      </div>

      <!-- 2D FLOORPLAN MAP -->
      <div class="card-header" style="margin-top: 0.5rem;">
        <div class="card-title">🗺️ House Map Pose</div>
      </div>
      <canvas id="mapCanvas" width="300" height="180"></canvas>

      <!-- LOG CONSOLE -->
      <div class="card-header" style="margin-top: 0.5rem;">
        <div class="card-title">📜 Event Logs</div>
      </div>
      <div class="log-terminal" id="logTerminal"></div>
    </div>

  </main>

  <script>
    let currentStream = 'raw';
    let driveInterval = null;
    let recognition = null;
    let isListening = false;

    // Switch video stream between RAW and AI PROCESSED
    function switchStream(type) {
      currentStream = type;
      const img = document.getElementById('videoStream');
      const badge = document.getElementById('streamTypeBadge');
      const btnRaw = document.getElementById('btnRaw');
      const btnProc = document.getElementById('btnProcessed');

      if (type === 'processed') {
        img.src = '/video_feed?type=processed';
        badge.innerHTML = '<span>🧠</span> AI VISION';
        btnRaw.classList.remove('active');
        btnProc.classList.add('active');
      } else {
        img.src = '/video_feed?type=raw';
        badge.innerHTML = '<span>📹</span> RAW STREAM';
        btnProc.classList.remove('active');
        btnRaw.classList.add('active');
      }
    }

    // Send Voice / Action Command
    function sendCommand(cmdText) {
      fetch('/api/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmdText })
      })
      .then(res => res.json())
      .then(data => {
        // Speak in browser TTS as vocal acknowledgment
        speakFeedback(cmdText);
      })
      .catch(err => console.error(err));
    }

    function sendInputCmd() {
      const input = document.getElementById('cmdInput');
      const val = input.value.trim();
      if (val) {
        sendCommand(val);
        input.value = '';
      }
    }

    // Teleop Driving Logic
    let currentLinearSpeed = 0.35;
    let currentAngularSpeed = 0.65;

    function setSpeedPreset(linear, angular, el) {
      currentLinearSpeed = linear;
      currentAngularSpeed = angular;
      document.querySelectorAll('.speed-pill-min').forEach(p => p.classList.remove('active'));
      if (el) el.classList.add('active');
    }

    function sendTeleop(linear, angular) {
      fetch('/api/teleop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ linear: linear, angular: angular })
      }).catch(err => console.error(err));
    }

    function startDrive(linearFactor, angularFactor, keyId) {
      stopDrive();
      if (keyId) {
        const btn = document.getElementById(keyId);
        if (btn) btn.classList.add('active-key');
      }
      const lin = linearFactor * currentLinearSpeed;
      const ang = angularFactor * currentAngularSpeed;
      sendTeleop(lin, ang);
      driveInterval = setInterval(() => {
        sendTeleop(lin, ang);
      }, 100);
    }

    function stopDrive() {
      if (driveInterval) {
        clearInterval(driveInterval);
        driveInterval = null;
      }
      document.querySelectorAll('.dpad-cell').forEach(b => b.classList.remove('active-key'));
      sendTeleop(0, 0);
    }

    function updateTeleopStatus(lin, ang) {
      const lblLin = document.getElementById('teleopLinDisplay');
      const lblAng = document.getElementById('teleopAngDisplay');
      const badgeLin = document.getElementById('teleopLinBadge');
      const badgeAng = document.getElementById('teleopAngBadge');

      if (lblLin) lblLin.innerText = `Vx: ${lin.toFixed(2)} m/s`;
      if (lblAng) lblAng.innerText = `Wz: ${ang.toFixed(2)} rad/s`;

      if (badgeLin) {
        if (Math.abs(lin) > 0.01) badgeLin.classList.add('active');
        else badgeLin.classList.remove('active');
      }
      if (badgeAng) {
        if (Math.abs(ang) > 0.01) badgeAng.classList.add('active');
        else badgeAng.classList.remove('active');
      }
    }

    // Keyboard WASD Controls with interactive key glows
    window.addEventListener('keydown', (e) => {
      if (document.activeElement === document.getElementById('cmdInput')) return;
      if (e.repeat) return;
      if (e.key === 'w' || e.key === 'W' || e.key === 'ArrowUp') startDrive(1, 0, 'btnUp');
      else if (e.key === 's' || e.key === 'S' || e.key === 'ArrowDown') startDrive(-1, 0, 'btnDown');
      else if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') startDrive(0, 1, 'btnLeft');
      else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') startDrive(0, -1, 'btnRight');
      else if (e.key === ' ') {
        stopDrive();
        const btnStop = document.getElementById('btnStop');
        if (btnStop) btnStop.classList.add('active-key');
      }
    });

    window.addEventListener('keyup', (e) => {
      if (document.activeElement === document.getElementById('cmdInput')) return;
      if (['w', 's', 'a', 'd', 'W', 'S', 'A', 'D', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {
        stopDrive();
      }
    });

    // Browser Speech Synthesis (Audio response)
    function speakFeedback(text) {
      if ('speechSynthesis' in window) {
        const utter = new SpeechSynthesisUtterance("Command acknowledged: " + text);
        utter.rate = 1.05;
        window.speechSynthesis.speak(utter);
      }
    }

    // Web Speech API / Universal Microphone Recording
    function toggleMic() {
      const micBtn = document.getElementById('micBtn');
      const input = document.getElementById('cmdInput');
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

      // If browser supports Web Speech API (Chrome / Chromium / Edge / Safari)
      if (SpeechRecognition) {
        if (isListening) {
          try { recognition.stop(); } catch(e) {}
          return;
        }

        recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = false;

        recognition.onstart = () => {
          isListening = true;
          micBtn.classList.add('listening');
          input.placeholder = "🎙️ Listening... speak your command now!";
        };

        recognition.onresult = (event) => {
          const transcript = event.results[0][0].transcript;
          input.value = transcript;
          input.placeholder = "e.g. 'go to kitchen', 'start patrol'";
          sendCommand(transcript);
        };

        recognition.onerror = (event) => {
          console.warn("Client Speech Recognition unavailable, using server audio recording:", event.error);
          isListening = false;
          micBtn.classList.remove('listening');
          input.placeholder = "e.g. 'go to kitchen', 'start patrol'";
          fallbackServerVoiceRecord();
        };

        recognition.onend = () => {
          isListening = false;
          micBtn.classList.remove('listening');
          input.placeholder = "e.g. 'go to kitchen', 'start patrol'";
        };

        try {
          recognition.start();
        } catch(e) {
          fallbackServerVoiceRecord();
        }
      } else {
        // Mozilla Firefox, Zen Browser, Librewolf, or unsupported client: Server-side Microphone Recording
        fallbackServerVoiceRecord();
      }
    }

    // Server-side Microphone Recording Fallback (Works in 100% of all browsers)
    function fallbackServerVoiceRecord() {
      const micBtn = document.getElementById('micBtn');
      const input = document.getElementById('cmdInput');
      if (isListening) return;

      isListening = true;
      micBtn.classList.add('listening');
      input.placeholder = "🎙️ Listening for 4s... (Speak now!)";

      fetch('/api/voice_record', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
          isListening = false;
          micBtn.classList.remove('listening');
          input.placeholder = "e.g. 'go to kitchen', 'start patrol'";
          if (data.status === 'success' && data.transcript) {
            input.value = data.transcript;
            speakFeedback(data.transcript);
          } else if (data.message) {
            alert("Voice detection: " + data.message);
          }
        })
        .catch(err => {
          console.error("Server Voice Record Error:", err);
          isListening = false;
          micBtn.classList.remove('listening');
          input.placeholder = "e.g. 'go to kitchen', 'start patrol'";
          alert("Microphone recording error: " + err);
        });
    }

    // 2D Canvas Map Drawing
    function drawMap(x, y, yawDeg) {
      const canvas = document.getElementById('mapCanvas');
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;

      ctx.clearRect(0, 0, w, h);

      // Floor boundary
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.lineWidth = 2;
      ctx.strokeRect(15, 15, w - 30, h - 30);

      // Room Partitions
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.beginPath();
      ctx.moveTo(w / 2, 15); ctx.lineTo(w / 2, h - 15);
      ctx.moveTo(15, h / 2); ctx.lineTo(w - 15, h / 2);
      ctx.stroke();

      // Room Labels
      ctx.font = '10px Inter';
      ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.fillText('🍳 KITCHEN', 25, 32);
      ctx.fillText('🛏️ BEDROOM', w / 2 + 15, 32);
      ctx.fillText('🛋️ LIVING ROOM', 25, h / 2 + 22);
      ctx.fillText('⚡ CHARGER', w / 2 + 15, h / 2 + 22);

      // Convert world (X, Y) to canvas coords
      // House is approx [-3.5, 3.5] in X, [-3.5, 3.5] in Y
      const cx = (w / 2) + (y * 22);
      const cy = (h / 2) - (x * 22);

      // Draw Robot Marker
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate((-yawDeg * Math.PI) / 180);

      // Glow circle
      ctx.fillStyle = '#06b6d4';
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(0, 0, 7, 0, Math.PI * 2);
      ctx.fill();

      // Heading Arrow
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(12, 0);
      ctx.stroke();

      ctx.restore();
    }

    // Periodic Telemetry Fetching
    function updateStatus() {
      fetch('/api/status')
        .then(res => res.json())
        .then(data => {
          // Mode & State
          document.getElementById('headerMode').innerText = data.current_mode;
          document.getElementById('valPos').innerText = `${data.x}, ${data.y}`;
          document.getElementById('valYaw').innerText = `${data.yaw}°`;
          document.getElementById('valSpeed').innerText = `${data.linear_speed} m/s`;
          document.getElementById('valObstacle').innerText = `${data.min_laser_dist} m`;

          // Battery
          const batBar = document.getElementById('batteryBar');
          const batVal = document.getElementById('valBattery');
          batBar.style.width = data.battery + '%';
          batVal.innerText = `${data.battery}% (${data.battery_state})`;

          if (data.battery > 50) {
            batBar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
            batVal.style.color = 'var(--emerald)';
          } else if (data.battery > 20) {
            batBar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
            batVal.style.color = 'var(--amber)';
          } else {
            batBar.style.background = 'linear-gradient(90deg, #f43f5e, #f87171)';
            batVal.style.color = 'var(--rose)';
          }

          // Emergency Alarm Banner
          const banner = document.getElementById('alarmBanner');
          if (data.emergency_alarm) {
            banner.style.display = 'flex';
          } else {
            banner.style.display = 'none';
          }

          // Draw Map
          drawMap(data.x, data.y, data.yaw);

          // Logs
          const logTerm = document.getElementById('logTerminal');
          logTerm.innerHTML = data.logs.map(l =>
            `<div class="log-item ${l.category}">
              <span class="log-time">[${l.time}]</span>
              <span>${l.msg}</span>
             </div>`
          ).join('');
        })
        .catch(err => console.error(err));
    }

    setInterval(updateStatus, 250);
  </script>
</body>
</html>
"""


def main(args=None):
    global dashboard_node
    rclpy.init(args=args)
    dashboard_node = WebDashboardNode()

    # Spin ROS 2 in background thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(dashboard_node,), daemon=True)
    ros_thread.start()

    # Run Flask Web Server on 0.0.0.0:5000
    try:
        print("\n" + "=" * 60)
        print("  🌐 AI HOUSEHOLD SERVICE ROBOT - WEB DASHBOARD READY")
        print("  👉 Open in your browser: http://localhost:5000")
        print("=" * 60 + "\n")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except (KeyboardInterrupt, SystemExit, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if dashboard_node:
            dashboard_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
