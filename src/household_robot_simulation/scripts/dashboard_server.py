#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import json
import time
import math
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

# Global node reference
node = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Robot Status Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
        body {
            font-family: 'DM Sans', sans-serif;
        }
        .mono {
            font-family: 'JetBrains Mono', monospace;
        }
    </style>
</head>
<body class="bg-zinc-950 text-zinc-100 min-h-screen flex flex-col">

    <!-- Header -->
    <header class="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-md px-6 py-4 flex justify-between items-center sticky top-0 z-40">
        <div class="flex items-center space-x-3">
            <div class="bg-indigo-600 p-2.5 rounded-lg flex items-center justify-center">
                <i class="fa-solid fa-robot text-xl text-white"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold tracking-tight">Antigravity Service Robot</h1>
                <p class="text-xs text-zinc-400">ROS 2 Telemetry & Control Dashboard</p>
            </div>
        </div>
        <div class="flex items-center space-x-3">
            <span id="conn-badge" class="px-2.5 py-1 rounded-full text-xs font-semibold bg-red-950/50 text-red-400 border border-red-900/50 flex items-center space-x-1.5">
                <span class="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></span>
                <span>Disconnected</span>
            </span>
        </div>
    </header>

    <!-- Main Content Grid -->
    <main class="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-7xl mx-auto w-full">
        
        <!-- Left Side: Live Video Feed (7 Cols) -->
        <div class="lg:col-span-7 flex flex-col space-y-4">
            <div class="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden shadow-xl flex flex-col flex-1">
                <div class="border-b border-zinc-800 bg-zinc-900/50 px-5 py-3.5 flex justify-between items-center">
                    <h2 class="font-semibold text-sm flex items-center space-x-2 text-zinc-200">
                        <i class="fa-solid fa-video text-indigo-400"></i>
                        <span>Live YOLO Camera Feed</span>
                    </h2>
                    <span class="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 font-medium font-mono">30 FPS</span>
                </div>
                <div class="relative bg-zinc-950 flex-1 flex items-center justify-center min-h-[400px] overflow-hidden">
                    <img id="camera-feed" src="/video_feed" alt="Camera Feed" class="w-full h-full object-cover">
                    <!-- Emergency Overlay -->
                    <div id="alarm-overlay" class="hidden absolute inset-0 bg-red-950/20 border-4 border-red-500 animate-pulse pointer-events-none"></div>
                </div>
            </div>
        </div>

        <!-- Right Side: Telemetry & Controls (5 Cols) -->
        <div class="lg:col-span-5 flex flex-col space-y-6">
            
            <!-- Health, Status & Performance Panels (Grid) -->
            <div class="grid grid-cols-2 gap-4">
                
                <!-- Battery Card -->
                <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-4 shadow-md flex flex-col justify-between">
                    <div class="flex justify-between items-start">
                        <span class="text-xs text-zinc-400 font-medium">Battery Status</span>
                        <span id="bat-icon" class="text-zinc-500"><i class="fa-solid fa-battery-half text-lg"></i></span>
                    </div>
                    <div class="mt-3">
                        <h3 id="bat-level" class="text-2xl font-bold mono">100%</h3>
                        <p id="bat-state" class="text-xs text-zinc-400 mt-0.5">Standby</p>
                    </div>
                </div>

                <!-- Emergency State Card -->
                <div id="alarm-card" class="bg-zinc-900 border border-zinc-800 rounded-xl p-4 shadow-md flex flex-col justify-between transition-all duration-300">
                    <div class="flex justify-between items-start">
                        <span class="text-xs text-zinc-400 font-medium">Security State</span>
                        <span id="alarm-icon" class="text-zinc-500"><i class="fa-solid fa-shield-halved text-lg"></i></span>
                    </div>
                    <div class="mt-3">
                        <h3 id="alarm-text" class="text-xl font-bold">Secure</h3>
                        <p id="alarm-desc" class="text-xs text-zinc-400 mt-0.5">Monitoring Active</p>
                    </div>
                </div>

                <!-- Total Distance (Performance Log) -->
                <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-4 shadow-md flex flex-col justify-between">
                    <div class="flex justify-between items-start">
                        <span class="text-xs text-zinc-400 font-medium">Path Length</span>
                        <div class="flex items-center space-x-2">
                            <button onclick="resetMetrics()" title="Reset Performance Metrics" class="text-zinc-500 hover:text-indigo-400 active:scale-90 transition">
                                <i class="fa-solid fa-arrows-rotate text-xs"></i>
                            </button>
                            <span class="text-zinc-500"><i class="fa-solid fa-route text-lg"></i></span>
                        </div>
                    </div>
                    <div class="mt-3">
                        <h3 id="metrics-path" class="text-2xl font-bold mono">0.0 m</h3>
                        <p class="text-xs text-zinc-400 mt-0.5">Cumulative Distance</p>
                    </div>
                </div>

                <!-- Active Time (Performance Log) -->
                <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-4 shadow-md flex flex-col justify-between">
                    <div class="flex justify-between items-start">
                        <span class="text-xs text-zinc-400 font-medium">Active Duration</span>
                        <span class="text-zinc-500"><i class="fa-solid fa-hourglass-half text-lg"></i></span>
                    </div>
                    <div class="mt-3">
                        <h3 id="metrics-time" class="text-2xl font-bold mono">0s</h3>
                        <p class="text-xs text-zinc-400 mt-0.5">Time Spent Moving</p>
                    </div>
                </div>

                <!-- Pose Coordinates -->
                <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-4 shadow-md">
                    <span class="text-xs text-zinc-400 font-medium">Coordinates</span>
                    <div class="mt-2.5 space-y-1 text-sm font-semibold text-zinc-200">
                        <div class="flex justify-between">
                            <span class="text-zinc-500">X:</span>
                            <span id="pose-x" class="mono">0.00 m</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-zinc-500">Y:</span>
                            <span id="pose-y" class="mono">0.00 m</span>
                        </div>
                    </div>
                </div>

                <!-- Speed Card -->
                <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-4 shadow-md">
                    <span class="text-xs text-zinc-400 font-medium">Velocities</span>
                    <div class="mt-2.5 space-y-1 text-sm font-semibold text-zinc-200">
                        <div class="flex justify-between">
                            <span class="text-zinc-500">Linear:</span>
                            <span id="vel-linear" class="mono">0.00 m/s</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-zinc-500">Angular:</span>
                            <span id="vel-angular" class="mono">0.00 rad/s</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Current State Banner -->
            <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-4 shadow-md flex items-center justify-between">
                <div>
                    <span class="text-xs text-zinc-400 font-medium block">Current Task / Activity</span>
                    <span id="state-text" class="text-sm font-bold text-zinc-200 mt-0.5 block">Idle</span>
                </div>
                <span class="px-2.5 py-1 rounded bg-indigo-950/50 text-indigo-400 border border-indigo-900/50 text-xs font-semibold mono">ACTIVE</span>
            </div>

            <!-- Control Actions Panel -->
            <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-5 shadow-lg flex flex-col space-y-4">
                <h2 class="font-semibold text-sm text-zinc-200 flex items-center space-x-2">
                    <i class="fa-solid fa-gamepad text-indigo-400"></i>
                    <span>Interactive Robot Controls</span>
                </h2>
                
                <!-- Action Buttons -->
                <div class="grid grid-cols-2 gap-3">
                    <button onclick="sendCommand('patrol')" class="py-2.5 rounded-lg text-sm font-semibold bg-indigo-600 hover:bg-indigo-700 active:scale-[0.98] transition text-white shadow-md flex items-center justify-center space-x-2">
                        <i class="fa-solid fa-route"></i>
                        <span>Start Patrol</span>
                    </button>
                    <button onclick="sendCommand('stop')" class="py-2.5 rounded-lg text-sm font-semibold bg-red-600 hover:bg-red-700 active:scale-[0.98] transition text-white shadow-md flex items-center justify-center space-x-2">
                        <i class="fa-solid fa-hand"></i>
                        <span>Active Stop</span>
                    </button>
                    <button onclick="sendCommand('follow me')" class="py-2.5 rounded-lg text-sm font-semibold bg-zinc-800 hover:bg-zinc-700 active:scale-[0.98] transition text-zinc-200 shadow-md border border-zinc-700 flex items-center justify-center space-x-2">
                        <i class="fa-solid fa-person-walking"></i>
                        <span>Follow Me</span>
                    </button>
                    <button onclick="sendCommand('dock')" class="py-2.5 rounded-lg text-sm font-semibold bg-zinc-800 hover:bg-zinc-700 active:scale-[0.98] transition text-zinc-200 shadow-md border border-zinc-700 flex items-center justify-center space-x-2">
                        <i class="fa-solid fa-plug"></i>
                        <span>Go Charge</span>
                    </button>
                </div>

                <!-- Emergency Alarm Reset -->
                <button onclick="sendCommand('clear alarm')" class="w-full py-2 rounded-lg text-xs font-bold bg-zinc-950 text-zinc-400 hover:text-zinc-200 border border-zinc-800 hover:border-zinc-700 transition flex items-center justify-center space-x-2">
                    <i class="fa-solid fa-circle-check text-green-500"></i>
                    <span>Acknowledge & Reset Alarm</span>
                </button>

                <hr class="border-zinc-800 my-1">

                <!-- Advanced Delivery Dispatches -->
                <div class="space-y-3">
                    <h3 class="text-xs font-bold text-zinc-400 tracking-wider uppercase">Virtual Delivery Tasks</h3>
                    <div class="flex space-x-2.5">
                        <div class="flex-1">
                            <label class="text-[10px] text-zinc-500 block mb-1 font-bold">SELECT ITEM</label>
                            <select id="select-item" class="w-full bg-zinc-950 border border-zinc-800 rounded-lg py-2 px-3 text-xs text-zinc-300 focus:outline-none focus:border-indigo-500">
                                <option value="water">Water Bottle</option>
                                <option value="medicine">Medicine</option>
                            </select>
                        </div>
                        <div class="flex-1">
                            <label class="text-[10px] text-zinc-500 block mb-1 font-bold">DESTINATION</label>
                            <select id="select-dest" class="w-full bg-zinc-950 border border-zinc-800 rounded-lg py-2 px-3 text-xs text-zinc-300 focus:outline-none focus:border-indigo-500">
                                <option value="bedroom">Bedroom</option>
                                <option value="living room">Living Room</option>
                                <option value="kitchen">Kitchen</option>
                            </select>
                        </div>
                    </div>
                    <button onclick="dispatchDelivery()" class="w-full py-2.5 rounded-lg text-xs font-bold bg-indigo-600/20 text-indigo-400 border border-indigo-500/20 hover:bg-indigo-600 hover:text-white transition flex items-center justify-center space-x-2 shadow-sm">
                        <i class="fa-solid fa-truck-ramp-box"></i>
                        <span>Dispatch Delivery Plan</span>
                    </button>
                </div>
            </div>

            <!-- Command Log Card -->
            <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-5 shadow-lg flex-1 flex flex-col min-h-[220px]">
                <h2 class="font-semibold text-sm text-zinc-200 flex items-center space-x-2 mb-3">
                    <i class="fa-solid fa-clock-rotate-left text-zinc-400"></i>
                    <span>Voice Command Logs</span>
                </h2>
                <div class="flex-1 overflow-y-auto max-h-[160px] pr-2.5 space-y-2" id="command-list">
                    <div class="text-zinc-500 text-xs italic text-center py-6">No voice commands received yet.</div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="border-t border-zinc-800 py-3 text-center text-xs text-zinc-500 bg-zinc-950 mt-auto">
        Antigravity Coding Assistant Dashboard System. Open-source ROS 2 UI.
    </footer>

    <!-- WebSockets/SSE Logic -->
    <script>
        const source = new EventSource("/telemetry");
        
        source.onopen = function() {
            const badge = document.getElementById("conn-badge");
            badge.className = "px-2.5 py-1 rounded-full text-xs font-semibold bg-green-950/50 text-green-400 border border-green-900/50 flex items-center space-x-1.5";
            badge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span><span>Connected</span>`;
        };

        source.onerror = function() {
            const badge = document.getElementById("conn-badge");
            badge.className = "px-2.5 py-1 rounded-full text-xs font-semibold bg-red-950/50 text-red-400 border border-red-900/50 flex items-center space-x-1.5";
            badge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse"></span><span>Disconnected</span>`;
        };

        source.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            // 1. Update Battery
            const batLevel = parseFloat(data.battery).toFixed(1);
            document.getElementById("bat-level").innerText = batLevel + "%";
            const batState = document.getElementById("bat-state");
            const batIcon = document.getElementById("bat-icon");
            
            if (batLevel == 100.0) {
                batState.innerText = "Fully Charged";
            } else {
                batState.innerText = "Discharging";
            }
            
            if (data.state.toLowerCase().includes("charge") || data.state.toLowerCase().includes("dock")) {
                batState.innerText = "Charging";
                batIcon.innerHTML = `<i class="fa-solid fa-bolt text-lg text-yellow-500 animate-pulse"></i>`;
            } else {
                if (batLevel > 50.0) {
                    batIcon.innerHTML = `<i class="fa-solid fa-battery-three-quarters text-lg text-green-500"></i>`;
                } else if (batLevel > 20.0) {
                    batIcon.innerHTML = `<i class="fa-solid fa-battery-half text-lg text-yellow-500"></i>`;
                } else {
                    batIcon.innerHTML = `<i class="fa-solid fa-battery-quarter text-lg text-red-500 animate-bounce"></i>`;
                }
            }

            // 2. Update Emergency Alarm Status
            const alarmCard = document.getElementById("alarm-card");
            const alarmIcon = document.getElementById("alarm-icon");
            const alarmText = document.getElementById("alarm-text");
            const alarmDesc = document.getElementById("alarm-desc");
            const alarmOverlay = document.getElementById("alarm-overlay");
            
            if (data.emergency) {
                alarmCard.className = "bg-red-950/30 border border-red-800 rounded-xl p-4 shadow-md flex flex-col justify-between animate-pulse";
                alarmIcon.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-lg text-red-500"></i>`;
                alarmText.innerText = "ALERT ACTIVE";
                alarmText.className = "text-xl font-bold text-red-500";
                alarmDesc.innerText = "Robot Brakes Locked";
                alarmOverlay.classList.remove("hidden");
            } else {
                alarmCard.className = "bg-zinc-900 border border-zinc-800 rounded-xl p-4 shadow-md flex flex-col justify-between";
                alarmIcon.innerHTML = `<i class="fa-solid fa-shield-halved text-lg text-zinc-500"></i>`;
                alarmText.innerText = "Secure";
                alarmText.className = "text-xl font-bold text-zinc-100";
                alarmDesc.innerText = "Monitoring Active";
                alarmOverlay.classList.add("hidden");
            }

            // 3. Update Performance Metrics
            document.getElementById("metrics-path").innerText = parseFloat(data.path_length).toFixed(1) + " m";
            document.getElementById("metrics-time").innerText = Math.round(data.active_duration) + " s";

            // 4. Update Pose
            document.getElementById("pose-x").innerText = parseFloat(data.pose.x).toFixed(2) + " m";
            document.getElementById("pose-y").innerText = parseFloat(data.pose.y).toFixed(2) + " m";

            // 5. Update Velocities
            document.getElementById("vel-linear").innerText = parseFloat(data.speed.linear).toFixed(2) + " m/s";
            document.getElementById("vel-angular").innerText = parseFloat(data.speed.angular).toFixed(2) + " rad/s";

            // 6. Update State
            document.getElementById("state-text").innerText = data.state;

            // 7. Update Command Log
            const cmdList = document.getElementById("command-list");
            if (data.history && data.history.length > 0) {
                cmdList.innerHTML = data.history.map(item => `
                    <div class="flex items-start space-x-3 p-2 bg-zinc-950 border border-zinc-800/80 rounded-lg text-xs">
                        <span class="mono text-zinc-500 font-medium">${item.time}</span>
                        <span class="font-semibold text-zinc-300 flex-1">${item.command}</span>
                    </div>
                `).join("");
            } else {
                cmdList.innerHTML = `<div class="text-zinc-500 text-xs italic text-center py-6">No voice commands received yet.</div>`;
            }
        };

        // Post Command API Call
        function sendCommand(cmd) {
            fetch("/api/command", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ command: cmd })
            })
            .then(res => res.json())
            .then(data => {
                console.log("Command dispatched successfully:", data);
            })
            .catch(err => {
                console.error("Failed to send command:", err);
            });
        }

        // Reset Performance Metrics API
        function resetMetrics() {
            fetch("/api/reset_metrics", {
                method: "POST"
            })
            .then(res => res.json())
            .then(data => {
                console.log("Performance metrics reset successfully.");
            })
            .catch(err => {
                console.error("Failed to reset metrics:", err);
            });
        }

        // Dispatch Custom Delivery Plan
        function dispatchDelivery() {
            const item = document.getElementById("select-item").value;
            const dest = document.getElementById("select-dest").value;
            const voiceCmd = `deliver ${item} to ${dest}`;
            sendCommand(voiceCmd);
        }
    </script>
</body>
</html>
"""

class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_server')
        self.bridge = CvBridge()
        
        # Telemetry Cache
        self.battery_level = 100.0
        self.emergency_active = False
        self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.current_speed = {'linear': 0.0, 'angular': 0.0}
        self.latest_frame = None
        self.command_history = []
        self.max_history = 10
        self.current_state = "Idle"
        
        # Performance Log Variables
        self.total_path_length = 0.0
        self.last_pose = None
        self.active_duration = 0.0
        self.last_time = None
        self.start_battery = 100.0
        self.battery_used = 0.0
        
        # Subscriptions
        self.battery_sub = self.create_subscription(
            Float32, '/battery/percentage', self.battery_callback, 10)
        self.emergency_sub = self.create_subscription(
            Bool, '/emergency/alarm', self.emergency_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_processed', self.image_callback, 10)
        self.cmd_sub = self.create_subscription(
            String, '/voice/command', self.command_callback, 10)
            
        # Publisher to trigger actions
        self.cmd_pub = self.create_publisher(String, '/voice/command', 10)
        
        self.get_logger().info("Dashboard ROS 2 Node Initialized with Performance Logs.")

    def battery_callback(self, msg: Float32):
        self.battery_level = msg.data
        if self.start_battery == 100.0 and msg.data < 100.0:
            self.start_battery = msg.data
        self.battery_used = max(0.0, self.start_battery - msg.data)

    def emergency_callback(self, msg: Bool):
        self.emergency_active = msg.data

    def odom_callback(self, msg: Odometry):
        # Update current pose
        self.current_pose['x'] = msg.pose.pose.position.x
        self.current_pose['y'] = msg.pose.pose.position.y
        self.current_speed['linear'] = msg.twist.twist.linear.x
        self.current_speed['angular'] = msg.twist.twist.angular.z
        
        # Calculate cumulative path length
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_pose is not None:
            dx = x - self.last_pose['x']
            dy = y - self.last_pose['y']
            dist = math.sqrt(dx*dx + dy*dy)
            # Filter out minor odometry noise when standing still
            if dist > 0.002:
                self.total_path_length += dist
        self.last_pose = {'x': x, 'y': y}

        # Calculate active travel duration
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            # Count moving time if velocity is significant
            if abs(msg.twist.twist.linear.x) > 0.01 or abs(msg.twist.twist.angular.z) > 0.01:
                self.active_duration += dt
        self.last_time = now

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, jpeg = cv2.imencode('.jpg', cv_image)
            self.latest_frame = jpeg.tobytes()
        except Exception as e:
            pass

    def command_callback(self, msg: String):
        cmd = msg.data
        self.command_history.insert(0, {
            'time': time.strftime('%H:%M:%S'),
            'command': cmd
        })
        if len(self.command_history) > self.max_history:
            self.command_history.pop()
            
        # Track current state
        cmd_lower = cmd.lower()
        if "follow" in cmd_lower:
            self.current_state = "Following Person"
        elif "patrol" in cmd_lower:
            self.current_state = "Patrolling"
        elif "stop" in cmd_lower or "halt" in cmd_lower:
            self.current_state = "Stopped"
        elif "deliver" in cmd_lower or "bring" in cmd_lower:
            self.current_state = "Delivering Item"
        elif "dock" in cmd_lower or "charge" in cmd_lower:
            self.current_state = "Returning to Charger"

    def publish_command(self, cmd_text: str):
        msg = String()
        msg.data = cmd_text
        self.cmd_pub.publish(msg)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def gen_frames():
    while True:
        if node and node.latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + node.latest_frame + b'\r\n')
        else:
            time.sleep(0.1)
        time.sleep(0.033)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/telemetry')
def telemetry():
    def event_stream():
        while True:
            if node:
                data = {
                    'battery': node.battery_level,
                    'emergency': node.emergency_active,
                    'pose': node.current_pose,
                    'speed': node.current_speed,
                    'state': node.current_state,
                    'history': node.command_history,
                    'path_length': node.total_path_length,
                    'active_duration': node.active_duration,
                    'battery_used': node.battery_used
                }
                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.2)  # 5Hz updates
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/command', methods=['POST'])
def send_command():
    data = request.json
    cmd = data.get('command')
    if cmd and node:
        node.publish_command(cmd)
        return jsonify({'status': 'success', 'command': cmd})
    return jsonify({'status': 'error', 'message': 'Invalid command'}), 400

@app.route('/api/reset_metrics', methods=['POST'])
def reset_metrics():
    if node:
        node.total_path_length = 0.0
        node.active_duration = 0.0
        node.start_battery = node.battery_level
        node.battery_used = 0.0
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'ROS Node not ready'}), 400

def main(args=None):
    global node
    rclpy.init(args=args)
    node = DashboardNode()
    
    # Run ROS 2 spin in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
