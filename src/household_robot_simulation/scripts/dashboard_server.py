#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import json
import time
import math
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from flask import Flask, Response, render_template_string, request, jsonify

app = Flask(__name__)

# Global node reference
node = None

# Config File Paths (Src and Share)
CONFIG_PATHS = [
    '/home/rudrarathod/ros2_ws/src/household_robot_simulation/config/semantic_locations.yaml',
]
try:
    pkg_share = get_package_share_directory('household_robot_simulation')
    installed_p = os.path.join(pkg_share, 'config', 'semantic_locations.yaml')
    if installed_p not in CONFIG_PATHS:
        CONFIG_PATHS.append(installed_p)
except Exception:
    pass

def load_semantic_config():
    for p in CONFIG_PATHS:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                print(f"Error loading {p}: {e}")
    return {'locations': {}, 'patrol_waypoints': []}

def save_semantic_config(data):
    success = False
    for p in CONFIG_PATHS:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            success = True
        except Exception as e:
            print(f"Error saving to {p}: {e}")
    return success

def create_standby_frame(text="WAITING FOR CAMERA FEED", subtext="Topic: /camera/image_raw"):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (18, 18, 24)  # dark modern background
    
    # Grid pattern (subtle)
    for y in range(0, 480, 40):
        cv2.line(img, (0, y), (640, y), (26, 26, 34), 1)
    for x in range(0, 640, 40):
        cv2.line(img, (x, 0), (x, 480), (26, 26, 34), 1)

    # Viewfinder corners
    c_len = 25
    c_color = (99, 102, 241) # indigo
    cv2.line(img, (30, 30), (30 + c_len, 30), c_color, 2)
    cv2.line(img, (30, 30), (30, 30 + c_len), c_color, 2)
    cv2.line(img, (610, 30), (610 - c_len, 30), c_color, 2)
    cv2.line(img, (610, 30), (610, 30 + c_len), c_color, 2)
    cv2.line(img, (30, 450), (30 + c_len, 450), c_color, 2)
    cv2.line(img, (30, 450), (30, 450 - c_len), c_color, 2)
    cv2.line(img, (610, 450), (610 - c_len, 450), c_color, 2)
    cv2.line(img, (610, 450), (610, 450 - c_len), c_color, 2)

    # Center crosshair
    cv2.circle(img, (320, 240), 30, (50, 50, 65), 1)
    cv2.line(img, (310, 240), (330, 240), (99, 102, 241), 1)
    cv2.line(img, (320, 230), (320, 250), (99, 102, 241), 1)

    # Text overlay
    cv2.putText(img, text, (170, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 225, 230), 2, cv2.LINE_AA)
    cv2.putText(img, subtext, (190, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 145, 155), 1, cv2.LINE_AA)
    cv2.putText(img, "RESOLUTION: 640x480 | 30 FPS", (205, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 105, 115), 1, cv2.LINE_AA)

    _, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes()

STANDBY_RAW_FRAME = create_standby_frame("STANDBY - NO RAW FEED", "Awaiting: /camera/image_raw")
STANDBY_PROC_FRAME = create_standby_frame("STANDBY - NO PROCESSED FEED", "Awaiting: /camera/image_processed")

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
            <div class="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden shadow-xl flex flex-col">
                <div class="border-b border-zinc-800 bg-zinc-900/50 px-5 py-3.5 flex justify-between items-center">
                    <div class="flex items-center space-x-2.5">
                        <span class="relative flex h-2.5 w-2.5">
                            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                            <span class="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                        </span>
                        <h2 class="font-semibold text-sm flex items-center space-x-2 text-zinc-200">
                            <i class="fa-solid fa-video text-indigo-400"></i>
                            <span id="camera-feed-title">Live Camera Feed (Raw)</span>
                        </h2>
                    </div>
                    <div class="flex items-center space-x-2">
                        <div class="inline-flex bg-zinc-950 p-0.5 rounded-lg border border-zinc-800 text-xs">
                            <button id="btn-feed-raw" onclick="setFeedType('raw')" class="px-2.5 py-1 rounded-md font-medium text-white bg-indigo-600 transition">Raw</button>
                            <button id="btn-feed-processed" onclick="setFeedType('processed')" class="px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-zinc-200 transition">AI / Processed</button>
                        </div>
                        <span class="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 font-medium font-mono">640x480</span>
                        <span class="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 font-medium font-mono">30 FPS</span>
                    </div>
                </div>

                <!-- Proper 4:3 Camera Viewport -->
                <div class="relative bg-zinc-950 w-full aspect-[4/3] flex items-center justify-center overflow-hidden border-b border-zinc-800/80">
                    <img id="camera-feed" src="/video_feed?type=raw" alt="Camera Feed" class="w-full h-full object-contain">
                    
                    <!-- Viewfinder HUD Corner Accents -->
                    <div class="absolute inset-4 pointer-events-none flex flex-col justify-between p-2">
                        <div class="flex justify-between">
                            <div class="w-4 h-4 border-t-2 border-l-2 border-indigo-500/70"></div>
                            <div class="w-4 h-4 border-t-2 border-r-2 border-indigo-500/70"></div>
                        </div>
                        <!-- Center Reticle -->
                        <div class="self-center flex items-center justify-center opacity-30">
                            <div class="w-6 h-0.5 bg-indigo-400"></div>
                            <div class="w-0.5 h-6 bg-indigo-400 absolute"></div>
                        </div>
                        <div class="flex justify-between">
                            <div class="w-4 h-4 border-b-2 border-l-2 border-indigo-500/70"></div>
                            <div class="w-4 h-4 border-b-2 border-r-2 border-indigo-500/70"></div>
                        </div>
                    </div>

                    <!-- HUD Status Tag -->
                    <div class="absolute top-3 left-3 bg-black/70 backdrop-blur-sm px-2.5 py-1 rounded-md border border-white/10 text-[11px] font-mono text-zinc-300 flex items-center space-x-1.5 pointer-events-none">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                        <span id="hud-feed-label">CAM: RAW (/camera/image_raw)</span>
                    </div>

                    <!-- Emergency Overlay -->
                    <div id="alarm-overlay" class="hidden absolute inset-0 bg-red-950/40 border-4 border-red-500 animate-pulse pointer-events-none flex items-center justify-center">
                        <div class="bg-red-900/90 text-white font-bold px-4 py-2 rounded-lg text-sm tracking-wider uppercase border border-red-500 shadow-2xl flex items-center space-x-2">
                            <i class="fa-solid fa-triangle-exclamation animate-bounce"></i>
                            <span>EMERGENCY OBSTACLE STOP</span>
                        </div>
                    </div>
                </div>

                <!-- Feed Quick Info Bar -->
                <div class="px-5 py-2.5 bg-zinc-900/70 text-xs text-zinc-400 flex justify-between items-center border-t border-zinc-800/50 font-mono">
                    <div class="flex items-center space-x-3">
                        <span>FOV: 60°</span>
                        <span>•</span>
                        <span>Format: RGB8</span>
                        <span>•</span>
                        <span>Ratio: 4:3</span>
                    </div>
                    <div class="flex items-center space-x-1.5 text-zinc-400">
                        <i class="fa-solid fa-signal text-emerald-400 text-[10px]"></i>
                        <span class="text-zinc-300">Live Stream</span>
                    </div>
                </div>
            </div>

            <!-- Manual Teleoperation Controller Card -->
            <div class="bg-zinc-900 border border-zinc-800 rounded-xl p-5 shadow-xl flex flex-col space-y-4">
                <div class="flex justify-between items-center border-b border-zinc-800/80 pb-3">
                    <h2 class="font-semibold text-sm text-zinc-200 flex items-center space-x-2">
                        <i class="fa-solid fa-gamepad text-indigo-400"></i>
                        <span>Manual Teleop Controller</span>
                    </h2>
                    <span class="px-2.5 py-0.5 rounded bg-indigo-950/60 text-indigo-400 border border-indigo-900/60 text-[11px] font-mono font-semibold flex items-center space-x-1.5">
                        <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-ping"></span>
                        <span>HOLD WASD / ARROWS</span>
                    </span>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-12 gap-5 items-center">
                    <!-- Left: On-Screen D-Pad -->
                    <div class="md:col-span-5 flex justify-center">
                        <div class="grid grid-cols-3 gap-2 w-44 select-none">
                            <div></div>
                            <button id="btn-dpad-w" title="Forward (W / Up Arrow)" class="p-3.5 bg-zinc-800 hover:bg-indigo-600 active:bg-indigo-500 rounded-xl text-zinc-200 hover:text-white font-bold flex flex-col items-center justify-center transition touch-none shadow-md border border-zinc-700 select-none">
                                <i class="fa-solid fa-arrow-up text-lg"></i>
                                <span class="text-[10px] font-mono opacity-60">W</span>
                            </button>
                            <div></div>

                            <button id="btn-dpad-a" title="Turn Left (A / Left Arrow)" class="p-3.5 bg-zinc-800 hover:bg-indigo-600 active:bg-indigo-500 rounded-xl text-zinc-200 hover:text-white font-bold flex flex-col items-center justify-center transition touch-none shadow-md border border-zinc-700 select-none">
                                <i class="fa-solid fa-arrow-left text-lg"></i>
                                <span class="text-[10px] font-mono opacity-60">A</span>
                            </button>
                            <button id="btn-dpad-stop" title="Emergency Active Stop (Space)" class="p-3.5 bg-red-950/40 hover:bg-red-600 active:bg-red-500 text-red-400 hover:text-white rounded-xl font-bold flex flex-col items-center justify-center transition border border-red-800/60 shadow-md select-none">
                                <i class="fa-solid fa-stop text-lg"></i>
                                <span class="text-[10px] font-mono opacity-80">STOP</span>
                            </button>
                            <button id="btn-dpad-d" title="Turn Right (D / Right Arrow)" class="p-3.5 bg-zinc-800 hover:bg-indigo-600 active:bg-indigo-500 rounded-xl text-zinc-200 hover:text-white font-bold flex flex-col items-center justify-center transition touch-none shadow-md border border-zinc-700 select-none">
                                <i class="fa-solid fa-arrow-right text-lg"></i>
                                <span class="text-[10px] font-mono opacity-60">D</span>
                            </button>

                            <div></div>
                            <button id="btn-dpad-s" title="Backward (S / Down Arrow)" class="p-3.5 bg-zinc-800 hover:bg-indigo-600 active:bg-indigo-500 rounded-xl text-zinc-200 hover:text-white font-bold flex flex-col items-center justify-center transition touch-none shadow-md border border-zinc-700 select-none">
                                <i class="fa-solid fa-arrow-down text-lg"></i>
                                <span class="text-[10px] font-mono opacity-60">S</span>
                            </button>
                            <div></div>
                        </div>
                    </div>

                    <!-- Right: Velocity Calibration & Keyboard Guide -->
                    <div class="md:col-span-7 flex flex-col justify-between space-y-3.5 border-t md:border-t-0 md:border-l border-zinc-800/80 md:pl-5 pt-3 md:pt-0">
                        <!-- Linear Speed Slider -->
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1.5">
                                <span class="text-zinc-400">Linear Velocity</span>
                                <span id="label-linear-speed" class="mono text-indigo-400 font-bold">1.0 m/s</span>
                            </div>
                            <div class="flex items-center space-x-2">
                                <button onclick="adjustSpeed(-0.2, 0)" class="w-7 h-7 bg-zinc-800 hover:bg-zinc-700 active:scale-95 rounded-lg text-xs text-zinc-300 font-mono font-bold flex items-center justify-center border border-zinc-700 transition">-</button>
                                <input id="slider-linear-speed" type="range" min="0.2" max="3.0" step="0.1" value="1.0" class="w-full accent-indigo-500 cursor-pointer" oninput="updateSpeedFromSlider()">
                                <button onclick="adjustSpeed(0.2, 0)" class="w-7 h-7 bg-zinc-800 hover:bg-zinc-700 active:scale-95 rounded-lg text-xs text-zinc-300 font-mono font-bold flex items-center justify-center border border-zinc-700 transition">+</button>
                            </div>
                        </div>

                        <!-- Angular Speed Slider -->
                        <div>
                            <div class="flex justify-between text-xs font-semibold mb-1.5">
                                <span class="text-zinc-400">Angular Velocity</span>
                                <span id="label-angular-speed" class="mono text-indigo-400 font-bold">2.0 rad/s</span>
                            </div>
                            <div class="flex items-center space-x-2">
                                <button onclick="adjustSpeed(0, -0.5)" class="w-7 h-7 bg-zinc-800 hover:bg-zinc-700 active:scale-95 rounded-lg text-xs text-zinc-300 font-mono font-bold flex items-center justify-center border border-zinc-700 transition">-</button>
                                <input id="slider-angular-speed" type="range" min="0.5" max="5.0" step="0.5" value="2.0" class="w-full accent-indigo-500 cursor-pointer" oninput="updateSpeedFromSlider()">
                                <button onclick="adjustSpeed(0, 0.5)" class="w-7 h-7 bg-zinc-800 hover:bg-zinc-700 active:scale-95 rounded-lg text-xs text-zinc-300 font-mono font-bold flex items-center justify-center border border-zinc-700 transition">+</button>
                            </div>
                        </div>

                        <!-- Live Key Feedback / Instructions -->
                        <div class="text-[11px] text-zinc-400 bg-zinc-950/80 p-2.5 rounded-lg border border-zinc-800 flex items-center space-x-2.5">
                            <i class="fa-solid fa-keyboard text-indigo-400 text-sm"></i>
                            <span><b>WASD</b> / <b>Arrows</b> to drive • <b>Space</b> to Stop • <b>+/-</b> to adjust speed.</span>
                        </div>
                    </div>
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

        <!-- Bottom Full-Width: Semantic Locations & Patrol Manager (12 Cols) -->
        <div class="lg:col-span-12 bg-zinc-900 border border-zinc-800 rounded-xl p-6 shadow-xl flex flex-col space-y-5">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-4">
                <div class="flex items-center space-x-3">
                    <div class="p-2.5 rounded-lg bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
                        <i class="fa-solid fa-map-location-dot text-xl"></i>
                    </div>
                    <div>
                        <h2 class="font-bold text-base text-zinc-100">Semantic Locations & Patrol Path Manager</h2>
                        <p class="text-xs text-zinc-400">Live remap room coordinates, capture robot's current pose, and construct custom patrol routes.</p>
                    </div>
                </div>

                <!-- Tab Selector Buttons & Status -->
                <div class="flex items-center space-x-2">
                    <div class="inline-flex bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs">
                        <button id="tab-btn-locations" onclick="switchManagerTab('locations')" class="px-3 py-1.5 rounded-md font-semibold text-white bg-indigo-600 transition flex items-center space-x-1.5">
                            <i class="fa-solid fa-house-signal"></i>
                            <span>Room Locations</span>
                        </button>
                        <button id="tab-btn-patrol" onclick="switchManagerTab('patrol')" class="px-3 py-1.5 rounded-md font-semibold text-zinc-400 hover:text-zinc-200 transition flex items-center space-x-1.5">
                            <i class="fa-solid fa-route"></i>
                            <span>Patrol Waypoints</span>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Tab 1: Room Locations Management -->
            <div id="tab-pane-locations" class="space-y-4">
                <div class="flex flex-wrap justify-between items-center gap-3">
                    <div class="text-xs text-zinc-400">
                        Define target rooms for voice and delivery dispatch commands (<span class="text-zinc-200 font-mono">"go to &lt;room&gt;"</span>).
                    </div>
                    <div class="flex items-center space-x-2">
                        <button onclick="addCurrentPoseAsRoom()" class="px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600 text-emerald-400 hover:text-white border border-emerald-500/30 rounded-lg text-xs font-semibold transition flex items-center space-x-1.5 shadow-sm">
                            <i class="fa-solid fa-location-crosshairs"></i>
                            <span>Save Current Pose as Room</span>
                        </button>
                        <button onclick="addNewRoomRow()" class="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-xs font-semibold transition flex items-center space-x-1.5 border border-zinc-700">
                            <i class="fa-solid fa-plus"></i>
                            <span>Add Custom Room</span>
                        </button>
                        <button onclick="saveLocationsToServer()" class="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold transition flex items-center space-x-1.5 shadow-md">
                            <i class="fa-solid fa-floppy-disk"></i>
                            <span>Save Locations Database</span>
                        </button>
                    </div>
                </div>

                <!-- Locations Table -->
                <div class="overflow-x-auto border border-zinc-800 rounded-lg bg-zinc-950">
                    <table class="w-full text-left text-xs text-zinc-300">
                        <thead class="bg-zinc-900/80 text-[11px] text-zinc-400 uppercase font-mono border-b border-zinc-800">
                            <tr>
                                <th class="p-3">Room / Location Name</th>
                                <th class="p-3">X (meters)</th>
                                <th class="p-3">Y (meters)</th>
                                <th class="p-3">Yaw (rad)</th>
                                <th class="p-3 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody id="locations-table-body" class="divide-y divide-zinc-800/60 font-mono">
                            <!-- Populated via JS -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Tab 2: Custom Patrol Waypoints Management -->
            <div id="tab-pane-patrol" class="hidden space-y-4">
                <div class="flex flex-wrap justify-between items-center gap-3">
                    <div class="text-xs text-zinc-400">
                        Create an ordered waypoint patrol loop. Robot will cycle through points 1 &rarr; N &rarr; 1 when <span class="text-zinc-200 font-mono">"patrol"</span> is triggered.
                    </div>
                    <div class="flex items-center space-x-2">
                        <button onclick="addCurrentPoseAsWaypoint()" class="px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600 text-emerald-400 hover:text-white border border-emerald-500/30 rounded-lg text-xs font-semibold transition flex items-center space-x-1.5 shadow-sm">
                            <i class="fa-solid fa-map-pin"></i>
                            <span>Add Current Pose as Waypoint</span>
                        </button>
                        <button onclick="addNewWaypointRow()" class="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg text-xs font-semibold transition flex items-center space-x-1.5 border border-zinc-700">
                            <i class="fa-solid fa-plus"></i>
                            <span>Add Empty Waypoint</span>
                        </button>
                        <button onclick="savePatrolToServer()" class="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-xs font-semibold transition flex items-center space-x-1.5 shadow-md">
                            <i class="fa-solid fa-floppy-disk"></i>
                            <span>Save Patrol Path</span>
                        </button>
                        <button onclick="sendCommand('patrol')" class="px-4 py-1.5 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-xs font-semibold transition flex items-center space-x-1.5 shadow-md">
                            <i class="fa-solid fa-play"></i>
                            <span>Start Patrol Now</span>
                        </button>
                    </div>
                </div>

                <!-- Patrol Waypoints Table -->
                <div class="overflow-x-auto border border-zinc-800 rounded-lg bg-zinc-950">
                    <table class="w-full text-left text-xs text-zinc-300">
                        <thead class="bg-zinc-900/80 text-[11px] text-zinc-400 uppercase font-mono border-b border-zinc-800">
                            <tr>
                                <th class="p-3 w-16">Step</th>
                                <th class="p-3">X Coordinate (m)</th>
                                <th class="p-3">Y Coordinate (m)</th>
                                <th class="p-3">Yaw Angle (rad)</th>
                                <th class="p-3 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody id="patrol-table-body" class="divide-y divide-zinc-800/60 font-mono">
                            <!-- Populated via JS -->
                        </tbody>
                    </table>
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
            if (data.pose) {
                livePose.x = parseFloat(data.pose.x) || 0.0;
                livePose.y = parseFloat(data.pose.y) || 0.0;
                livePose.theta = parseFloat(data.pose.theta) || 0.0;
            }
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

        // Video Feed Switcher
        function setFeedType(type) {
            const btnRaw = document.getElementById("btn-feed-raw");
            const btnProc = document.getElementById("btn-feed-processed");
            const title = document.getElementById("camera-feed-title");
            const hudLabel = document.getElementById("hud-feed-label");
            const feedImg = document.getElementById("camera-feed");
            
            if (type === 'raw') {
                btnRaw.className = "px-2.5 py-1 rounded-md font-medium text-white bg-indigo-600 transition";
                btnProc.className = "px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-zinc-200 transition";
                title.innerText = "Live Camera Feed (Raw)";
                if (hudLabel) hudLabel.innerText = "CAM: RAW (/camera/image_raw)";
                feedImg.src = "/video_feed?type=raw";
            } else {
                btnProc.className = "px-2.5 py-1 rounded-md font-medium text-white bg-indigo-600 transition";
                btnRaw.className = "px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-zinc-200 transition";
                title.innerText = "Live Camera Feed (AI Processed)";
                if (hudLabel) hudLabel.innerText = "CAM: AI (/camera/image_processed)";
                feedImg.src = "/video_feed?type=processed";
            }
        }

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

        // ==========================================
        // MANUAL TELEOP CONTROLLER LOGIC (WASD & DPAD)
        // ==========================================
        let linearSpeed = 1.0;
        let angularSpeed = 2.0;
        let activeKeys = new Set();
        let mouseDriveDirection = null;
        let driveTimer = null;

        function updateSpeedFromSlider() {
            linearSpeed = parseFloat(document.getElementById("slider-linear-speed").value);
            angularSpeed = parseFloat(document.getElementById("slider-angular-speed").value);
            document.getElementById("label-linear-speed").innerText = linearSpeed.toFixed(1) + " m/s";
            document.getElementById("label-angular-speed").innerText = angularSpeed.toFixed(1) + " rad/s";
        }

        function adjustSpeed(linDelta, angDelta) {
            const linInput = document.getElementById("slider-linear-speed");
            const angInput = document.getElementById("slider-angular-speed");
            if (linDelta !== 0) {
                linInput.value = Math.max(0.2, Math.min(3.0, parseFloat(linInput.value) + linDelta)).toFixed(1);
            }
            if (angDelta !== 0) {
                angInput.value = Math.max(0.5, Math.min(5.0, parseFloat(angInput.value) + angDelta)).toFixed(1);
            }
            updateSpeedFromSlider();
        }

        function sendTwist(linear, angular) {
            fetch("/api/teleop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ linear: linear, angular: angular })
            }).catch(err => console.error("Teleop API error:", err));
        }

        function calculateVelocity() {
            let lin = 0.0;
            let ang = 0.0;

            // 1. Mouse/touch hold direction
            if (mouseDriveDirection) {
                if (mouseDriveDirection === 'forward') lin += linearSpeed;
                if (mouseDriveDirection === 'backward') lin -= linearSpeed;
                if (mouseDriveDirection === 'left') ang += angularSpeed;
                if (mouseDriveDirection === 'right') ang -= angularSpeed;
                return { linear: lin, angular: ang };
            }

            // 2. Keyboard held keys
            if (activeKeys.has('w') || activeKeys.has('arrowup')) lin += linearSpeed;
            if (activeKeys.has('s') || activeKeys.has('arrowdown')) lin -= linearSpeed;
            if (activeKeys.has('a') || activeKeys.has('arrowleft')) ang += angularSpeed;
            if (activeKeys.has('d') || activeKeys.has('arrowright')) ang -= angularSpeed;

            return { linear: lin, angular: ang };
        }

        function setBtnActive(id, active) {
            const btn = document.getElementById(id);
            if (!btn) return;
            if (active) {
                btn.classList.add('bg-indigo-600', 'text-white', 'scale-95', 'shadow-indigo-500/50');
                btn.classList.remove('bg-zinc-800', 'text-zinc-200');
            } else {
                btn.classList.remove('bg-indigo-600', 'text-white', 'scale-95', 'shadow-indigo-500/50');
                btn.classList.add('bg-zinc-800', 'text-zinc-200');
            }
        }

        function updateDpadVisuals() {
            const isFwd = activeKeys.has('w') || activeKeys.has('arrowup') || mouseDriveDirection === 'forward';
            const isBwd = activeKeys.has('s') || activeKeys.has('arrowdown') || mouseDriveDirection === 'backward';
            const isLeft = activeKeys.has('a') || activeKeys.has('arrowleft') || mouseDriveDirection === 'left';
            const isRight = activeKeys.has('d') || activeKeys.has('arrowright') || mouseDriveDirection === 'right';

            setBtnActive('btn-dpad-w', isFwd);
            setBtnActive('btn-dpad-s', isBwd);
            setBtnActive('btn-dpad-a', isLeft);
            setBtnActive('btn-dpad-d', isRight);
        }

        function driveLoop() {
            const vel = calculateVelocity();
            if (vel.linear !== 0.0 || vel.angular !== 0.0) {
                sendTwist(vel.linear, vel.angular);
            } else {
                sendTwist(0.0, 0.0);
                if (driveTimer) {
                    clearInterval(driveTimer);
                    driveTimer = null;
                }
            }
            updateDpadVisuals();
        }

        function startDriving() {
            if (!driveTimer) {
                driveLoop();
                driveTimer = setInterval(driveLoop, 100);
            }
        }

        function stopDriving() {
            activeKeys.clear();
            mouseDriveDirection = null;
            if (driveTimer) {
                clearInterval(driveTimer);
                driveTimer = null;
            }
            sendTwist(0.0, 0.0);
            updateDpadVisuals();
        }

        // Attach mouse & touch holding listeners to D-Pad buttons
        function attachHoldingListener(btnId, direction) {
            const el = document.getElementById(btnId);
            if (!el) return;

            const startHold = (e) => {
                e.preventDefault();
                mouseDriveDirection = direction;
                startDriving();
            };

            const endHold = (e) => {
                e.preventDefault();
                if (mouseDriveDirection === direction) {
                    mouseDriveDirection = null;
                    if (activeKeys.size === 0) {
                        stopDriving();
                    } else {
                        driveLoop();
                    }
                }
            };

            el.addEventListener('mousedown', startHold);
            el.addEventListener('mouseup', endHold);
            el.addEventListener('mouseleave', endHold);
            el.addEventListener('touchstart', startHold, { passive: false });
            el.addEventListener('touchend', endHold, { passive: false });
            el.addEventListener('touchcancel', endHold, { passive: false });
        }

        attachHoldingListener('btn-dpad-w', 'forward');
        attachHoldingListener('btn-dpad-s', 'backward');
        attachHoldingListener('btn-dpad-a', 'left');
        attachHoldingListener('btn-dpad-d', 'right');

        const stopBtn = document.getElementById('btn-dpad-stop');
        if (stopBtn) {
            stopBtn.addEventListener('click', (e) => {
                e.preventDefault();
                stopDriving();
            });
        }

        // Window keyboard listeners for continuous holding
        window.addEventListener('keydown', (e) => {
            if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

            const key = e.key.toLowerCase();
            if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(key)) {
                e.preventDefault();
                if (key === ' ') {
                    stopDriving();
                    return;
                }
                if (!activeKeys.has(key)) {
                    activeKeys.add(key);
                    startDriving();
                }
            } else if (key === '+' || key === '=') {
                adjustSpeed(0.2, 0.5);
            } else if (key === '-') {
                adjustSpeed(-0.2, -0.5);
            }
        });

        window.addEventListener('keyup', (e) => {
            if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

            const key = e.key.toLowerCase();
            if (activeKeys.has(key)) {
                activeKeys.delete(key);
                if (activeKeys.size === 0 && !mouseDriveDirection) {
                    stopDriving();
                } else {
                    driveLoop();
                }
            }
        });

        window.addEventListener('blur', () => {
            stopDriving();
        });

        // ==========================================
        // SEMANTIC LOCATIONS & PATROL MANAGER LOGIC
        // ==========================================
        let livePose = { x: 0.0, y: 0.0, theta: 0.0 };
        let semanticLocations = {};
        let patrolWaypoints = [];

        function switchManagerTab(tab) {
            const btnLoc = document.getElementById("tab-btn-locations");
            const btnPat = document.getElementById("tab-btn-patrol");
            const paneLoc = document.getElementById("tab-pane-locations");
            const panePat = document.getElementById("tab-pane-patrol");

            if (tab === 'locations') {
                btnLoc.className = "px-3 py-1.5 rounded-md font-semibold text-white bg-indigo-600 transition flex items-center space-x-1.5";
                btnPat.className = "px-3 py-1.5 rounded-md font-semibold text-zinc-400 hover:text-zinc-200 transition flex items-center space-x-1.5";
                paneLoc.classList.remove("hidden");
                panePat.classList.add("hidden");
            } else {
                btnPat.className = "px-3 py-1.5 rounded-md font-semibold text-white bg-indigo-600 transition flex items-center space-x-1.5";
                btnLoc.className = "px-3 py-1.5 rounded-md font-semibold text-zinc-400 hover:text-zinc-200 transition flex items-center space-x-1.5";
                panePat.classList.remove("hidden");
                paneLoc.classList.add("hidden");
            }
        }

        function fetchLocationsDatabase() {
            fetch("/api/locations")
                .then(res => res.json())
                .then(data => {
                    semanticLocations = data.locations || {};
                    patrolWaypoints = data.patrol_waypoints || [];
                    renderLocationsTable();
                    renderPatrolTable();
                    updateDeliveryDropdown();
                })
                .catch(err => console.error("Failed to load locations:", err));
        }

        function updateDeliveryDropdown() {
            const select = document.getElementById("select-dest");
            if (!select) return;
            const currentVal = select.value;
            const roomNames = Object.keys(semanticLocations).filter(k => !['start', 'charging_station'].includes(k));
            if (roomNames.length > 0) {
                select.innerHTML = roomNames.map(r => `<option value="${r}">${r.charAt(0).toUpperCase() + r.slice(1)}</option>`).join("");
                if (roomNames.includes(currentVal)) {
                    select.value = currentVal;
                }
            }
        }

        function renderLocationsTable() {
            const tbody = document.getElementById("locations-table-body");
            if (!tbody) return;
            const keys = Object.keys(semanticLocations);
            if (keys.length === 0) {
                tbody.innerHTML = `<tr><td colspan="5" class="p-4 text-center text-zinc-500 italic">No semantic locations defined yet. Click "Save Current Pose as Room" or "Add Custom Room".</td></tr>`;
                return;
            }
            tbody.innerHTML = keys.map(name => {
                const loc = semanticLocations[name];
                return `
                    <tr class="hover:bg-zinc-900/50 transition">
                        <td class="p-3 font-semibold text-zinc-200">${name}</td>
                        <td class="p-3 text-indigo-300">${parseFloat(loc.x).toFixed(2)}</td>
                        <td class="p-3 text-indigo-300">${parseFloat(loc.y).toFixed(2)}</td>
                        <td class="p-3 text-zinc-400">${parseFloat(loc.yaw || 0.0).toFixed(2)}</td>
                        <td class="p-3 text-right space-x-1.5">
                            <button onclick="sendCommand('go to ${name}')" title="Navigate Robot to ${name}" class="px-2.5 py-1 rounded bg-indigo-600/20 hover:bg-indigo-600 text-indigo-400 hover:text-white border border-indigo-500/20 text-xs font-sans transition">
                                <i class="fa-solid fa-location-arrow mr-1"></i>Go
                            </button>
                            <button onclick="updateRoomWithCurrentPose('${name}')" title="Overwrite ${name} with Current Pose" class="px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 hover:text-white border border-zinc-700 text-xs font-sans transition">
                                <i class="fa-solid fa-crosshairs mr-1"></i>Capture Pose
                            </button>
                            <button onclick="deleteRoom('${name}')" title="Delete ${name}" class="px-2 py-1 rounded bg-red-950/40 hover:bg-red-600 text-red-400 hover:text-white border border-red-800/40 text-xs font-sans transition">
                                <i class="fa-solid fa-trash"></i>
                            </button>
                        </td>
                    </tr>
                `;
            }).join("");
        }

        function renderPatrolTable() {
            const tbody = document.getElementById("patrol-table-body");
            if (!tbody) return;
            if (patrolWaypoints.length === 0) {
                tbody.innerHTML = `<tr><td colspan="5" class="p-4 text-center text-zinc-500 italic">No patrol waypoints. Click "Add Current Pose as Waypoint" to build a path.</td></tr>`;
                return;
            }
            tbody.innerHTML = patrolWaypoints.map((wp, idx) => `
                <tr class="hover:bg-zinc-900/50 transition">
                    <td class="p-3 text-zinc-500 font-bold">#${idx + 1}</td>
                    <td class="p-3 text-emerald-400">${parseFloat(wp.x).toFixed(2)}</td>
                    <td class="p-3 text-emerald-400">${parseFloat(wp.y).toFixed(2)}</td>
                    <td class="p-3 text-zinc-400">${parseFloat(wp.yaw || 0.0).toFixed(2)}</td>
                    <td class="p-3 text-right space-x-1.5">
                        <button onclick="moveWaypoint(${idx}, -1)" ${idx === 0 ? 'disabled class="opacity-30 px-2 py-1"' : 'class="px-2 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 text-xs transition"'} title="Move Up">
                            <i class="fa-solid fa-arrow-up"></i>
                        </button>
                        <button onclick="moveWaypoint(${idx}, 1)" ${idx === patrolWaypoints.length - 1 ? 'disabled class="opacity-30 px-2 py-1"' : 'class="px-2 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 text-xs transition"'} title="Move Down">
                            <i class="fa-solid fa-arrow-down"></i>
                        </button>
                        <button onclick="deleteWaypoint(${idx})" title="Delete Waypoint" class="px-2 py-1 bg-red-950/40 hover:bg-red-600 text-red-400 hover:text-white rounded border border-red-800/40 text-xs transition">
                            <i class="fa-solid fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `).join("");
        }

        function addCurrentPoseAsRoom() {
            const name = prompt("Enter Room / Location Name (e.g., 'balcony', 'study_room'):");
            if (!name || !name.trim()) return;
            const key = name.trim().toLowerCase();
            semanticLocations[key] = {
                x: parseFloat(livePose.x.toFixed(2)),
                y: parseFloat(livePose.y.toFixed(2)),
                yaw: parseFloat((livePose.theta || 0.0).toFixed(2))
            };
            renderLocationsTable();
            updateDeliveryDropdown();
            saveLocationsToServer();
        }

        function updateRoomWithCurrentPose(name) {
            if (confirm(`Overwrite coordinates of '${name}' with current robot pose (x: ${livePose.x.toFixed(2)}, y: ${livePose.y.toFixed(2)})?`)) {
                semanticLocations[name] = {
                    x: parseFloat(livePose.x.toFixed(2)),
                    y: parseFloat(livePose.y.toFixed(2)),
                    yaw: parseFloat((livePose.theta || 0.0).toFixed(2))
                };
                renderLocationsTable();
                saveLocationsToServer();
            }
        }

        function addNewRoomRow() {
            const name = prompt("Enter Room Name:");
            if (!name || !name.trim()) return;
            const x = prompt("Enter X coordinate (meters):", "0.0");
            const y = prompt("Enter Y coordinate (meters):", "0.0");
            const yaw = prompt("Enter Yaw orientation (radians):", "0.0");
            const key = name.trim().toLowerCase();
            semanticLocations[key] = {
                x: parseFloat(x) || 0.0,
                y: parseFloat(y) || 0.0,
                yaw: parseFloat(yaw) || 0.0
            };
            renderLocationsTable();
            updateDeliveryDropdown();
            saveLocationsToServer();
        }

        function deleteRoom(name) {
            if (confirm(`Are you sure you want to delete '${name}'?`)) {
                delete semanticLocations[name];
                renderLocationsTable();
                updateDeliveryDropdown();
                saveLocationsToServer();
            }
        }

        function addCurrentPoseAsWaypoint() {
            patrolWaypoints.push({
                x: parseFloat(livePose.x.toFixed(2)),
                y: parseFloat(livePose.y.toFixed(2)),
                yaw: parseFloat((livePose.theta || 0.0).toFixed(2))
            });
            renderPatrolTable();
            savePatrolToServer();
        }

        function addNewWaypointRow() {
            const x = prompt("Enter X coordinate (meters):", "0.0");
            const y = prompt("Enter Y coordinate (meters):", "0.0");
            const yaw = prompt("Enter Yaw orientation (radians):", "0.0");
            patrolWaypoints.push({
                x: parseFloat(x) || 0.0,
                y: parseFloat(y) || 0.0,
                yaw: parseFloat(yaw) || 0.0
            });
            renderPatrolTable();
            savePatrolToServer();
        }

        function moveWaypoint(idx, delta) {
            const targetIdx = idx + delta;
            if (targetIdx < 0 || targetIdx >= patrolWaypoints.length) return;
            const item = patrolWaypoints.splice(idx, 1)[0];
            patrolWaypoints.splice(targetIdx, 0, item);
            renderPatrolTable();
            savePatrolToServer();
        }

        function deleteWaypoint(idx) {
            patrolWaypoints.splice(idx, 1);
            renderPatrolTable();
            savePatrolToServer();
        }

        function saveLocationsToServer() {
            fetch("/api/locations", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    locations: semanticLocations,
                    patrol_waypoints: patrolWaypoints
                })
            })
            .then(res => res.json())
            .then(data => {
                console.log("Locations database updated successfully.");
            })
            .catch(err => console.error("Error saving locations:", err));
        }

        function savePatrolToServer() {
            saveLocationsToServer();
        }

        // Initialize locations database on page load
        fetchLocationsDatabase();
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
        self.raw_frame = None
        self.processed_frame = None
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
        
        # Teleop State & Watchdog
        self.last_teleop_time = 0.0
        self.is_teleop_active = False
        
        # Subscriptions
        self.battery_sub = self.create_subscription(
            Float32, '/battery/percentage', self.battery_callback, 10)
        self.emergency_sub = self.create_subscription(
            Bool, '/emergency/alarm', self.emergency_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.raw_image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.raw_image_callback, 10)
        self.processed_image_sub = self.create_subscription(
            Image, '/camera/image_processed', self.processed_image_callback, 10)
        self.cmd_sub = self.create_subscription(
            String, '/voice/command', self.command_callback, 10)
            
        # Publisher to trigger actions
        self.cmd_pub = self.create_publisher(String, '/voice/command', 10)
        
        # Publisher for raw teleop velocity commands
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel_raw', 10)
        self.teleop_watchdog = self.create_timer(0.1, self.teleop_watchdog_callback)
        
        self.get_logger().info("Dashboard ROS 2 Node Initialized with Manual Teleop & Feeds.")

    def teleop_watchdog_callback(self):
        # Stop robot if teleop command has stopped streaming for > 0.35 seconds
        if self.is_teleop_active and (time.time() - self.last_teleop_time > 0.35):
            stop_twist = Twist()
            self.twist_pub.publish(stop_twist)
            self.is_teleop_active = False
            self.current_state = "Idle"

    def publish_twist(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.twist_pub.publish(msg)
        self.last_teleop_time = time.time()
        
        if abs(linear) > 0.001 or abs(angular) > 0.001:
            self.is_teleop_active = True
            if linear > 0.0:
                self.current_state = "Manual Driving (Forward)"
            elif linear < 0.0:
                self.current_state = "Manual Driving (Backward)"
            elif angular > 0.0:
                self.current_state = "Manual Turning (Left)"
            elif angular < 0.0:
                self.current_state = "Manual Turning (Right)"
        else:
            self.is_teleop_active = False
            self.current_state = "Manual Stopped"

    def battery_callback(self, msg: Float32):
        self.battery_level = msg.data
        if self.start_battery == 100.0 and msg.data < 100.0:
            self.start_battery = msg.data
        self.battery_used = max(0.0, self.start_battery - msg.data)

    def emergency_callback(self, msg: Bool):
        self.emergency_active = msg.data

    def odom_callback(self, msg: Odometry):
        # Update current pose (x, y, theta)
        self.current_pose['x'] = msg.pose.pose.position.x
        self.current_pose['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_pose['theta'] = math.atan2(siny_cosp, cosy_cosp)
        
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

    def raw_image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, jpeg = cv2.imencode('.jpg', cv_image)
            self.raw_frame = jpeg.tobytes()
            self.latest_frame = self.raw_frame
        except Exception as e:
            pass

    def processed_image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, jpeg = cv2.imencode('.jpg', cv_image)
            self.processed_frame = jpeg.tobytes()
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

def gen_frames(feed_type='raw'):
    while True:
        frame = None
        if node:
            if feed_type == 'processed':
                frame = node.processed_frame if node.processed_frame is not None else STANDBY_PROC_FRAME
            else:
                # Default to raw camera frame
                frame = node.raw_frame if node.raw_frame is not None else STANDBY_RAW_FRAME
        else:
            frame = STANDBY_RAW_FRAME

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    feed_type = request.args.get('type', 'raw')
    return Response(gen_frames(feed_type), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/api/teleop', methods=['POST'])
def manual_teleop():
    data = request.json or {}
    linear = float(data.get('linear', 0.0))
    angular = float(data.get('angular', 0.0))
    if node:
        node.publish_twist(linear, angular)
        return jsonify({'status': 'success', 'linear': linear, 'angular': angular})
    return jsonify({'status': 'error', 'message': 'ROS Node not ready'}), 400

@app.route('/api/locations', methods=['GET'])
def get_locations():
    config = load_semantic_config()
    return jsonify({
        'locations': config.get('locations', {}),
        'patrol_waypoints': config.get('patrol_waypoints', [])
    })

@app.route('/api/locations', methods=['POST'])
def update_locations():
    data = request.json or {}
    locations = data.get('locations', {})
    patrol_waypoints = data.get('patrol_waypoints', [])
    
    config = {
        'locations': locations,
        'patrol_waypoints': patrol_waypoints
    }
    
    saved = save_semantic_config(config)
    if saved and node:
        # Send reload command to voice interpreter
        node.publish_command("reload locations")
        return jsonify({'status': 'success', 'message': 'Locations and patrol path saved & reloaded'})
    elif saved:
        return jsonify({'status': 'success', 'message': 'Locations and patrol path saved'})
    return jsonify({'status': 'error', 'message': 'Failed to save configuration'}), 500

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
