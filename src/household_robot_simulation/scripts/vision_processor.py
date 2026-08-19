#!/usr/bin/env python3
"""
ROS 2 Node for Module 8: Computer Vision.
Subscribes to raw camera images, processes them using OpenCV, and publishes
an annotated image stream showing detected objects (by color), faces, and QR codes.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import json
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


class VisionProcessor(Node):
    def __init__(self):
        super().__init__('vision_processor')
        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Load Face Detection Cascade
        cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            self.get_logger().error(f"Failed to load Haar Cascade from {cascade_path}")
        else:
            self.get_logger().info("Successfully loaded Haar Cascade Face Classifier")

        # Initialize QR Code Detector
        self.qr_detector = cv2.QRCodeDetector()

        # Load YOLO model (OpenVINO optimized for Intel Iris GPU)
        try:
            pkg_share = get_package_share_directory('household_robot_simulation')
            model_path = os.path.join(pkg_share, 'models', 'yolov8n_openvino_model')
            self.get_logger().info(f"Loading OpenVINO YOLO model from {model_path}...")
            self.yolo_model = YOLO(model_path, task='detect')
            self.get_logger().info("YOLO model loaded successfully on OpenVINO backend.")
        except Exception as e:
            self.get_logger().error(f"Failed to load OpenVINO YOLO model: {e}. Falling back to PyTorch CPU...")
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.get_logger().info("YOLO CPU model loaded successfully.")
            except Exception as ex:
                self.get_logger().error(f"Failed to load fallback YOLO model: {ex}")
                self.yolo_model = None

        # Define HSV Color Ranges for Object Detection
        # Red has two ranges in HSV due to wrap-around
        # Saturation and Value thresholds are lowered to 50 to properly handle shadows in Gazebo Sim.
        self.color_ranges = {
            'Red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'Green': [
                (np.array([35, 50, 50]), np.array([85, 255, 255]))
            ],
            'Blue': [
                (np.array([100, 50, 50]), np.array([140, 255, 255]))
            ],
            'Yellow': [
                (np.array([20, 50, 50]), np.array([30, 255, 255]))
            ]
        }

        # Frame skip for YOLO to conserve CPU (run YOLO on every 2nd frame)
        self.frame_counter = 0
        self.yolo_frame_skip = 2

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.image_pub = self.create_publisher(
            Image,
            '/camera/image_processed',
            10
        )

        self.yolo_detections_pub = self.create_publisher(
            String,
            '/yolo/detections',
            10
        )

        # Cache for YOLO detections to prevent flickering on skipped frames
        self.last_yolo_detections = []

        self.get_logger().info("Vision Processor Node started.")

    def draw_text_with_outline(self, image, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
        # Draw black outline
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Draw inner text
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def image_callback(self, msg: Image):
        try:
            # Convert ROS 2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Create a clean copy of the frame to perform all detections on,
        # ensuring overlays from one detector don't corrupt subsequent ones.
        detection_image = cv_image.copy()

        # 1. Process Object Detection (Color-based contours)
        self.detect_colored_objects(detection_image, cv_image)

        # 2. Process Face Detection (Haar Cascades)
        self.detect_faces(detection_image, cv_image)

        # 3. Process QR Code Detection
        self.detect_qr_codes(detection_image, cv_image)

        # 4. Process YOLO Object Detection
        self.frame_counter += 1
        if self.yolo_model is not None:
            if self.frame_counter % self.yolo_frame_skip == 0:
                self.detect_yolo_objects(detection_image)
            self.draw_cached_yolo_objects(cv_image)

        # Publish the processed/annotated image
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            processed_msg.header = msg.header  # Preserve timestamp and frame_id
            self.image_pub.publish(processed_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to publish processed image: {e}")

    def detect_colored_objects(self, input_image, output_image):
        # Convert image to HSV color space
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        for color_name, ranges in self.color_ranges.items():
            # Combine masks if multiple ranges exist (like Red)
            mask = None
            for lower, upper in ranges:
                r_mask = cv2.inRange(hsv, lower, upper)
                if mask is None:
                    mask = r_mask
                else:
                    mask = cv2.bitwise_or(mask, r_mask)

            # Apply morphological opening to filter noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 400:  # Minimum area threshold (pixels)
                    # Compute bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Compute centroid (moments)
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = x + w//2, y + h//2

                    # Draw outline and bounding box on output_image
                    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 128, 0), 2)
                    cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)

                    # Label overlay with outline
                    label = f"{color_name} Obj ({cx},{cy})"
                    self.draw_text_with_outline(output_image, label, (x, y - 5), font_scale=0.45, color=(255, 255, 255))

    def detect_faces(self, input_image, output_image):
        if self.face_cascade.empty():
            return

        # Convert to grayscale for detection
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around detected faces (Magenta) on output_image
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            self.draw_text_with_outline(output_image, "Face Detected", (x, y - 5), font_scale=0.5, color=(255, 0, 255))
            self.get_logger().info("Human Face Detected!", throttle_duration_sec=2.0)

    def detect_qr_codes(self, input_image, output_image):
        # Detect and decode QR Code on input_image
        decoded_info, points, _ = self.qr_detector.detectAndDecode(input_image)

        if points is not None and len(points) > 0:
            # Convert points to integers
            pts = points[0].astype(int)
            
            # Draw bounding box around QR code (Green) on output_image
            for i in range(len(pts)):
                cv2.line(output_image, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)

            if decoded_info:
                # Log detection and draw decoded text label
                self.get_logger().info(f"QR Code Detected: '{decoded_info}'", throttle_duration_sec=2.0)
                x, y = pts[0][0], pts[0][1]
                self.draw_text_with_outline(output_image, f"QR: {decoded_info}", (x, y - 10), font_scale=0.5, color=(0, 255, 0))

    def detect_yolo_objects(self, input_image):
        if self.yolo_model is None:
            return
        try:
            results = self.yolo_model.predict(input_image, conf=0.15, verbose=False, device='intel:gpu')
        except Exception:
            results = self.yolo_model.predict(input_image, conf=0.15, verbose=False, device='cpu')
        result = results[0]

        detections_list = []
        self.last_yolo_detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = self.yolo_model.names[cls_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in bbox]

            detections_list.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
            self.last_yolo_detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })

        # Publish structured JSON detections
        detections_msg = String()
        detections_msg.data = json.dumps(detections_list)
        self.yolo_detections_pub.publish(detections_msg)

        if detections_list:
            summary = ", ".join([f"{d['class']} ({d['confidence']:.2f})" for d in detections_list])
            self.get_logger().info(f"YOLO Detected: {summary}", throttle_duration_sec=2.0)

    def draw_cached_yolo_objects(self, output_image):
        for d in self.last_yolo_detections:
            class_name = d['class']
            conf = d['confidence']
            x1, y1, x2, y2 = d['bbox']

            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box (Orange: B=0, G=165, R=255)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1)

            # Label overlay
            label = f"{class_name} ({conf:.2f})"
            self.draw_text_with_outline(output_image, label, (x1, y1 - 5), font_scale=0.45, color=(255, 255, 255))


def main(args=None):
    rclpy.init(args=args)
    node = VisionProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
