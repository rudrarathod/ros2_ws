#!/usr/bin/env python3
"""
ROS 2 Node for YOLOv8 Object Detection in Module 9.
Subscribes to raw camera images, processes them using the YOLOv8-nano model,
and publishes both an annotated image stream and structured detection data.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import json
import numpy as np
from ultralytics import YOLO


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('frame_skip', 6)  # Process every Nth frame to save CPU
        self.declare_parameter('model_name', 'yolov8n.pt')

        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.frame_skip = self.get_parameter('frame_skip').value
        model_name = self.get_parameter('model_name').value

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model: {model_name}...")
        try:
            self.model = YOLO(model_name)
            self.get_logger().info("YOLO model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise e

        # Frame counter for skipping frames
        self.frame_counter = 0

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
            '/camera/yolo_detections',
            10
        )

        self.detections_pub = self.create_publisher(
            String,
            '/yolo/detections',
            10
        )

        self.get_logger().info("YOLO Detector Node initialized.")

    def draw_text_with_outline(self, image, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
        # Draw black outline
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # Draw inner text
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    def image_callback(self, msg: Image):
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return

        try:
            # Convert ROS 2 Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Run YOLO inference
        # Use clean copy or run inference directly (YOLO doesn't modify input in-place by default)
        results = self.model.predict(cv_image, conf=self.conf_threshold, verbose=False)
        result = results[0]

        detections_list = []
        annotated_image = cv_image.copy()

        # Parse detection results
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            # Bounding box coordinates
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            detections_list.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })

            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box (Orange: B=0, G=165, R=255)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.circle(annotated_image, (cx, cy), 5, (0, 0, 255), -1)

            # Draw label
            label = f"{class_name} ({conf:.2f})"
            self.draw_text_with_outline(annotated_image, label, (x1, y1 - 5), font_scale=0.45, color=(255, 255, 255))

        # Publish structured detection data
        detections_msg = String()
        detections_msg.data = json.dumps(detections_list)
        self.detections_pub.publish(detections_msg)

        # Log detections (throttled to 2 seconds)
        if detections_list:
            summary = ", ".join([f"{d['class']} ({d['confidence']:.2f})" for d in detections_list])
            self.get_logger().info(f"YOLO Detected: {summary}", throttle_duration_sec=2.0)

        # Publish annotated image
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            processed_msg.header = msg.header  # Preserve timestamp and frame_id
            self.image_pub.publish(processed_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to publish YOLO annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
