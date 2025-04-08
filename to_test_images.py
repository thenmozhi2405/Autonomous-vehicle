# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 10:08:50 2025

@author: sgpdh
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO(r"D:\SEM 6\Open lab\project\roboflow_dataset1\traffic_light_model_v2.pt")

print("✅ Model loaded successfully!")

# Path to the test image (UPDATE THIS PATH)
test_image_path = r"D:\SEM 6\Open lab\project\roboflow_dataset1\test\images\traffic-light-119-_jpg.rf.7a174ade98f4e601c42fb7c91a0542e2.jpg"

# Run inference on the test image
results = model(test_image_path)

# Display results
for result in results:
    print("\n🔹 Detected Objects:")
    for box in result.boxes:
        class_id = int(box.cls[0])  # Get class ID
        conf = round(float(box.conf[0]) * 100, 2)  # Get confidence score (%)
        print(f"  Class: {model.names[class_id]}, Confidence: {conf}%")

    # Show image with bounding boxes
    result.show()  # Displays the image with detections

# Save the output image with detections
results[0].save(filename="output.jpg")

print("✅ Inference completed! Check 'output.jpg' for results.")
