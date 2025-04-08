import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

# ------------------- Lane Detection Functions -------------------
def convert_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def HSL_color_selection(image):
    converted_image = convert_hsl(image)
    lower_white, upper_white = np.uint8([0, 200, 0]), np.uint8([255, 255, 255])
    lower_yellow, upper_yellow = np.uint8([10, 0, 100]), np.uint8([40, 255, 255])

    white_mask = cv2.inRange(converted_image, lower_white, upper_white)
    yellow_mask = cv2.inRange(converted_image, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    return cv2.bitwise_and(image, image, mask=mask)

def region_selection(image):
    mask = np.zeros_like(image)
    rows, cols = image.shape[:2]

    vertices = np.array([[
        [cols * 0.1, rows * 0.95], [cols * 0.4, rows * 0.6], 
        [cols * 0.6, rows * 0.6], [cols * 0.9, rows * 0.95]
    ]], dtype=np.int32)

    mask_color = (255,) * image.shape[2] if len(image.shape) > 2 else 255
    cv2.fillPoly(mask, vertices, mask_color)
    return cv2.bitwise_and(image, mask)

def hough_transform(image):
    return cv2.HoughLinesP(image, 1, np.pi/180, 20, minLineLength=20, maxLineGap=300)

def average_slope_intercept(lines):
    left_lines, right_lines = [], []
    left_weights, right_weights = [], []

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, int(y1)), (x2, int(y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    return pixel_points(y1, y2, left_lane), pixel_points(y1, y2, right_lane)

def draw_lane_lines(image, lines, color=(255, 0, 0), thickness=12):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    roi = region_selection(edges)
    hough_lines = hough_transform(roi)
    lane_lines_detected = lane_lines(image, hough_lines)
    final_image = draw_lane_lines(image, lane_lines_detected)

    return final_image

# ------------------- Accuracy Metrics Functions -------------------
def compute_iou(detected, ground_truth):
    """Intersection over Union (IoU) Calculation"""
    intersection = np.logical_and(detected, ground_truth)
    union = np.logical_or(detected, ground_truth)
    return np.sum(intersection) / np.sum(union)

def pixel_accuracy(detected, ground_truth):
    """Pixel-wise Accuracy Calculation"""
    correct_pixels = np.sum(detected == ground_truth)
    total_pixels = detected.size
    return correct_pixels / total_pixels

# ------------------- Load Images -------------------
image_path = r"C:\Users\Admin\Desktop\download.jpeg"  # Change to your image path
ground_truth_path = r"C:\Users\Admin\Desktop\ground_truth.jpeg"  # Change to your ground truth path

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# Process the image (Lane Detection)
detected_lanes = process_image(image)

# Convert detected lanes to binary mask
detected_lanes_gray = cv2.cvtColor(detected_lanes, cv2.COLOR_RGB2GRAY)
detected_lanes_bin = cv2.threshold(detected_lanes_gray, 128, 255, cv2.THRESH_BINARY)[1]

# Check if ground truth exists
if not os.path.exists(ground_truth_path):
    print(f"Ground truth image not found! Saving detected lanes as ground truth...")
    cv2.imwrite(ground_truth_path, detected_lanes_bin)

# Load Ground Truth Image (Binary Mask)
ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
if ground_truth is None:
    raise FileNotFoundError(f"Could not load ground truth image from: {ground_truth_path}")

ground_truth = cv2.threshold(ground_truth, 128, 255, cv2.THRESH_BINARY)[1]

# ------------------- Compute Accuracy -------------------
iou_score = compute_iou(detected_lanes_bin, ground_truth)
ssim_score = ssim(ground_truth, detected_lanes_bin)
accuracy = pixel_accuracy(detected_lanes_bin, ground_truth)

# ------------------- Print Results -------------------
print(f"IoU Score: {iou_score:.2f}")
print(f"SSIM Score: {ssim_score:.2f}")
print(f"Pixel-wise Accuracy: {accuracy:.2f}")

# ------------------- Display Images -------------------
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(detected_lanes)
plt.title("Detected Lanes")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ground_truth, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.show()
