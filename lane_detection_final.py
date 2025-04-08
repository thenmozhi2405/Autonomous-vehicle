import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# Pin Configuration for L298N Motor Driver (Front & Back Motors)
IN1_FRONT, IN2_FRONT, IN3_FRONT, IN4_FRONT = 17, 18, 27, 22
ENA_FRONT, ENB_FRONT = 23, 24  
IN1_BACK, IN2_BACK, IN3_BACK, IN4_BACK = 5, 6, 13, 19
ENA_BACK, ENB_BACK = 20, 21  

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

motor_pins = [IN1_FRONT, IN2_FRONT, IN3_FRONT, IN4_FRONT, ENA_FRONT, ENB_FRONT,
              IN1_BACK, IN2_BACK, IN3_BACK, IN4_BACK, ENA_BACK, ENB_BACK]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# PWM Setup for speed control
pwm_front_a = GPIO.PWM(ENA_FRONT, 1000)
pwm_front_b = GPIO.PWM(ENB_FRONT, 1000)
pwm_back_a = GPIO.PWM(ENA_BACK, 1000)
pwm_back_b = GPIO.PWM(ENB_BACK, 1000)

pwm_front_a.start(0)
pwm_front_b.start(0)
pwm_back_a.start(0)
pwm_back_b.start(0)

running = False  # 🚗 Car starts in STOPPED mode

# === MOTOR CONTROL FUNCTIONS === #
def move_forward(speed=19):
    GPIO.output(IN1_FRONT, GPIO.HIGH)
    GPIO.output(IN2_FRONT, GPIO.LOW)
    GPIO.output(IN3_FRONT, GPIO.HIGH)
    GPIO.output(IN4_FRONT, GPIO.LOW)
    GPIO.output(IN1_BACK, GPIO.HIGH)
    GPIO.output(IN2_BACK, GPIO.LOW)
    GPIO.output(IN3_BACK, GPIO.HIGH)
    GPIO.output(IN4_BACK, GPIO.LOW)
    pwm_front_a.ChangeDutyCycle(speed)
    pwm_front_b.ChangeDutyCycle(speed)
    pwm_back_a.ChangeDutyCycle(speed)
    pwm_back_b.ChangeDutyCycle(speed)
    print("🚗 Moving forward")

def turn_left(speed=15):
    GPIO.output(IN1_FRONT, GPIO.LOW)
    GPIO.output(IN2_FRONT, GPIO.HIGH)
    GPIO.output(IN3_FRONT, GPIO.HIGH)
    GPIO.output(IN4_FRONT, GPIO.LOW)
    GPIO.output(IN1_BACK, GPIO.LOW)
    GPIO.output(IN2_BACK, GPIO.HIGH)
    GPIO.output(IN3_BACK, GPIO.HIGH)
    GPIO.output(IN4_BACK, GPIO.LOW)
    pwm_front_a.ChangeDutyCycle(speed)
    pwm_front_b.ChangeDutyCycle(speed)
    pwm_back_a.ChangeDutyCycle(speed)
    pwm_back_b.ChangeDutyCycle(speed)
    print("⬅ Turning Left")

def turn_right(speed=15):
    GPIO.output(IN1_FRONT, GPIO.HIGH)
    GPIO.output(IN2_FRONT, GPIO.LOW)
    GPIO.output(IN3_FRONT, GPIO.LOW)
    GPIO.output(IN4_FRONT, GPIO.HIGH)
    GPIO.output(IN1_BACK, GPIO.HIGH)
    GPIO.output(IN2_BACK, GPIO.LOW)
    GPIO.output(IN3_BACK, GPIO.LOW)
    GPIO.output(IN4_BACK, GPIO.HIGH)
    pwm_front_a.ChangeDutyCycle(speed)
    pwm_front_b.ChangeDutyCycle(speed)
    pwm_back_a.ChangeDutyCycle(speed)
    pwm_back_b.ChangeDutyCycle(speed)
    print("➡ Turning Right")

def stop():
    pwm_front_a.ChangeDutyCycle(0)
    pwm_front_b.ChangeDutyCycle(0)
    pwm_back_a.ChangeDutyCycle(0)
    pwm_back_b.ChangeDutyCycle(0)
    print("🛑 Stopping")

# === LANE DETECTION FUNCTIONS === #
def detect_lane(frame):
    """Detects lanes using color thresholding and edge detection."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresholded, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Define Region of Interest (ROI)
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    roi_corners = np.array([[(50, height), (width // 2 - 50, height // 2), 
                             (width // 2 + 50, height // 2), (width-50, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, 255)

    masked_edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
    return lines, masked_edges

def process_frame(frame):
    """Processes the frame, detects lanes, and controls the car based on lane position."""
    global running  # Use global flag

    lines, masked_edges = detect_lane(frame)
    left_count, right_count = 0, 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            if slope < -0.3:  # Left lane
                left_count += 1
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Draw left lane (blue)
            elif slope > 0.3:  # Right lane
                right_count += 1
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Draw right lane (green)

    if running:  # ✅ Move only when 's' is pressed
        if abs(left_count - right_count) > 2:  # Reduce wobbling
            if left_count > right_count:
                turn_left()
            elif right_count > left_count:
                turn_right()
            else:
                move_forward()
        else:
            move_forward()  # If almost equal, go straight
    
    return frame, masked_edges

# === MAIN FUNCTION === #
def main():
    global running  # Use global flag

    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame for lane detection
        processed_frame, masked_edges = process_frame(frame)

        # Show processed video feed
        cv2.imshow("Lane Detection", processed_frame)
        cv2.imshow("Masked Edges", masked_edges)

        # Press 's' to start moving, 'q' to stop, ESC to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            running = True  # 🚗 Start moving
            print("✅ Movement Enabled")
        elif key == ord('q'):
            running = False  # 🛑 Stop moving
            stop()
            print("🛑 Movement Stopped")
        elif key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

if _name_ == "_main_":
    main()