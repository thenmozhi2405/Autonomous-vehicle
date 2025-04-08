import cv2
import RPi.GPIO as GPIO
import time
import numpy as np
import threading
import os

# Load Haar cascades
STOP_SIGN_CASCADE_PATH = "/home/thenmozhi/Downloads/stop.xml"
SPEED_SIGN_CASCADE_PATH = "/home/thenmozhi/Downloads/TrafficLight_HAAR_16Stages.xml"

# Check if Haar cascades exist
if not os.path.exists(STOP_SIGN_CASCADE_PATH):
    print(f"❌ ERROR: Stop sign Haar cascade file not found at {STOP_SIGN_CASCADE_PATH}")
    exit()

if not os.path.exists(SPEED_SIGN_CASCADE_PATH):
    print(f"❌ ERROR: Speed limit Haar cascade file not found at {SPEED_SIGN_CASCADE_PATH}")
    exit()

stop_sign_cascade = cv2.CascadeClassifier(STOP_SIGN_CASCADE_PATH)
speed_sign_cascade = cv2.CascadeClassifier(SPEED_SIGN_CASCADE_PATH)

# *Debug Haar Cascades*
if stop_sign_cascade.empty():
    print("❌ ERROR: Stop sign cascade not loaded correctly!")
if speed_sign_cascade.empty():
    print("❌ ERROR: Speed sign cascade not loaded correctly!")

# GPIO Configuration for L298N Motor Driver
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

# PWM Setup
pwm_front_left = GPIO.PWM(ENA_FRONT, 1000)
pwm_front_right = GPIO.PWM(ENB_FRONT, 1000)
pwm_back_left = GPIO.PWM(ENA_BACK, 1000)
pwm_back_right = GPIO.PWM(ENB_BACK, 1000)

pwm_front_left.start(20)
pwm_front_right.start(20)
pwm_back_left.start(20)
pwm_back_right.start(20)

# Speed Levels
NORMAL_SPEED =20
SLOW_SPEED = 10
LIMITED_SPEED = 15

# Movement Functions
def move_forward(speed=NORMAL_SPEED):
    pwm_front_left.ChangeDutyCycle(speed)
    pwm_front_right.ChangeDutyCycle(speed)
    pwm_back_left.ChangeDutyCycle(speed)
    pwm_back_right.ChangeDutyCycle(speed)
    GPIO.output(IN1_FRONT, GPIO.HIGH)
    GPIO.output(IN2_FRONT, GPIO.LOW)
    GPIO.output(IN3_FRONT, GPIO.HIGH)
    GPIO.output(IN4_FRONT, GPIO.LOW)
    GPIO.output(IN1_BACK, GPIO.HIGH)
    GPIO.output(IN2_BACK, GPIO.LOW)
    GPIO.output(IN3_BACK, GPIO.HIGH)
    GPIO.output(IN4_BACK, GPIO.LOW)
    print(f"🚗 Moving Forward at speed {speed}")

def stop_motors():
    pwm_front_left.ChangeDutyCycle(0)
    pwm_front_right.ChangeDutyCycle(0)
    pwm_back_left.ChangeDutyCycle(0)
    pwm_back_right.ChangeDutyCycle(0)
    GPIO.output(IN1_FRONT, GPIO.LOW)
    GPIO.output(IN2_FRONT, GPIO.LOW)
    GPIO.output(IN3_FRONT, GPIO.LOW)
    GPIO.output(IN4_FRONT, GPIO.LOW)
    GPIO.output(IN1_BACK, GPIO.LOW)
    GPIO.output(IN2_BACK, GPIO.LOW)
    GPIO.output(IN3_BACK, GPIO.LOW)
    GPIO.output(IN4_BACK, GPIO.LOW)
    print("🛑 Motors Stopped.")

# *Improved Traffic Light Detection*
def detect_traffic_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # *Red Color Range (Two Ranges)*
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 100, 100]), np.array([180, 255, 255])

    # *Yellow Color Range*
    lower_yellow, upper_yellow = np.array([20, 100, 100]), np.array([40, 255, 255])

    # *Green Color Range*
    lower_green, upper_green = np.array([40, 50, 50]), np.array([90, 255, 255])

    # *Create Masks & Reduce Noise*
    kernel = np.ones((5, 5), np.uint8)
    
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    # *Get Pixel Counts*
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > max(yellow_pixels, green_pixels):
        return "red"
    elif yellow_pixels > max(red_pixels, green_pixels):
        return "yellow"
    elif green_pixels > max(red_pixels, yellow_pixels):
        return "green"
    else:
        return None

# Start Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

# Detection Thread Function
def detection_loop():
    global stop_sign_detected, speed_sign_detected, detected_color

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Traffic Light Detection
        detected_color = detect_traffic_light(frame)

        # Stop Sign Detection
        stop_sign_detected = len(stop_sign_cascade.detectMultiScale(gray, 1.2, 5)) > 0

        # Speed Limit Sign Detection
        speed_sign_detected = len(speed_sign_cascade.detectMultiScale(gray, 1.2, 4)) > 0

        cv2.imshow("Traffic Light Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

# Start Detection Thread
stop_sign_detected, speed_sign_detected, detected_color = False, False, None
detection_thread = threading.Thread(target=detection_loop, daemon=True)
detection_thread.start()

try:
    while True:
        if stop_sign_detected:
            stop_motors()
            print("🛑 STOP SIGN DETECTED! Car Stopped.")
            time.sleep(3)
        elif detected_color == "red":
            stop_motors()
            print("🔴 RED LIGHT! Car Stopped.")
        elif detected_color == "yellow":
            move_forward(SLOW_SPEED)
            print("🟡 YELLOW LIGHT! Slowing down.")
        elif detected_color == "green":
            move_forward(NORMAL_SPEED)
            print("🟢 GREEN LIGHT! Moving Forward.")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n⏹ Stopped by user.")

finally:
    stop_motors()
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cleaned up GPIO and closed webcam.")