import RPi.GPIO as GPIO
import time

# 🚀 GPIO Mode Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# 📏 Corrected Ultrasonic Sensor Pins (New)
TRIG = 16  # ✅ Changed to GPIO 16
ECHO = 26  # ✅ Changed to GPIO 26

# 🔧 Motor Driver Pins (L298N) - Your Existing Pins
IN1_FRONT, IN2_FRONT, IN3_FRONT, IN4_FRONT = 17, 18, 22, 27  # ✅ Your assigned pins
ENA_FRONT, ENB_FRONT = 23, 24  
IN1_BACK, IN2_BACK, IN3_BACK, IN4_BACK = 5, 6, 13, 19
ENA_BACK, ENB_BACK = 20, 21  

# 🏎 Setup Pins
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.setup(IN1_FRONT, GPIO.OUT)
GPIO.setup(IN2_FRONT, GPIO.OUT)
GPIO.setup(IN3_FRONT, GPIO.OUT)
GPIO.setup(IN4_FRONT, GPIO.OUT)
GPIO.setup(ENA_FRONT, GPIO.OUT)
GPIO.setup(ENB_FRONT, GPIO.OUT)

GPIO.setup(IN1_BACK, GPIO.OUT)
GPIO.setup(IN2_BACK, GPIO.OUT)
GPIO.setup(IN3_BACK, GPIO.OUT)
GPIO.setup(IN4_BACK, GPIO.OUT)
GPIO.setup(ENA_BACK, GPIO.OUT)
GPIO.setup(ENB_BACK, GPIO.OUT)

# 📏 Ultrasonic Distance Function
def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.1)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    timeout = start_time + 1  # 1-second timeout

    # Wait for Echo HIGH
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
        if time.time() > timeout:
            print("⚠ Timeout waiting for ECHO HIGH")
            return 999  # Error value

    end_time = time.time()
    timeout = end_time + 1

    # Wait for Echo LOW
    while GPIO.input(ECHO) == 1:
        end_time = time.time()
        if time.time() > timeout:
            print("⚠ Timeout waiting for ECHO LOW")
            return 999  # Error value

    # Calculate Distance
    pulse_duration = end_time - start_time
    distance = pulse_duration * 17150  # Convert to cm

    return round(distance, 2)

# 🏎 Motor Functions
def stop():
    print("🛑 Stopping")
    GPIO.output(IN1_FRONT, 0)
    GPIO.output(IN2_FRONT, 0)
    GPIO.output(IN3_FRONT, 0)
    GPIO.output(IN4_FRONT, 0)
    GPIO.output(IN1_BACK, 0)
    GPIO.output(IN2_BACK, 0)
    GPIO.output(IN3_BACK, 0)
    GPIO.output(IN4_BACK, 0)

def forward():
    print("🚀 Moving Forward")
    GPIO.output(IN1_FRONT, 0)
    GPIO.output(IN2_FRONT, 1)
    GPIO.output(IN3_FRONT, 1)
    GPIO.output(IN4_FRONT, 0)
    GPIO.output(IN1_BACK, 0)
    GPIO.output(IN2_BACK, 1)
    GPIO.output(IN3_BACK, 1)
    GPIO.output(IN4_BACK, 0)

# 🚀 Main Loop
try:
    while True:
        distance = get_distance()
        print(f"📏 Distance: {distance} cm")

        if distance < 30:  # Obstacle detected within 30 cm
            stop()
            print("⚠ Obstacle Detected! Stopping motors.")
        else:
            forward()

        time.sleep(0.5)  # Small delay

except KeyboardInterrupt:
    print("\n🔴 Stopping Program!")
    stop()
    GPIO.cleanup()