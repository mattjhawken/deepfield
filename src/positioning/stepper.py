try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    import types
    print("RPi.GPIO not found. Using mock GPIO.")
    GPIO = types.SimpleNamespace()
    GPIO.BCM = "BCM"
    GPIO.OUT = "OUT"
    GPIO.LOW = 0
    GPIO.HIGH = 1
    GPIO.IN = "IN"
    GPIO.input = lambda pin: 1  # Always return 1 (not triggered)

    def dummy(*args, **kwargs): pass
    GPIO.setmode = dummy
    GPIO.setup = dummy
    GPIO.output = dummy
    GPIO.cleanup = dummy


import time


class DRV8833Stepper:
    def __init__(self, ain1, ain2, bin1, bin2, delay=0.01):
        self.AZ_IN1 = ain1
        self.AZ_IN2 = ain2
        self.EL_IN1 = bin1
        self.EL_IN2 = bin2

        self.delay = delay

        # Hall effect sensor input pins
        self.AZ_SENSOR = 5
        self.EL_SENSOR = 6

        # State
        self.azimuth = None
        self.elevation = None
        self.calibrated = False

        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in [self.AZ_IN1, self.AZ_IN2, self.EL_IN1, self.EL_IN2]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        for sensor in [self.AZ_SENSOR, self.EL_SENSOR]:
            GPIO.setup(sensor, GPIO.IN)

    def stop_motor(self, in1, in2):
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)

    def move_motor_forward(self, in1, in2):
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)

    def perform_calibration(self):
        print("[Drive] Starting homing with magnetic calibration...")

        # Calibrate azimuth
        print("[Drive] Homing azimuth axis...")
        self.move_motor_forward(self.AZ_IN1, self.AZ_IN2)
        start_time = time.time()
        while GPIO.input(self.AZ_SENSOR):
            if time.time() - start_time > 10:
                print("[Warning] Azimuth calibration timeout.")
                break
            time.sleep(0.01)

        self.stop_motor(self.AZ_IN1, self.AZ_IN2)
        print("[Drive] Azimuth homed.")

        # Calibrate elevation
        print("[Drive] Homing elevation axis...")
        self.move_motor_forward(self.EL_IN1, self.EL_IN2)
        start_time = time.time()
        while GPIO.input(self.EL_SENSOR):
            if time.time() - start_time > 10:
                print("[Warning] Elevation calibration timeout.")
                break
            time.sleep(0.01)

        self.stop_motor(self.EL_IN1, self.EL_IN2)
        print("[Drive] Elevation homed.")

        # Set state
        self.azimuth = 0.0
        self.elevation = 0.0
        self.calibrated = True
        print("[Drive] Homing complete. Position set to Az: 0°, El: 0°")

    def cleanup(self):
        GPIO.cleanup()

    def __del__(self):
        self.cleanup()
