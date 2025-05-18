import serial
import pynmea2
import time
import subprocess


class SatellitePositioningSystem:
    def __init__(self, port="/dev/ttyS0", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.latitude = None
        self.longitude = None

    def setup_serial(self):
        """Ensure serial port is free and not used by console login"""
        try:
            linked_device = subprocess.check_output(["readlink", "-f", "/dev/serial0"]).decode().strip()
            if "ttyAMA0" in linked_device:
                service = "serial-getty@ttyAMA0.service"
            elif "ttyS0" in linked_device:
                service = "serial-getty@ttyS0.service"
            else:
                raise Exception("Unknown serial device: " + linked_device)

            subprocess.run(["sudo", "systemctl", "stop", service])
            subprocess.run(["sudo", "systemctl", "disable", service])
            print(f"[GPS] Serial console disabled for {linked_device}")
        except Exception as e:
            print(f"[GPS] Serial setup error: {e}")

    def connect(self):
        """Open serial connection to GPS module"""
        try:
            self.ser = serial.Serial(self.port, baudrate=self.baudrate, timeout=1)
            print(f"[GPS] Connected to {self.port}")
        except Exception as e:
            print(f"[GPS] Failed to connect to {self.port}: {e}")

    def update_position(self):
        """Parse and update current GPS coordinates from available NMEA sentences"""
        if not self.ser:
            print("[GPS] Serial not connected")
            return

        try:
            data = self.ser.readline()
            if not data.startswith(b"$"):
                return

            msg = pynmea2.parse(data.decode("ascii", errors="ignore"))

            if isinstance(msg, (pynmea2.RMC, pynmea2.GGA)):
                if msg.latitude and msg.longitude:
                    self.latitude = msg.latitude
                    self.longitude = msg.longitude
                    print(f"[GPS] Lat: {self.latitude}, Lon: {self.longitude}")

        except pynmea2.ParseError as e:
            print(f"[GPS] Parse error: {e}")
        except Exception as e:
            print(f"[GPS] Unexpected error: {e}")

    def get_coordinates(self):
        """Return the last known GPS coordinates (lat, lon)"""
        return self.latitude, self.longitude

    def close(self):
        if self.ser:
            self.ser.close()
            print("[GPS] Serial connection closed")
