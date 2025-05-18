from src.positioning.gps import SatellitePositioningSystem
from stepper import DRV8833Stepper
from typing import Tuple
import time


class PlanetaryDrive:
    def __init__(
        self,
        azimuth_motor: DRV8833Stepper,
        elevation_motor: DRV8833Stepper,
        steps_per_degree: float = 10.0
    ):
        self.azimuth = 0.0
        self.elevation = 0.0
        self.calibrated = False

        self.gps = SatellitePositioningSystem()
        self.gps.setup_serial()
        self.gps.connect()

        self.az_motor = azimuth_motor
        self.el_motor = elevation_motor
        self.steps_per_degree = steps_per_degree

    def perform_calibration(self):
        """Homing using magnetic sensor (placeholder)"""
        print("[Drive] Starting homing with magnetic calibration...")
        # TODO: Implement GPIO pin read for hall effect / magnet sensor
        self.az_motor.perform_calibration()
        self.el_motor.perform_calibration()

        self.azimuth = 0.0
        self.elevation = 0.0
        self.calibrated = True

        print("[Drive] Homing complete. Position set to Az: 0°, El: 0°")

    def move_to(self, target_azimuth: float, target_elevation: float):
        if not self.calibrated:
            print("[Drive] Cannot move until calibrated")
            return

        print(f"[Drive] Moving to Az: {target_azimuth}°, El: {target_elevation}°")

        az_delta = target_azimuth - self.azimuth
        el_delta = target_elevation - self.elevation

        az_steps = int(abs(az_delta) * self.steps_per_degree)
        el_steps = int(abs(el_delta) * self.steps_per_degree)

        print(f"[Drive] Azimuth delta: {az_delta}° -> {az_steps} steps")
        print(f"[Drive] Elevation delta: {el_delta}° -> {el_steps} steps")

        if az_delta > 0:
            self.az_motor.step_forward(az_steps)
        elif az_delta < 0:
            self.az_motor.step_backward(az_steps)

        if el_delta > 0:
            self.el_motor.step_forward(el_steps)
        elif el_delta < 0:
            self.el_motor.step_backward(el_steps)

        self.azimuth = target_azimuth
        self.elevation = target_elevation

    def get_position(self) -> Tuple[float, float]:
        return self.azimuth, self.elevation

    def stop(self):
        self.az_motor.stop()
        self.el_motor.stop()

    def shutdown(self):
        self.stop()
        self.az_motor.cleanup()
        self.el_motor.cleanup()
