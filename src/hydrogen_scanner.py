"""
Hydrogen line manual testing and background scanning with web dashboard for remote access
"""
import threading
import time
import numpy as np
import scipy.signal
import logging
from contextlib import contextmanager
from collections import deque
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
import json
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTL-SDR availability check
try:
    from rtlsdr import RtlSdr
    RTL_SDR_AVAILABLE = True
except ImportError:
    RTL_SDR_AVAILABLE = False
    logger.warning("RTL-SDR not available, using simulation mode")

# Motor control
try:
    from adafruit_motor import stepper
    from adafruit_motorkit import MotorKit
    import board
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    logger.warning("Motor control not available")


class SDRManager:
    """Thread-safe SDR resource manager with proper cleanup"""

    def __init__(self, config):
        self.config = config
        self.sdr_lock = threading.Lock()
        self.sdr_instance = None
        self.last_used = 0
        self.connection_timeout = 5.0
        self.max_retries = 3
        self.retry_delay = 1.0
        self._shutdown = False

    @contextmanager
    def get_sdr(self):
        """Context manager for safe SDR access"""
        if self._shutdown:
            raise RuntimeError("SDR Manager is shutting down")

        sdr = None
        try:
            with self.sdr_lock:
                if self._shutdown:
                    raise RuntimeError("SDR Manager is shutting down")
                sdr = self._get_or_create_sdr()
                yield sdr
        except Exception as e:
            logger.error(f"SDR error: {e}")
            if sdr:
                self._cleanup_sdr(sdr)
            raise
        finally:
            if not self._shutdown:
                self.last_used = time.time()

    def _get_or_create_sdr(self):
        """Get existing SDR or create new one with proper initialization"""
        if not RTL_SDR_AVAILABLE or self._shutdown:
            raise RuntimeError("RTL-SDR not available or shutting down")

        # If we have an existing connection, check if it's still good
        if self.sdr_instance is not None and not self._shutdown:
            try:
                # Test the connection
                self.sdr_instance.get_center_freq()
                return self.sdr_instance
            except:
                logger.warning("Existing SDR connection failed, creating new one")
                self._cleanup_sdr(self.sdr_instance)
                self.sdr_instance = None

        # Create new SDR connection with retries
        for attempt in range(self.max_retries):
            if self._shutdown:
                raise RuntimeError("SDR Manager is shutting down")

            try:
                logger.info(f"Initializing SDR (attempt {attempt + 1}/{self.max_retries})")

                sdr = RtlSdr()

                # Configure with proper timing
                sdr.sample_rate = self.config['sample_rate']
                time.sleep(0.1)

                sdr.center_freq = self.config['center_freq']
                time.sleep(0.1)

                sdr.gain = self.config['gain']
                time.sleep(0.1)

                # Test the connection
                test_freq = sdr.get_center_freq()
                logger.info(f"SDR initialized successfully at {test_freq} Hz")

                self.sdr_instance = sdr
                return sdr

            except Exception as e:
                logger.error(f"SDR initialization attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1 and not self._shutdown:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to initialize SDR after {self.max_retries} attempts")

    def _cleanup_sdr(self, sdr):
        """Properly cleanup SDR resources"""
        if sdr:
            try:
                sdr.close()
                logger.info("SDR connection closed")
            except:
                pass

    def cleanup(self):
        """Clean up all SDR resources"""
        logger.info("Cleaning up SDR Manager...")
        self._shutdown = True

        with self.sdr_lock:
            if self.sdr_instance:
                self._cleanup_sdr(self.sdr_instance)
                self.sdr_instance = None


def detect_hydrogen_line(freqs, psd):
    """Detect hydrogen line in spectrum"""
    # Find center frequency bin
    center_idx = np.argmin(np.abs(freqs))

    # Define hydrogen line window (±10 kHz)
    line_width = int(10e3 / (freqs[1] - freqs[0]))  # Convert to bins
    start_idx = max(0, center_idx - line_width)
    end_idx = min(len(psd), center_idx + line_width)

    # Extract line region and nearby continuum
    line_power = np.mean(psd[start_idx:end_idx])

    # Continuum from sides
    continuum_left = np.mean(psd[max(0, start_idx - line_width):start_idx])
    continuum_right = np.mean(psd[end_idx:min(len(psd), end_idx + line_width)])
    continuum = (continuum_left + continuum_right) / 2

    return line_power - continuum


class HydrogenScanner:
    def __init__(self):
        if MOTOR_AVAILABLE:
            self.kit = MotorKit(i2c=board.I2C())
        else:
            self.kit = None

        # Configuration
        self.config = {
            'center_freq': 1.42040575e9,
            'sample_rate': 2.048e6,
            'gain': 35
        }

        # Initialize SDR manager
        self.sdr_manager = SDRManager(self.config)

        # Data storage
        self.baseline_level = 0.0
        self.baseline_std = 0.0
        self.scan_data = []
        self.sky_map = None
        self.live_data = deque(maxlen=100)
        self.current_position = {'x': 0, 'y': 0}
        self.spectrum_history = deque(maxlen=100)
        self.h_line_history = deque(maxlen=100)
        self.detection_log = []
        self.current_spectrum = None

        # Web dashboard data
        self.web_data = {
            'status': 'idle',
            'progress': 0,
            'current_measurement': {'x': 0, 'y': 0, 'power': 0},
            'position': {'x': 0, 'y': 0},
            'stats': {
                'baseline_level': 0,
                'baseline_std': 0,
                'max_signal': 0,
                'total_points': 0,
                'detections_3sigma': 0,
                'detections_5sigma': 0,
                'h_line_detections': 0,
                'avg_h_line_strength': 0,
                'peak_h_line_strength': 0,
                'h_line_baseline': 0
            },
            'plots': {},
            'baseline_calibrated': False
        }

        # Control state - using threading.Event for better control
        self.shutdown_event = threading.Event()
        self.scanning_event = threading.Event()
        self.calibrating_event = threading.Event()
        self.scan_thread = None
        self.calibration_thread = None
        self.data_lock = threading.Lock()

        # Flask app with SocketIO
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'hydrogen_scanner_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.setup_web_routes()
        self.setup_signal_handlers()

    def check_should_continue(self):
        """Check if operations should continue"""
        return not self.shutdown_event.is_set()

    def stop_current_operation(self):
        """Stop current operation gracefully"""
        self.scanning_event.clear()
        self.calibrating_event.clear()
        logger.info("Stopping current operation...")

    def measure_power_spectrum(self, measurement_time=1.0):
        """Measure power spectrum with improved error handling"""
        if not RTL_SDR_AVAILABLE:
            # Simulate realistic H-line spectrum
            freqs = np.linspace(-1e6, 1e6, 1024)
            noise = np.random.normal(-45, 2, len(freqs))

            # Add hydrogen line peak
            if np.random.random() < 0.15:
                h_line_idx = len(freqs) // 2
                width = 20
                amplitude = np.random.normal(5, 1)
                gaussian = amplitude * np.exp(-0.5 * ((np.arange(len(freqs)) - h_line_idx) / width) ** 2)
                noise += gaussian

            return freqs, noise

        if not self.check_should_continue():
            raise RuntimeError("Operation cancelled")

        max_retries = 3
        for attempt in range(max_retries):
            if not self.check_should_continue():
                raise RuntimeError("Operation cancelled")

            try:
                with self.sdr_manager.get_sdr() as sdr:
                    # Read samples with timeout protection
                    samples = sdr.read_samples(int(self.config['sample_rate'] * measurement_time))

                    if len(samples) == 0:
                        raise RuntimeError("No samples received from SDR")

                    # Compute power spectral density
                    freqs, psd = scipy.signal.welch(samples, sdr.sample_rate, nperseg=1024)
                    freqs = freqs - sdr.sample_rate / 2
                    psd_db = 10 * np.log10(psd + 1e-12)

                    return freqs, psd_db

            except Exception as e:
                logger.error(f"SDR measurement attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1 and self.check_should_continue():
                    time.sleep(0.5)
                else:
                    logger.error("All SDR measurement attempts failed, using simulated data")
                    # Fall back to simulation
                    freqs = np.linspace(-1e6, 1e6, 1024)
                    noise = np.random.normal(-45, 2, len(freqs))
                    return freqs, noise

    def enhanced_measurement(self, measurement_time=1.0):
        """Enhanced measurement with improved error handling"""
        if not self.check_should_continue():
            raise RuntimeError("Operation cancelled")

        try:
            freqs, psd = self.measure_power_spectrum(measurement_time)

            # Store spectrum
            self.current_spectrum = (freqs, psd)
            self.spectrum_history.append(psd)

            # Detect hydrogen line
            h_line_power = detect_hydrogen_line(freqs, psd)
            total_power = np.mean(psd)

            # Store H-line measurement
            self.h_line_history.append({
                'timestamp': time.time(),
                'h_line_power': h_line_power,
                'total_power': total_power,
                'position': self.current_position.copy()
            })

            # Log significant detections
            if self.baseline_std > 0 and h_line_power > 3 * self.baseline_std:
                detection = {
                    'timestamp': time.time(),
                    'position': self.current_position.copy(),
                    'h_line_power': h_line_power,
                    'total_power': total_power,
                    'significance': h_line_power / self.baseline_std
                }
                self.detection_log.append(detection)

            return total_power, h_line_power

        except Exception as e:
            if "Operation cancelled" in str(e):
                raise
            logger.error(f"Enhanced measurement failed: {e}")
            return -45.0, 0.0

    def calibrate_baseline_web(self, num_samples=30, measurement_time=1.0):
        """Web-controlled baseline calibration with improved termination"""
        if not self.calibrating_event.is_set():
            return

        logger.info(f"Starting baseline calibration with {num_samples} samples")

        try:
            with self.data_lock:
                self.web_data['status'] = 'calibrating'
                self.web_data['progress'] = 0
                self.web_data['baseline_calibrated'] = False
            self.emit_web_update()

            baseline_data = []
            failed_measurements = 0
            max_failures = num_samples // 3

            for i in range(num_samples):
                if not self.calibrating_event.is_set() or not self.check_should_continue():
                    logger.info("Calibration cancelled")
                    break

                try:
                    total_power, h_line_power = self.enhanced_measurement(measurement_time)
                    baseline_data.append(total_power)

                    progress = (i + 1) / num_samples * 100

                    with self.data_lock:
                        self.web_data['progress'] = progress
                        self.web_data['current_measurement'] = {
                            'x': self.current_position['x'],
                            'y': self.current_position['y'],
                            'power': total_power,
                            'h_line_power': h_line_power
                        }
                    self.emit_web_update()

                    # Check for cancellation more frequently
                    if not self.calibrating_event.is_set():
                        break

                    time.sleep(0.1)

                except Exception as e:
                    if "Operation cancelled" in str(e):
                        logger.info("Calibration cancelled")
                        break
                    logger.error(f"Calibration measurement {i + 1} failed: {e}")
                    failed_measurements += 1
                    if failed_measurements > max_failures:
                        logger.error("Too many failed measurements, aborting calibration")
                        break

            # Process results if we have enough data and weren't cancelled
            if (self.calibrating_event.is_set() and
                    self.check_should_continue() and
                    len(baseline_data) > num_samples // 2):

                self.baseline_level = np.mean(baseline_data)
                self.baseline_std = np.std(baseline_data)

                with self.data_lock:
                    self.web_data['status'] = 'calibration_complete'
                    self.web_data['progress'] = 100
                    self.web_data['baseline_calibrated'] = True
                    self.web_data['stats']['baseline_level'] = self.baseline_level
                    self.web_data['stats']['baseline_std'] = self.baseline_std

                logger.info(f"Baseline calibration complete: {self.baseline_level:.2f} ± {self.baseline_std:.2f} dB")
            else:
                with self.data_lock:
                    if self.calibrating_event.is_set():
                        self.web_data['status'] = 'calibration_failed'
                    else:
                        self.web_data['status'] = 'calibration_cancelled'
                logger.info("Baseline calibration stopped or failed")

        except Exception as e:
            logger.error(f"Calibration error: {e}")
            with self.data_lock:
                self.web_data['status'] = 'calibration_error'
        finally:
            self.calibrating_event.clear()
            self.emit_web_update()

    def run_scan_web(self, x_steps=64, y_steps=64, measurement_time=1.0,
                     x_step_size=1, y_step_size=1):
        """Web-controlled scan with improved termination"""
        if not self.scanning_event.is_set():
            return

        logger.info(f"Starting scan: {x_steps}x{y_steps} points")

        try:
            with self.data_lock:
                self.web_data['status'] = 'scanning'
                self.web_data['progress'] = 0
            self.emit_web_update()

            # Initialize
            self.sky_map = np.zeros((y_steps, x_steps))
            self.h_line_map = np.zeros((y_steps, x_steps))
            self.scan_data = []
            total_points = x_steps * y_steps
            failed_measurements = 0
            max_failures = total_points // 10

            for y in range(y_steps):
                if not self.scanning_event.is_set() or not self.check_should_continue():
                    logger.info("Scan cancelled")
                    break

                for x in range(x_steps):
                    if not self.scanning_event.is_set() or not self.check_should_continue():
                        logger.info("Scan cancelled")
                        break

                    try:
                        # Enhanced measurement with retry logic
                        total_power, h_line_power = self.enhanced_measurement(measurement_time)
                        calibrated_power = total_power - self.baseline_level

                        # Store data
                        self.scan_data.append({
                            'x': x, 'y': y,
                            'raw_power': total_power,
                            'calibrated_power': calibrated_power,
                            'h_line_power': h_line_power,
                            'timestamp': time.time()
                        })

                        self.sky_map[y, x] = calibrated_power
                        self.h_line_map[y, x] = h_line_power

                        # Update web data
                        points_completed = len(self.scan_data)
                        progress = (points_completed / total_points) * 100

                        with self.data_lock:
                            self.web_data['current_measurement'] = {
                                'x': x, 'y': y,
                                'power': calibrated_power,
                                'h_line_power': h_line_power
                            }
                            self.web_data['progress'] = progress

                            # Update statistics
                            if self.baseline_std > 0:
                                h_line_detections = sum(
                                    1 for d in self.scan_data if d['h_line_power'] > 3 * self.baseline_std)
                                self.web_data['stats'].update({
                                    'h_line_detections': h_line_detections,
                                    'avg_h_line_strength': float(np.mean([d['h_line_power'] for d in self.scan_data])),
                                    'peak_h_line_strength': float(np.max([d['h_line_power'] for d in self.scan_data])),
                                    'detections_3sigma': int(np.sum(self.sky_map > 3 * self.baseline_std)),
                                    'detections_5sigma': int(np.sum(self.sky_map > 5 * self.baseline_std)),
                                    'max_signal': float(np.max(self.sky_map)),
                                    'total_points': len(self.scan_data)
                                })

                            if points_completed % 10 == 0:
                                self.update_plots()

                        self.emit_web_update()

                        # Check for cancellation more frequently
                        if not self.scanning_event.is_set():
                            break

                    except Exception as e:
                        if "Operation cancelled" in str(e):
                            logger.info("Scan cancelled")
                            break
                        logger.error(f"Scan measurement at ({x}, {y}) failed: {e}")
                        failed_measurements += 1
                        if failed_measurements > max_failures:
                            logger.error("Too many failed measurements, aborting scan")
                            break

                    # Move X (horizontal sweep)
                    if x < x_steps - 1 and self.scanning_event.is_set():
                        self.move_motor('x', stepper.FORWARD if MOTOR_AVAILABLE else None, x_step_size)

                # Move Y (next row)
                if (y < y_steps - 1 and
                        self.scanning_event.is_set() and
                        self.check_should_continue()):
                    self.move_motor('y', stepper.FORWARD if MOTOR_AVAILABLE else None, y_step_size)
                    self.move_motor('x', stepper.BACKWARD if MOTOR_AVAILABLE else None, (x_steps - 1) * x_step_size)

            # Set final status
            with self.data_lock:
                if (self.scanning_event.is_set() and
                        self.check_should_continue()):
                    self.web_data['status'] = 'scan_complete'
                    self.web_data['progress'] = 100
                    self.update_plots()
                    logger.info(f"Scan complete! {len(self.scan_data)} points collected")
                else:
                    self.web_data['status'] = 'scan_cancelled'
                    logger.info("Scan cancelled")

        except Exception as e:
            logger.error(f"Scan error: {e}")
            with self.data_lock:
                self.web_data['status'] = 'scan_error'
        finally:
            self.scanning_event.clear()
            self.emit_web_update()

    def shutdown(self):
        """Properly shutdown the scanner"""
        logger.info("Shutting down scanner...")

        # Signal shutdown to all operations
        self.shutdown_event.set()
        self.stop_current_operation()

        # Wait for threads to finish with timeout
        if self.scan_thread and self.scan_thread.is_alive():
            logger.info("Waiting for scan thread to finish...")
            self.scan_thread.join(timeout=10)
            if self.scan_thread.is_alive():
                logger.warning("Scan thread did not finish gracefully")

        if self.calibration_thread and self.calibration_thread.is_alive():
            logger.info("Waiting for calibration thread to finish...")
            self.calibration_thread.join(timeout=10)
            if self.calibration_thread.is_alive():
                logger.warning("Calibration thread did not finish gracefully")

        # Cleanup SDR resources
        self.sdr_manager.cleanup()

        logger.info("Scanner shutdown complete")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def setup_web_routes(self):
        """Setup Flask routes and SocketIO events"""
        @self.app.route('/')
        def dashboard():
            return render_template('../templates/dashboard.html')

        @self.app.route('/api/data')
        def get_data():
            with self.data_lock:
                return jsonify(self.web_data)

        @self.socketio.on('connect')
        def handle_connect():
            print("Client connected to web dashboard")
            with self.data_lock:
                emit('data_update', self.web_data)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print("Client disconnected from web dashboard")

        # Motor control events
        @self.socketio.on('move_motor')
        def handle_move_motor(data):
            try:
                direction = data['direction']
                steps = data.get('steps', 1)
                self.move_motor_web(direction, steps)
            except Exception as e:
                emit('error', {'message': f'Motor error: {str(e)}'})

        @self.socketio.on('reset_position')
        def handle_reset_position():
            self.current_position = {'x': 0, 'y': 0}
            with self.data_lock:
                self.web_data['position'] = self.current_position.copy()
            self.emit_web_update()

        # Calibration events
        @self.socketio.on('start_calibration')
        def handle_start_calibration(data):
            if self.calibrating or self.scanning:
                emit('error', {'message': 'System busy'})
                return

            samples = data.get('samples', 30)
            measurement_time = data.get('measurement_time', 1.0)
            self.start_calibration_thread(samples, measurement_time)

        @self.socketio.on('stop_calibration')
        def handle_stop_calibration():
            self.calibrating = False

        # Scan events
        @self.socketio.on('start_scan')
        def handle_start_scan(data):
            if self.scanning or self.calibrating:
                emit('error', {'message': 'System busy'})
                return

            if not self.web_data['baseline_calibrated']:
                emit('error', {'message': 'Please calibrate baseline first'})
                return

            x_steps = data.get('x_steps', 15)
            y_steps = data.get('y_steps', 15)
            measurement_time = data.get('measurement_time', 1.0)
            self.start_scan_thread(x_steps, y_steps, measurement_time)

        @self.socketio.on('stop_scan')
        def handle_stop_scan():
            self.scanning = False

        # Quick measurement
        @self.socketio.on('quick_measurement')
        def handle_quick_measurement():
            if self.scanning or self.calibrating:
                emit('error', {'message': 'System busy'})
                return

            total_power, h_line_power = self.enhanced_measurement(1.0)
            calibrated_power = total_power - self.baseline_level if self.baseline_level != 0 else total_power

            with self.data_lock:
                self.web_data['current_measurement'] = {
                    'x': self.current_position['x'],
                    'y': self.current_position['y'],
                    'power': calibrated_power,
                    'h_line_power': h_line_power
                }
            self.emit_web_update()

    def emit_web_update(self):
        """Emit real-time update to web dashboard"""
        with self.data_lock:
            self.socketio.emit('data_update', self.web_data)

    def move_motor_web(self, direction, steps=1):
        """Move motor from web interface"""
        if direction == 'up':
            self.move_motor('y', stepper.BACKWARD, steps, 0.05)
        elif direction == 'down':
            self.move_motor('y', stepper.FORWARD, steps, 0.05)
        elif direction == 'left':
            self.move_motor('x', stepper.BACKWARD, steps, 0.05)
        elif direction == 'right':
            self.move_motor('x', stepper.FORWARD, steps, 0.05)

        # Update position in web data
        with self.data_lock:
            self.web_data['position'] = self.current_position.copy()
        self.emit_web_update()

    def move_motor(self, axis, direction, steps=1, sleep=0.02):
        """Move motor and update position tracking"""
        motor = self.kit.stepper1 if axis == 'x' else self.kit.stepper2

        for _ in range(steps):
            motor.onestep(direction=direction, style=stepper.INTERLEAVE)
            time.sleep(sleep)

        # Update position
        if axis == 'x':
            self.current_position['x'] += steps if direction == stepper.FORWARD else -steps
        else:
            self.current_position['y'] += steps if direction == stepper.FORWARD else -steps

    def start_calibration_thread(self, samples, measurement_time):
        """Start calibration in separate thread"""
        self.calibration_thread = threading.Thread(
            target=self.calibrate_baseline_web,
            args=(samples, measurement_time),
            daemon=True
        )
        self.calibration_thread.start()

    def start_scan_thread(self, x_steps, y_steps, measurement_time):
        """Start scan in separate thread"""
        self.scan_thread = threading.Thread(
            target=self.run_scan_web,
            args=(x_steps, y_steps, measurement_time),
            daemon=True
        )
        self.scan_thread.start()

    def update_plots(self):
        """Update all plots"""
        plots = {}
        for plot_type in ['skymap', 'hydrogen_line', 'distribution', 'spectrum']:
            plot_img = self.create_plot_image(plot_type)
            if plot_img:
                plots[plot_type] = plot_img
        self.web_data['plots'] = plots

    def create_plot_image(self, plot_type='skymap'):
        """Create plot and return as base64 encoded image"""
        fig, ax = plt.subplots(figsize=(10, 8))

        try:
            if plot_type == 'skymap' and self.sky_map is not None:
                im = ax.imshow(self.sky_map, origin='lower', cmap='plasma', aspect='auto')
                ax.set_title('Sky Map (dB above baseline)', fontsize=14)
                ax.set_xlabel('Azimuth Steps', fontsize=12)
                ax.set_ylabel('Elevation Steps', fontsize=12)
                plt.colorbar(im, ax=ax, label='Signal Strength (dB)')

                # Add detection markers
                if self.baseline_std > 0:
                    y_coords, x_coords = np.where(self.sky_map > 3 * self.baseline_std)
                    ax.scatter(x_coords, y_coords, c='red', s=30, alpha=0.7, marker='o')

            elif plot_type == 'hydrogen_line' and len(self.h_line_history) > 0:
                # Plot hydrogen line detections over time
                times = [h['timestamp'] for h in self.h_line_history]
                h_powers = [h['h_line_power'] for h in self.h_line_history]

                # Convert timestamps to relative time in minutes
                if times:
                    start_time = min(times)
                    rel_times = [(t - start_time) / 60 for t in times]  # Convert to minutes

                    ax.plot(rel_times, h_powers, 'b-', linewidth=2, alpha=0.8, label='H-line Power')
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Baseline')

                    if self.baseline_std > 0:
                        ax.axhline(y=3 * self.baseline_std, color='orange', linestyle=':',
                                   alpha=0.8, label='3σ Detection')
                        ax.axhline(y=5 * self.baseline_std, color='red', linestyle=':',
                                   alpha=0.8, label='5σ Strong Detection')

                    # Highlight detections
                    detection_times = [((h['timestamp'] - start_time) / 60) for h in self.h_line_history
                                       if h['h_line_power'] > 3 * self.baseline_std]
                    detection_powers = [h['h_line_power'] for h in self.h_line_history
                                        if h['h_line_power'] > 3 * self.baseline_std]

                    if detection_times:
                        ax.scatter(detection_times, detection_powers, c='red', s=50,
                                   alpha=0.8, marker='*', zorder=5, label='Detections')

                    ax.set_title('Hydrogen Line Monitoring', fontsize=14)
                    ax.set_xlabel('Time (minutes)', fontsize=12)
                    ax.set_ylabel('H-line Power (dB)', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    # Add statistics text
                    if len(h_powers) > 0:
                        avg_power = np.mean(h_powers)
                        max_power = np.max(h_powers)
                        detections = sum(1 for p in h_powers if p > 3 * self.baseline_std)

                        stats_text = f'Avg: {avg_power:.2f} dB\nMax: {max_power:.2f} dB\nDetections: {detections}'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            elif plot_type == 'spectrum' and hasattr(self, 'current_spectrum') and self.current_spectrum:
                freqs, psd = self.current_spectrum
                ax.plot(freqs / 1e3, psd, 'b-', linewidth=1)  # Convert to kHz
                ax.set_title('Current Power Spectrum', fontsize=14)
                ax.set_xlabel('Frequency Offset (kHz)', fontsize=12)
                ax.set_ylabel('Power Spectral Density (dB/Hz)', fontsize=12)
                ax.grid(True, alpha=0.3)

                # Highlight hydrogen line region
                ax.axvspan(-10, 10, alpha=0.2, color='red', label='H-line region')
                ax.legend()

            elif plot_type == 'waterfall' and len(self.spectrum_history) > 0:
                # Show frequency evolution over time
                waterfall_data = np.array(list(self.spectrum_history))
                im = ax.imshow(waterfall_data.T, aspect='auto', origin='lower',
                               cmap='plasma', extent=[0, len(waterfall_data), -500, 500])
                ax.set_title('Spectral Waterfall', fontsize=14)
                ax.set_xlabel('Time (measurements)', fontsize=12)
                ax.set_ylabel('Frequency Offset (kHz)', fontsize=12)
                plt.colorbar(im, ax=ax, label='Power (dB)')

            elif plot_type == 'distribution' and len(self.scan_data) > 5:
                all_powers = [d['calibrated_power'] for d in self.scan_data]
                ax.hist(all_powers, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Baseline')
                if self.baseline_std > 0:
                    ax.axvline(x=3 * self.baseline_std, color='orange', linestyle=':',
                               alpha=0.8, label='3σ threshold')
                ax.set_title('Signal Distribution', fontsize=14)
                ax.set_xlabel('Power - Baseline (dB)', fontsize=12)
                ax.set_ylabel('Count', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()

            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.read()).decode()
            plt.close(fig)

            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            print(f"Error creating plot: {e}")
            plt.close(fig)
            return None

    def save_data(self, filename_prefix=None):
        """Save scan data"""
        if not filename_prefix:
            filename_prefix = datetime.now().strftime("scan_%Y%m%d_%H%M%S")

        saved_files = []

        if self.sky_map is not None:
            skymap_file = f"{filename_prefix}_skymap.npy"
            np.save(skymap_file, self.sky_map)
            saved_files.append(skymap_file)

        if self.scan_data:
            scan_array = np.array([(d['x'], d['y'], d['raw_power'], d['calibrated_power'], d['timestamp'])
                                   for d in self.scan_data])
            data_file = f"{filename_prefix}_data.npy"
            np.save(data_file, scan_array)
            saved_files.append(data_file)

        return saved_files

    def start_web_server(self):
        """Start the web server with SocketIO"""

        def run_server():
            self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=False)

        web_thread = threading.Thread(target=run_server, daemon=True)
        web_thread.start()
        print("Web-controlled scanner started at http://localhost:5000")
        return web_thread
