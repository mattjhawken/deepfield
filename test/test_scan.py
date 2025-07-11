"""
Hydrogen line manual testing and background scanning with web dashboard for remote access
"""
"""
Hydrogen line scanner with improved keyboard handling and real-time web dashboard
"""
import sys
import time
import board
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import io
import base64
from datetime import datetime
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import threading
from collections import deque
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json

# Try to import RTL-SDR
try:
    from rtlsdr import RtlSdr

    RTL_SDR_AVAILABLE = True
except ImportError:
    RTL_SDR_AVAILABLE = False
    print("RTL-SDR not available, using simulation mode")


class HydrogenScanner:
    def __init__(self):
        self.kit = MotorKit(i2c=board.I2C())

        # Configuration
        self.config = {
            'center_freq': 1.42040575e9,
            'sample_rate': 2.048e6,
            'gain': 'auto',
            'samples_per_measurement': 256 * 1024
        }

        # Data storage
        self.baseline_level = 0.0
        self.baseline_std = 0.0
        self.scan_data = []
        self.sky_map = None
        self.live_data = deque(maxlen=100)
        self.current_position = {'x': 0, 'y': 0}

        # Control state
        self.scanning = False
        self.calibrating = False
        self.scan_thread = None
        self.calibration_thread = None

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
                'detections_5sigma': 0
            },
            'plots': {},
            'baseline_calibrated': False
        }
        self.data_lock = threading.Lock()

        # Flask app with SocketIO
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'hydrogen_scanner_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.setup_web_routes()

    def setup_web_routes(self):
        """Setup Flask routes and SocketIO events"""

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

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
                steps = data.get('steps', 5)
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

            power = self.measure_power(1.0)
            calibrated = power - self.baseline_level if self.baseline_level != 0 else power

            with self.data_lock:
                self.web_data['current_measurement'] = {
                    'x': self.current_position['x'],
                    'y': self.current_position['y'],
                    'power': calibrated
                }
            self.emit_web_update()

    def emit_web_update(self):
        """Emit real-time update to web dashboard"""
        with self.data_lock:
            self.socketio.emit('data_update', self.web_data)

    def measure_power(self, measurement_time=1.0):
        """Measure power with RTL-SDR or simulation"""
        if not RTL_SDR_AVAILABLE:
            # Realistic simulation
            base_noise = np.random.normal(-45, 2)
            if np.random.random() < 0.15:  # 15% chance of H-line detection
                base_noise += np.random.normal(2, 0.3)
            return base_noise

        try:
            sdr = RtlSdr()
            sdr.sample_rate = self.config['sample_rate']
            sdr.center_freq = self.config['center_freq']
            sdr.gain = self.config['gain']

            samples = sdr.read_samples(self.config['samples_per_measurement'])
            power_db = 10 * np.log10(np.mean(np.abs(samples) ** 2))
            sdr.close()
            return power_db

        except Exception as e:
            print(f"SDR error: {e}")
            return np.random.normal(-45, 2)

    def move_motor_web(self, direction, steps=1):
        """Move motor from web interface"""
        if direction == 'up':
            self.move_motor('y', stepper.BACKWARD, steps)
        elif direction == 'down':
            self.move_motor('y', stepper.FORWARD, steps)
        elif direction == 'left':
            self.move_motor('x', stepper.BACKWARD, steps)
        elif direction == 'right':
            self.move_motor('x', stepper.FORWARD, steps)

        # Update position in web data
        with self.data_lock:
            self.web_data['position'] = self.current_position.copy()
        self.emit_web_update()

    def move_motor(self, axis, direction, steps=1):
        """Move motor and update position tracking"""
        motor = self.kit.stepper1 if axis == 'x' else self.kit.stepper2

        for _ in range(steps):
            motor.onestep(direction=direction, style=stepper.INTERLEAVE)
            time.sleep(0.01)

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

    def calibrate_baseline_web(self, num_samples=30, measurement_time=1.0):
        """Web-controlled baseline calibration"""
        self.calibrating = True

        with self.data_lock:
            self.web_data['status'] = 'calibrating'
            self.web_data['progress'] = 0
            self.web_data['baseline_calibrated'] = False
        self.emit_web_update()

        baseline_data = []

        for i in range(num_samples):
            if not self.calibrating:  # Check for stop signal
                break

            power = self.measure_power(measurement_time)
            baseline_data.append(power)

            progress = (i + 1) / num_samples * 100

            with self.data_lock:
                self.web_data['progress'] = progress
                self.web_data['current_measurement']['power'] = power
            self.emit_web_update()

            time.sleep(0.1)  # Small delay

        if self.calibrating and len(baseline_data) > 0:
            self.baseline_level = np.mean(baseline_data)
            self.baseline_std = np.std(baseline_data)

            with self.data_lock:
                self.web_data['status'] = 'calibration_complete'
                self.web_data['progress'] = 100
                self.web_data['baseline_calibrated'] = True
                self.web_data['stats']['baseline_level'] = self.baseline_level
                self.web_data['stats']['baseline_std'] = self.baseline_std

            print(f"Baseline calibration complete: {self.baseline_level:.2f} ± {self.baseline_std:.2f} dB")
        else:
            with self.data_lock:
                self.web_data['status'] = 'calibration_stopped'

        self.calibrating = False
        self.emit_web_update()

    def start_scan_thread(self, x_steps, y_steps, measurement_time):
        """Start scan in separate thread"""
        self.scan_thread = threading.Thread(
            target=self.run_scan_web,
            args=(x_steps, y_steps, measurement_time),
            daemon=True
        )
        self.scan_thread.start()
        

    def run_scan_web(self, x_steps=15, y_steps=15, measurement_time=1.0,
                     x_step_size=1, y_step_size=1):
        """
        Earth-rotation scan: Move vertically (Y) and let Earth's rotation provide horizontal sweep

        Args:
            y_steps: Number of elevation steps to scan
            measurement_time: Time for each individual measurement
            time_per_position: Time to spend at each Y position (seconds)
            y_step_size: Motor steps between Y positions
        """
        self.scanning = True

        with self.data_lock:
            self.web_data['status'] = 'scanning'
            self.web_data['progress'] = 0
        self.emit_web_update()

        # Calculate how many measurements we'll take at each Y position
        measurements_per_y = int(time_per_position / (measurement_time + 0.1))  # 0.1s for processing
        total_measurements = y_steps * measurements_per_y

        # Initialize data structures
        self.scan_data = []
        # Sky map: Y positions x Time samples (simulating Earth rotation)
        self.sky_map = np.zeros((y_steps, measurements_per_y))

        measurement_count = 0
        scan_start_time = time.time()

        try:
            print(
                f"Starting Earth-rotation scan: {y_steps} Y positions, {measurements_per_y} measurements per position")
            print(f"Total scan time: {(y_steps * time_per_position) / 60:.1f} minutes")

            for y_idx in range(y_steps):
                if not self.scanning:
                    break

                print(f"Moving to Y position {y_idx + 1}/{y_steps}")

                # Move to next Y position (except for first position)
                if y_idx > 0:
                    self.move_motor('y', stepper.FORWARD, y_step_size)
                    time.sleep(0.5)  # Allow settling time

                # Take measurements at this Y position while Earth rotates
                position_start_time = time.time()

                for time_idx in range(measurements_per_y):
                    if not self.scanning:
                        break

                    # Calculate elapsed time since scan start (represents Earth rotation)
                    elapsed_time = time.time() - scan_start_time

                    # Measure power
                    raw_power = self.measure_power(measurement_time)
                    calibrated_power = raw_power - self.baseline_level

                    # Store data with time information
                    self.scan_data.append({
                        'y_idx': y_idx,
                        'time_idx': time_idx,
                        'y_position': y_idx,
                        'elapsed_time': elapsed_time,
                        'raw_power': raw_power,
                        'calibrated_power': calibrated_power,
                        'timestamp': time.time()
                    })

                    # Store in sky map
                    self.sky_map[y_idx, time_idx] = calibrated_power
                    self.live_data.append(calibrated_power)

                    measurement_count += 1
                    progress = (measurement_count / total_measurements) * 100

                    # Update web data
                    with self.data_lock:
                        self.web_data['current_measurement'] = {
                            'x': time_idx,  # X now represents time/Earth rotation
                            'y': y_idx,
                            'power': calibrated_power
                        }
                        self.web_data['progress'] = progress

                        # Update statistics
                        detections_3sigma = int(np.sum(self.sky_map > 3 * self.baseline_std))
                        detections_5sigma = int(np.sum(self.sky_map > 5 * self.baseline_std))
                        max_signal = float(np.max(self.sky_map))

                        self.web_data['stats'].update({
                            'detections_3sigma': detections_3sigma,
                            'detections_5sigma': detections_5sigma,
                            'max_signal': max_signal,
                            'total_points': len(self.scan_data)
                        })

                        # Update plots periodically
                        if measurement_count % 10 == 0:
                            self.update_plots()

                    self.emit_web_update()

                    # Wait until it's time for the next measurement
                    time_to_wait = (position_start_time + (time_idx + 1) * (
                                time_per_position / measurements_per_y)) - time.time()
                    if time_to_wait > 0:
                        time.sleep(time_to_wait)

                print(f"Completed Y position {y_idx + 1}/{y_steps} ({measurements_per_y} measurements)")

            if self.scanning:
                with self.data_lock:
                    self.web_data['status'] = 'scan_complete'
                    self.web_data['progress'] = 100
                    self.update_plots()

                total_time = time.time() - scan_start_time
                print(f"Earth-rotation scan complete!")
                print(f"Total measurements: {len(self.scan_data)}")
                print(f"Total time: {total_time / 60:.1f} minutes")
                print(f"Sky coverage: {y_steps} elevation positions × {measurements_per_y} time samples")
            else:
                with self.data_lock:
                    self.web_data['status'] = 'scan_stopped'

        except Exception as e:
            print(f"Scan error: {e}")
            with self.data_lock:
                self.web_data['status'] = 'scan_error'

        self.scanning = False
        self.emit_web_update()
        
    def update_plots(self):
        """Update all plots"""
        plots = {}
        for plot_type in ['skymap', 'recent', 'distribution']:
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

            elif plot_type == 'recent' and len(self.live_data) > 0:
                ax.plot(list(self.live_data), 'b-', linewidth=2, alpha=0.8)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Baseline')
                if self.baseline_std > 0:
                    ax.axhline(y=3 * self.baseline_std, color='orange', linestyle=':',
                               alpha=0.8, label='3σ threshold')
                    ax.axhline(y=5 * self.baseline_std, color='red', linestyle=':',
                               alpha=0.8, label='5σ threshold')
                ax.set_title('Recent Measurements', fontsize=14)
                ax.set_xlabel('Measurement Number', fontsize=12)
                ax.set_ylabel('Power - Baseline (dB)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()

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


def main():
    """Main execution function"""
    print("Web-Controlled Hydrogen Line Scanner v4.0")
    print("=" * 60)
    print("All controls available through web interface")
    print("Navigate to http://localhost:5000 to control the scanner")
    print("=" * 60)

    scanner = HydrogenScanner()

    try:
        # Start web server
        scanner.start_web_server()

        # Keep server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
