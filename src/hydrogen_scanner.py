"""
Hydrogen line manual testing and background scanning with web dashboard for remote access
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import board
import scipy

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


# Try to import RTL-SDR
try:
    from rtlsdr import RtlSdr

    RTL_SDR_AVAILABLE = True
except ImportError:
    RTL_SDR_AVAILABLE = False
    print("RTL-SDR not available, using simulation mode")


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
        self.kit = MotorKit(i2c=board.I2C())

        # Configuration
        self.config = {
            'center_freq': 1.42040575e9,
            'sample_rate': 2.048e6,
            'gain': 35
        }

        # Data storage
        self.baseline_level = 0.0
        self.baseline_std = 0.0
        self.scan_data = []
        self.sky_map = None
        self.live_data = deque(maxlen=100)
        self.current_position = {'x': 0, 'y': 0}
        self.spectrum_history = deque(maxlen=100)  # Store recent spectra
        self.h_line_history = deque(maxlen=100)  # Store H-line detections
        self.detection_log = []  # Log of significant detections
        self.current_spectrum = None

        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.config['sample_rate']
        self.sdr.center_freq = self.config['center_freq']
        self.sdr.gain = self.config['gain']

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

        # Control state
        self.scanning = False
        self.calibrating = False
        self.scan_thread = None
        self.calibration_thread = None
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

    def measure_power_spectrum(self, measurement_time=1.0):
        """Measure power spectrum instead of just total power"""
        if not RTL_SDR_AVAILABLE:
            # Simulate realistic H-line spectrum
            freqs = np.linspace(-1e6, 1e6, 1024)  # ±1MHz around center
            noise = np.random.normal(-45, 2, len(freqs))

            # Add hydrogen line peak
            if np.random.random() < 0.15:
                h_line_idx = len(freqs) // 2  # Center frequency
                width = 20  # Line width in bins
                amplitude = np.random.normal(5, 1)
                gaussian = amplitude * np.exp(-0.5 * ((np.arange(len(freqs)) - h_line_idx) / width) ** 2)
                noise += gaussian

            return freqs, noise

        try:
            samples = self.sdr.read_samples(int(self.config['sample_rate'] * measurement_time))

            # Compute power spectral density
            freqs, psd = scipy.signal.welch(samples, self.sdr.sample_rate, nperseg=1024)
            freqs = freqs - self.sdr.sample_rate / 2  # Center at 0
            psd_db = 10 * np.log10(psd)

            return freqs, psd_db

        except Exception as e:
            print(f"SDR error: {e}")

    def enhanced_measurement(self, measurement_time=1.0):
        """Enhanced measurement with spectral analysis"""
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
        if h_line_power > 3 * self.baseline_std:
            detection = {
                'timestamp': time.time(),
                'position': self.current_position.copy(),
                'h_line_power': h_line_power,
                'total_power': total_power,
                'significance': h_line_power / self.baseline_std if self.baseline_std > 0 else 0
            }
            self.detection_log.append(detection)

        return total_power, h_line_power

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

    def move_motor(self, axis, direction, steps=1, sleep=0.05):
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

            power = self.enhanced_measurement(measurement_time)
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

    def run_scan_web(self, x_steps=64, y_steps=64, measurement_time=1.0,
                     x_step_size=1, y_step_size=1):
        """Web-controlled scan with hydrogen line monitoring"""
        self.scanning = True

        with self.data_lock:
            self.web_data['status'] = 'scanning'
            self.web_data['progress'] = 0
        self.emit_web_update()

        # Initialize
        self.sky_map = self.sky_map = np.full((y_steps, x_steps), np.nan)
        self.h_line_map = np.zeros((y_steps, x_steps))  # New H-line map
        self.scan_data = []
        total_points = x_steps * y_steps

        try:
            for x in range(x_steps):
                if not self.scanning:
                    break

                for y in range(y_steps):
                    if not self.scanning:
                        break

                    # Enhanced measurement
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
                    self.h_line_map[y, x] = h_line_power  # Store H-line data
                    self.live_data.append(calibrated_power)

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

                        # Update H-line statistics
                        h_line_detections = sum(1 for d in self.scan_data if d['h_line_power'] > 3 * self.baseline_std)
                        avg_h_line = np.mean([d['h_line_power'] for d in self.scan_data])
                        peak_h_line = max([d['h_line_power'] for d in self.scan_data])

                        self.web_data['stats'].update({
                            'h_line_detections': h_line_detections,
                            'avg_h_line_strength': float(avg_h_line),
                            'peak_h_line_strength': float(peak_h_line),
                            'detections_3sigma': int(np.sum(self.sky_map > 3 * self.baseline_std)),
                            'detections_5sigma': int(np.sum(self.sky_map > 5 * self.baseline_std)),
                            'max_signal': float(np.max(self.sky_map)),
                            'total_points': len(self.scan_data)
                        })

                        if points_completed % 10 == 0:
                            self.update_plots()

                        if points_completed % 50 == 0:
                            self.save_data()

                    self.emit_web_update()

                    # Move X (horizontal sweep)
                    if y < y_steps - 1:
                        self.move_motor('y', stepper.FORWARD, y_step_size)

                # Move X (next column)
                if x < x_steps - 1 and self.scanning:
                    # self.move_motor('y', stepper.FORWARD, x_step_size)
                    self.move_motor('y', stepper.BACKWARD, (y_steps - 1) * y_step_size)

            if self.scanning:
                with self.data_lock:
                    self.web_data['status'] = 'scan_complete'
                    self.web_data['progress'] = 100
                    self.update_plots()
                print(f"Scan complete! {len(self.scan_data)} points collected")
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
        for plot_type in ['skymap', 'hydrogen_line', 'distribution', 'spectrum']:
            plot_img = self.create_plot_image(plot_type)
            if plot_img:
                plots[plot_type] = plot_img
        self.web_data['plots'] = plots

    def create_plot_image(self, plot_type='skymap'):
        """Create plot and return as base64 encoded image"""
        fig, ax = plt.subplots(figsize=(18, 8))

        try:
            if plot_type == 'skymap' and self.sky_map is not None:
                # Create masked array for display
                plot_data = self.sky_map.copy()
                plot_data[plot_data == 0] = np.nan

                if np.all(np.isnan(plot_data)):
                    plt.close(fig)
                    return None  # Nothing to plot yet

                # Use only valid data for color scaling
                valid_min = np.nanmin(plot_data)
                valid_max = np.nanmax(plot_data)

                im = ax.imshow(plot_data, origin='lower', cmap='plasma', aspect='auto',
                               vmin=valid_min, vmax=valid_max)
                ax.set_title('Sky Map (dB above baseline)', fontsize=14)
                ax.set_xlabel('Azimuth Steps', fontsize=12)
                ax.set_ylabel('Elevation Steps', fontsize=12)
                plt.colorbar(im, ax=ax, label='Signal Strength (dB)')

            # # Add detection markers
            # if self.baseline_std > 0:
            #     y_coords, x_coords = np.where(self.sky_map > 3 * self.baseline_std)
            #     ax.scatter(x_coords, y_coords, c='red', s=30, alpha=0.7, marker='o')

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
            filename_prefix = datetime.now().strftime("scan_%Y%m%d_%H")

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
