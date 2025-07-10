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

matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from datetime import datetime
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import threading
from collections import deque
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import json
import select
import termios
import tty
import atexit

# Try to import RTL-SDR
try:
    from rtlsdr import RtlSdr

    RTL_SDR_AVAILABLE = True
except ImportError:
    RTL_SDR_AVAILABLE = False
    print("RTL-SDR not available, using simulation mode")


class NonBlockingInput:
    """Non-blocking keyboard input that doesn't interfere with console output"""

    def __init__(self):
        self.old_settings = None
        self.setup_terminal()

    def setup_terminal(self):
        """Setup terminal for non-blocking input"""
        if sys.stdin.isatty():
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            atexit.register(self.restore_terminal)

    def restore_terminal(self):
        """Restore terminal settings"""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_char(self):
        """Get a single character without blocking"""
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None

    def get_arrow_key(self):
        """Get arrow key input"""
        char = self.get_char()
        if char == '\x1b':  # ESC sequence
            char = self.get_char()
            if char == '[':
                char = self.get_char()
                if char == 'A':
                    return 'up'
                elif char == 'B':
                    return 'down'
                elif char == 'C':
                    return 'right'
                elif char == 'D':
                    return 'left'
        return char


class HydrogenScanner:
    def __init__(self):
        self.kit = MotorKit(i2c=board.I2C())
        self.input_handler = NonBlockingInput()

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

        # Web dashboard data
        self.web_data = {
            'status': 'idle',
            'progress': 0,
            'current_measurement': {'x': 0, 'y': 0, 'power': 0},
            'stats': {
                'baseline_level': 0,
                'baseline_std': 0,
                'max_signal': 0,
                'total_points': 0,
                'detections_3sigma': 0,
                'detections_5sigma': 0
            },
            'plots': {}
        }
        self.data_lock = threading.Lock()

        # Flask app with SocketIO for real-time updates
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

    def emit_web_update(self):
        """Emit real-time update to web dashboard"""
        with self.data_lock:
            self.socketio.emit('data_update', self.web_data)

    def measure_power(self, measurement_time=1.0):
        """Measure power with RTL-SDR or simulation"""
        if not RTL_SDR_AVAILABLE:
            # Realistic simulation with occasional H-line detections
            base_noise = np.random.normal(-45, 2)
            if np.random.random() < 0.15:  # 15% chance of H-line detection
                base_noise += np.random.normal(2, 0.3)
            return base_noise

        try:
            sdr = RtlSdr()
            sdr.sample_rate = self.config['sample_rate']
            sdr.center_freq = self.config['center_freq']
            sdr.gain = self.config['gain']

            total_samples = int(measurement_time * sdr.sample_rate)
            chunk_size = self.config['samples_per_measurement']

            power_sum = 0
            samples_collected = 0

            while samples_collected < total_samples:
                remaining = total_samples - samples_collected
                chunk = sdr.read_samples(min(chunk_size, remaining))
                power_sum += np.sum(np.abs(chunk) ** 2)
                samples_collected += len(chunk)

            power_db = 10 * np.log10(power_sum / samples_collected)
            sdr.close()
            return power_db

        except Exception as e:
            print(f"SDR error: {e}")
            return np.random.normal(-45, 2)

    def move_motor(self, axis, direction, steps=1):
        """Move motor and update position tracking"""
        motor = self.kit.stepper1 if axis == 'x' else self.kit.stepper2

        for _ in range(steps):
            motor.onestep(direction=direction, style=stepper.INTERLEAVE)
            time.sleep(0.01)  # Small delay for smooth movement

        # Update position tracking
        if axis == 'x':
            self.current_position['x'] += steps if direction == stepper.FORWARD else -steps
        else:
            self.current_position['y'] += steps if direction == stepper.FORWARD else -steps

    def manual_position(self):
        """Manual positioning with improved keyboard handling"""
        print("\n" + "=" * 60)
        print("MANUAL POSITIONING MODE")
        print("=" * 60)
        print("Controls:")
        print("  Arrow keys: Move antenna")
        print("  'f': Toggle fast mode (5x steps)")
        print("  'q': Quit positioning mode")
        print("  'r': Reset position counter")
        print("  'p': Print current position")
        print("=" * 60)

        fast_mode = False
        last_move_time = 0
        move_delay = 0.1

        print(f"Current position: ({self.current_position['x']}, {self.current_position['y']})")
        print(f"Mode: {'FAST' if fast_mode else 'NORMAL'}")
        print("Web dashboard: http://localhost:5000")
        print("\nReady for input...")

        while True:
            # Get keyboard input
            key = self.input_handler.get_arrow_key()
            current_time = time.time()

            if key == 'q':
                print("\nExiting positioning mode...")
                break
            elif key == 'f':
                fast_mode = not fast_mode
                print(f"Mode: {'FAST' if fast_mode else 'NORMAL'}")
            elif key == 'r':
                self.current_position = {'x': 0, 'y': 0}
                print(f"Position reset to (0, 0)")
            elif key == 'p':
                print(f"Current position: ({self.current_position['x']}, {self.current_position['y']})")

            # Handle movement with rate limiting
            if current_time - last_move_time > move_delay:
                steps = 5 if fast_mode else 1

                if key == 'up':
                    self.move_motor('y', stepper.BACKWARD, steps)
                    print(f"↑ Position: ({self.current_position['x']}, {self.current_position['y']})")
                    last_move_time = current_time
                elif key == 'down':
                    self.move_motor('y', stepper.FORWARD, steps)
                    print(f"↓ Position: ({self.current_position['x']}, {self.current_position['y']})")
                    last_move_time = current_time
                elif key == 'left':
                    self.move_motor('x', stepper.BACKWARD, steps)
                    print(f"← Position: ({self.current_position['x']}, {self.current_position['y']})")
                    last_move_time = current_time
                elif key == 'right':
                    self.move_motor('x', stepper.FORWARD, steps)
                    print(f"→ Position: ({self.current_position['x']}, {self.current_position['y']})")
                    last_move_time = current_time

            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)

    def calibrate_baseline(self, num_samples=30, measurement_time=1.0):
        """Baseline calibration with real-time updates"""
        print(f"\n" + "=" * 60)
        print("BASELINE CALIBRATION")
        print("=" * 60)
        print(f"Samples: {num_samples}")
        print(f"Measurement time: {measurement_time}s per sample")
        print("Point antenna away from galactic plane.")
        print("Press Enter to start calibration...")
        input()

        with self.data_lock:
            self.web_data['status'] = 'calibrating'
            self.web_data['progress'] = 0
        self.emit_web_update()

        baseline_data = []
        start_time = time.time()

        for i in range(num_samples):
            power = self.measure_power(measurement_time)
            baseline_data.append(power)

            progress = (i + 1) / num_samples * 100
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (num_samples - i - 1)

            print(f"Sample {i + 1:2d}/{num_samples}: {power:6.2f} dB "
                  f"(Progress: {progress:5.1f}%, ETA: {eta:4.0f}s)")

            # Update web dashboard
            with self.data_lock:
                self.web_data['progress'] = progress
                self.web_data['current_measurement']['power'] = power
            self.emit_web_update()

        self.baseline_level = np.mean(baseline_data)
        self.baseline_std = np.std(baseline_data)

        print(f"\nBaseline calibration complete:")
        print(f"  Level: {self.baseline_level:.2f} dB")
        print(f"  Std Dev: {self.baseline_std:.2f} dB")
        print(f"  3σ threshold: {3 * self.baseline_std:.2f} dB")
        print(f"  5σ threshold: {5 * self.baseline_std:.2f} dB")

        with self.data_lock:
            self.web_data['status'] = 'baseline_complete'
            self.web_data['progress'] = 100
            self.web_data['stats']['baseline_level'] = self.baseline_level
            self.web_data['stats']['baseline_std'] = self.baseline_std
        self.emit_web_update()

        return baseline_data

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

    def update_web_data(self, x, y, calibrated_power, progress):
        """Update web dashboard data with real-time emission"""
        with self.data_lock:
            self.web_data['current_measurement'] = {
                'x': int(x), 'y': int(y), 'power': float(calibrated_power)
            }
            self.web_data['progress'] = float(progress)

            # Update stats
            if self.sky_map is not None:
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
            if len(self.scan_data) % 5 == 0:  # More frequent updates
                plots = {}
                for plot_type in ['skymap', 'recent', 'distribution']:
                    plot_img = self.create_plot_image(plot_type)
                    if plot_img:
                        plots[plot_type] = plot_img
                self.web_data['plots'] = plots

        # Emit real-time update
        self.emit_web_update()

    def run_scan(self, x_steps=20, y_steps=20, steps_per_point=2, measurement_time=1.0):
        """Run scan with improved progress tracking and real-time updates"""
        if self.baseline_level == 0:
            print("ERROR: Run baseline calibration first!")
            return None

        print(f"\n" + "=" * 60)
        print("SCANNING MODE")
        print("=" * 60)
        print(f"Grid size: {x_steps} × {y_steps} = {x_steps * y_steps} points")
        print(f"Measurement time: {measurement_time}s per point")
        print(f"Estimated time: {(x_steps * y_steps * measurement_time / 60):.1f} minutes")
        print(f"Web dashboard: http://localhost:5000")
        print("=" * 60)

        with self.data_lock:
            self.web_data['status'] = 'scanning'
            self.web_data['progress'] = 0
        self.emit_web_update()

        # Initialize
        self.sky_map = np.zeros((y_steps, x_steps))
        self.scan_data = []
        total_points = x_steps * y_steps
        start_time = time.time()

        try:
            for y in range(y_steps):
                row_start_time = time.time()
                print(f"\nRow {y + 1}/{y_steps}:")

                for x in range(x_steps):
                    # Measure power
                    raw_power = self.measure_power(measurement_time)
                    calibrated_power = raw_power - self.baseline_level

                    # Store data
                    self.scan_data.append({
                        'x': x, 'y': y,
                        'raw_power': raw_power,
                        'calibrated_power': calibrated_power,
                        'timestamp': time.time()
                    })

                    # Update arrays
                    self.sky_map[y, x] = calibrated_power
                    self.live_data.append(calibrated_power)

                    # Progress calculation
                    points_completed = len(self.scan_data)
                    progress = (points_completed / total_points) * 100

                    # Time estimates
                    elapsed = time.time() - start_time
                    eta = (elapsed / points_completed) * (total_points - points_completed)

                    # Console output
                    detection_flag = ""
                    if calibrated_power > 5 * self.baseline_std:
                        detection_flag = " ⭐ 5σ!"
                    elif calibrated_power > 3 * self.baseline_std:
                        detection_flag = " ⚡ 3σ"

                    print(f"  ({x:2d},{y:2d}): {calibrated_power:6.2f} dB{detection_flag}")

                    # Update web dashboard
                    self.update_web_data(x, y, calibrated_power, progress)

                    # Move to next position
                    if x < x_steps - 1:
                        self.move_motor('x', stepper.FORWARD, steps_per_point)

                # Row completion info
                row_time = time.time() - row_start_time
                print(f"  Row {y + 1} completed in {row_time:.1f}s (ETA: {eta / 60:.1f}m)")

                # Move to next row
                if y < y_steps - 1:
                    self.move_motor('x', stepper.BACKWARD, (x_steps - 1) * steps_per_point)
                    self.move_motor('y', stepper.FORWARD, steps_per_point)

            # Scan completion
            total_time = time.time() - start_time
            print(f"\n" + "=" * 60)
            print("SCAN COMPLETE!")
            print(f"Total time: {total_time / 60:.1f} minutes")
            print(f"Points scanned: {len(self.scan_data)}")

            # Final statistics
            detections_3sigma = np.sum(self.sky_map > 3 * self.baseline_std)
            detections_5sigma = np.sum(self.sky_map > 5 * self.baseline_std)
            max_signal = np.max(self.sky_map)

            print(f"3σ detections: {detections_3sigma}")
            print(f"5σ detections: {detections_5sigma}")
            print(f"Max signal: {max_signal:.2f} dB")
            print("=" * 60)

            with self.data_lock:
                self.web_data['status'] = 'complete'
                self.web_data['progress'] = 100
                # Final plot update
                plots = {}
                for plot_type in ['skymap', 'recent', 'distribution']:
                    plot_img = self.create_plot_image(plot_type)
                    if plot_img:
                        plots[plot_type] = plot_img
                self.web_data['plots'] = plots
            self.emit_web_update()

            return self.sky_map

        except KeyboardInterrupt:
            print("\n\nScan interrupted by user!")
            with self.data_lock:
                self.web_data['status'] = 'interrupted'
            self.emit_web_update()
            return self.sky_map

    def start_web_server(self):
        """Start the web server with SocketIO in a separate thread"""

        def run_server():
            self.socketio.run(self.app, host='0.0.0.0', port=5000, debug=False)

        web_thread = threading.Thread(target=run_server, daemon=True)
        web_thread.start()
        print("Web dashboard with real-time updates started at http://localhost:5000")
        time.sleep(1)  # Give server time to start
        return web_thread

    def save_data(self, filename_prefix=None):
        """Save scan data with comprehensive metadata"""
        if not filename_prefix:
            filename_prefix = datetime.now().strftime("scan_%Y%m%d_%H%M%S")

        saved_files = []

        # Save sky map
        if self.sky_map is not None:
            skymap_file = f"{filename_prefix}_skymap.npy"
            np.save(skymap_file, self.sky_map)
            saved_files.append(skymap_file)

        # Save scan data
        if self.scan_data:
            scan_array = np.array([(d['x'], d['y'], d['raw_power'], d['calibrated_power'], d['timestamp'])
                                   for d in self.scan_data])
            data_file = f"{filename_prefix}_data.npy"
            np.save(data_file, scan_array)
            saved_files.append(data_file)

        # Save comprehensive metadata
        metadata = {
            'baseline_level': self.baseline_level,
            'baseline_std': self.baseline_std,
            'config': self.config,
            'scan_stats': {
                'total_points': len(self.scan_data),
                'detections_3sigma': int(
                    np.sum(self.sky_map > 3 * self.baseline_std)) if self.sky_map is not None else 0,
                'detections_5sigma': int(
                    np.sum(self.sky_map > 5 * self.baseline_std)) if self.sky_map is not None else 0,
                'max_signal': float(np.max(self.sky_map)) if self.sky_map is not None else 0,
            },
            'timestamp': datetime.now().isoformat(),
            'rtl_sdr_available': RTL_SDR_AVAILABLE
        }
        metadata_file = f"{filename_prefix}_metadata.npy"
        np.save(metadata_file, metadata)
        saved_files.append(metadata_file)

        print(f"\nData saved:")
        for file in saved_files:
            print(f"  {file}")

        return saved_files


def main():
    """Main execution function"""
    # Configuration
    config = {
        'x_steps': 15,
        'y_steps': 15,
        'steps_per_point': 2,
        'measurement_time': 1.5
    }

    print("Hydrogen Line Scanner v3.0")
    print("=" * 60)
    print("Features:")
    print("  • Non-blocking keyboard input")
    print("  • Real-time web dashboard with WebSocket updates")
    print("  • Improved progress tracking")
    print("  • Enhanced data visualization")
    print("=" * 60)

    scanner = HydrogenScanner()

    try:
        # Start web server
        scanner.start_web_server()

        print(f"\nScan Configuration:")
        print(f"  Grid: {config['x_steps']}×{config['y_steps']} points")
        print(f"  Frequency: {scanner.config['center_freq'] / 1e9:.6f} GHz")
        print(f"  Measurement time: {config['measurement_time']}s per point")
        print(
            f"  Estimated scan time: {(config['x_steps'] * config['y_steps'] * config['measurement_time'] / 60):.1f} minutes")
        print(f"  Web dashboard: http://localhost:5000")

        # Step 1: Position for baseline
        print("\n" + "=" * 60)
        print("STEP 1: Position antenna for baseline calibration")
        scanner.manual_position()

        # Step 2: Calibrate baseline
        print("\n" + "=" * 60)
        print("STEP 2: Baseline calibration")
        scanner.calibrate_baseline(num_samples=20, measurement_time=config['measurement_time'])

        # Step 3: Position for scan
        print("\n" + "=" * 60)
        print("STEP 3: Position antenna for scan start")
        scanner.manual_position()

        # Step 4: Run scan
        print("\n" + "=" * 60)
        print("STEP 4: Execute scan")
        input("Press Enter to start scan...")

        sky_map = scanner.run_scan(
            x_steps=config['x_steps'],
            y_steps=config['y_steps'],
            steps_per_point=config['steps_per_point'],
            measurement_time=config['measurement_time']
        )

        # Step 5: Save data
        if sky_map is not None:
            scanner.save_data()
            print("\nScan complete! Web dashboard will remain active.")
            print("Press Ctrl+C to exit.")

            # Keep web server running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
        else:
            print("No data to save.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        scanner.input_handler.restore_terminal()


if __name__ == "__main__":
    main()
    