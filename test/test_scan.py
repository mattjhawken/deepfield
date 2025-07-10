"""
Hydrogen line manual testing and background scanning with web dashboard for remote access
"""
import curses
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
            'samples_per_measurement': 256*1024
        }

        # Data storage
        self.baseline_level = 0.0
        self.baseline_std = 0.0
        self.scan_data = []
        self.sky_map = None
        self.live_data = deque(maxlen=100)

        # Web dashboard data
        self.web_data = {
            'status': 'idle',
            'progress': 0,
            'current_measurement': {'x': 0, 'y': 0, 'power': 0},
            'stats': {},
            'plots': {}
        }
        self.data_lock = threading.Lock()

        # Flask app
        self.app = Flask(__name__)
        self.setup_web_routes()

    def setup_web_routes(self):
        """Setup Flask routes for web dashboard"""
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/api/data')
        def get_data():
            with self.data_lock:
                return jsonify(self.web_data)

        @self.app.route('/api/control/<action>')
        def control(action):
            # Add control endpoints if needed
            return jsonify({'status': 'ok', 'action': action})

    def measure_power(self, measurement_time=1.0):
        """Measure power with RTL-SDR or simulation"""
        if not RTL_SDR_AVAILABLE:
            # Realistic simulation
            samples = np.random.normal(-45, 2, int(measurement_time * 1000))
            if np.random.random() < 0.15:  # 15% chance of H-line detection
                samples += np.random.normal(2, 0.3)
            return np.mean(samples)

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
                chunk = sdr.read_samples(min(chunk_size, total_samples - samples_collected))
                power_sum += np.sum(np.abs(chunk)**2)
                samples_collected += len(chunk)

            power_db = 10 * np.log10(power_sum / samples_collected)
            sdr.close()
            return power_db

        except Exception as e:
            print(f"SDR error: {e}")
            return np.random.normal(-45, 2)

    def move_motor(self, axis, direction, steps=1):
        """Move motor with simplified interface"""
        motor = self.kit.stepper1 if axis == 'x' else self.kit.stepper2
        for _ in range(steps):
            motor.onestep(direction=direction, style=stepper.INTERLEAVE)

    def manual_position(self):
        """Manual positioning with curses interface"""
        def _position(stdscr):
            stdscr.nodelay(True)
            stdscr.clear()
            stdscr.addstr("Arrow keys to move, 'q' to quit, 'f' for fast mode\n")
            stdscr.addstr("Current mode: NORMAL\n")
            stdscr.addstr("Web dashboard: http://localhost:5000\n")
            stdscr.refresh()

            fast_mode = False
            last_step = time.time()

            while True:
                key = stdscr.getch()
                now = time.time()

                if key == ord('q'):
                    break
                elif key == ord('f'):
                    fast_mode = not fast_mode
                    stdscr.clear()
                    stdscr.addstr("Arrow keys to move, 'q' to quit, 'f' for fast mode\n")
                    stdscr.addstr(f"Current mode: {'FAST' if fast_mode else 'NORMAL'}\n")
                    stdscr.addstr("Web dashboard: http://localhost:5000\n")
                    stdscr.refresh()

                steps = 5 if fast_mode else 1
                delay = 0.05 if fast_mode else 0.1

                if now - last_step > delay:
                    if key == curses.KEY_UP:
                        self.move_motor('y', stepper.BACKWARD, steps)
                    elif key == curses.KEY_DOWN:
                        self.move_motor('y', stepper.FORWARD, steps)
                    elif key == curses.KEY_LEFT:
                        self.move_motor('x', stepper.BACKWARD, steps)
                    elif key == curses.KEY_RIGHT:
                        self.move_motor('x', stepper.FORWARD, steps)
                    last_step = now

                time.sleep(0.01)

        curses.wrapper(_position)

    def calibrate_baseline(self, num_samples=30, measurement_time=1.0):
        """Simplified baseline calibration"""
        print(f"\nCalibrating baseline...")
        print(f"Taking {num_samples} samples ({measurement_time}s each)")
        print("Point antenna away from galactic plane. Press Enter to start...")
        input()

        with self.data_lock:
            self.web_data['status'] = 'calibrating'

        baseline_data = []
        for i in range(num_samples):
            power = self.measure_power(measurement_time)
            baseline_data.append(power)
            print(f"Sample {i+1}/{num_samples}: {power:.2f} dB")

            with self.data_lock:
                self.web_data['progress'] = (i + 1) / num_samples * 100

        self.baseline_level = np.mean(baseline_data)
        self.baseline_std = np.std(baseline_data)

        print(f"\nBaseline: {self.baseline_level:.2f} ± {self.baseline_std:.2f} dB")

        with self.data_lock:
            self.web_data['status'] = 'baseline_complete'
            self.web_data['stats']['baseline_level'] = self.baseline_level
            self.web_data['stats']['baseline_std'] = self.baseline_std

        return baseline_data

    def create_plot_image(self, plot_type='skymap'):
        """Create plot and return as base64 encoded image"""
        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type == 'skymap' and self.sky_map is not None:
            im = ax.imshow(self.sky_map, origin='lower', cmap='plasma')
            ax.set_title('Sky Map (dB above baseline)')
            ax.set_xlabel('Azimuth')
            ax.set_ylabel('Elevation')
            plt.colorbar(im, ax=ax)

        elif plot_type == 'recent' and len(self.live_data) > 0:
            ax.plot(list(self.live_data), 'b-', alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            if self.baseline_std > 0:
                ax.axhline(y=3*self.baseline_std, color='orange', linestyle=':', alpha=0.7)
            ax.set_title('Recent Measurements')
            ax.set_xlabel('Measurement Number')
            ax.set_ylabel('Power - Baseline (dB)')
            ax.grid(True, alpha=0.3)

        elif plot_type == 'distribution' and len(self.scan_data) > 5:
            all_powers = [d['calibrated_power'] for d in self.scan_data]
            ax.hist(all_powers, bins=20, alpha=0.7, color='skyblue')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Signal Distribution')
            ax.set_xlabel('Power - Baseline (dB)')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)

        return f"data:image/png;base64,{img_str}"

    def update_web_data(self, x, y, calibrated_power, progress):
        """Update web dashboard data"""
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

            # Update plots (every 10 measurements to avoid overload)
            if len(self.scan_data) % 10 == 0:
                self.web_data['plots'] = {
                    'skymap': self.create_plot_image('skymap'),
                    'recent': self.create_plot_image('recent'),
                    'distribution': self.create_plot_image('distribution')
                }

    def run_scan(self, x_steps=20, y_steps=20, steps_per_point=2, measurement_time=1.0):
        """Run the main scan with web dashboard updates"""
        if self.baseline_level == 0:
            print("ERROR: Run baseline calibration first!")
            return None

        print(f"\nStarting {x_steps}x{y_steps} scan...")
        print(f"Web dashboard: http://localhost:5000")
        print(f"Estimated time: {(x_steps * y_steps * measurement_time / 60):.1f} minutes")

        with self.data_lock:
            self.web_data['status'] = 'scanning'

        # Initialize sky map
        self.sky_map = np.zeros((y_steps, x_steps))
        self.scan_data = []
        total_points = x_steps * y_steps
        start_time = time.time()

        try:
            for y in range(y_steps):
                print(f"\nRow {y+1}/{y_steps}")

                for x in range(x_steps):
                    # Measure
                    raw_power = self.measure_power(measurement_time)
                    calibrated_power = raw_power - self.baseline_level

                    # Store data
                    self.scan_data.append({
                        'x': x, 'y': y,
                        'raw_power': raw_power,
                        'calibrated_power': calibrated_power,
                        'timestamp': time.time()
                    })

                    # Update sky map and live data
                    self.sky_map[y, x] = calibrated_power
                    self.live_data.append(calibrated_power)

                    # Update web dashboard
                    progress = (len(self.scan_data) / total_points) * 100
                    self.update_web_data(x, y, calibrated_power, progress)

                    print(f"  ({x}, {y}): {calibrated_power:.2f} dB")

                    # Move to next position
                    if x < x_steps - 1:
                        self.move_motor('x', stepper.FORWARD, steps_per_point)

                # Move to next row
                if y < y_steps - 1:
                    # Return to start of row
                    self.move_motor('x', stepper.BACKWARD, (x_steps - 1) * steps_per_point)
                    # Move up one row
                    self.move_motor('y', stepper.FORWARD, steps_per_point)

            total_time = time.time() - start_time
            print(f"\nScan complete! Time: {total_time/60:.1f} minutes")

            with self.data_lock:
                self.web_data['status'] = 'complete'
                # Final plot update
                self.web_data['plots'] = {
                    'skymap': self.create_plot_image('skymap'),
                    'recent': self.create_plot_image('recent'),
                    'distribution': self.create_plot_image('distribution')
                }

            return self.sky_map

        except KeyboardInterrupt:
            print("\nScan interrupted!")
            with self.data_lock:
                self.web_data['status'] = 'interrupted'
            return self.sky_map

    def start_web_server(self):
        """Start the web server in a separate thread"""
        def run_server():
            self.app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

        web_thread = threading.Thread(target=run_server, daemon=True)
        web_thread.start()
        print("Web dashboard started at http://localhost:5000")
        return web_thread

    def save_data(self, filename_prefix=None):
        """Save scan data"""
        if not filename_prefix:
            filename_prefix = datetime.now().strftime("scan_%Y%m%d_%H%M%S")

        if self.sky_map is not None:
            np.save(f"{filename_prefix}_skymap.npy", self.sky_map)

        if self.scan_data:
            scan_array = np.array([(d['x'], d['y'], d['raw_power'], d['calibrated_power'])
                                  for d in self.scan_data])
            np.save(f"{filename_prefix}_data.npy", scan_array)

        metadata = {
            'baseline_level': self.baseline_level,
            'baseline_std': self.baseline_std,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        np.save(f"{filename_prefix}_metadata.npy", metadata)

        print(f"Data saved with prefix: {filename_prefix}")


def main():
    # Configuration
    config = {
        'x_steps': 15,
        'y_steps': 15,
        'steps_per_point': 2,
        'measurement_time': 1.5  # seconds per measurement
    }

    scanner = HydrogenScanner()

    # Start web server
    scanner.start_web_server()

    print("Hydrogen Line Scanner v2.0 - Web Dashboard")
    print("=" * 50)
    print(f"Grid: {config['x_steps']}x{config['y_steps']}")
    print(f"Frequency: {scanner.config['center_freq']/1e9:.6f} GHz")
    print(f"Estimated time: {(config['x_steps'] * config['y_steps'] * config['measurement_time'] / 60):.1f} minutes")
    print("Web dashboard: http://localhost:5000")
    print("=" * 50)

    # Step 1: Position for baseline
    print("\n1. Position antenna for baseline calibration")
    scanner.manual_position()

    # Step 2: Calibrate baseline
    print("\n2. Baseline calibration")
    scanner.calibrate_baseline(num_samples=20, measurement_time=config['measurement_time'])

    # Step 3: Position for scan
    print("\n3. Position for scan start")
    scanner.manual_position()

    # Step 4: Run scan
    print("\n4. Running scan with web dashboard")
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
        print("Scan complete! Web dashboard will remain active.")
        print("Press Ctrl+C to exit.")

        # Keep web server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
    else:
        print("No data to save.")


if __name__ == "__main__":
    main()