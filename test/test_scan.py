"""
Hydrogen line manual testing and background scanning
"""
import curses
import time
import board
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper

# Try to import RTL-SDR - if not available, use simulation
try:
    from rtlsdr import RtlSdr
    RTL_SDR_AVAILABLE = True
except ImportError:
    RTL_SDR_AVAILABLE = False
    print("RTL-SDR not available, using simulation mode")

kit = MotorKit(i2c=board.I2C())

FORWARD = stepper.FORWARD
BACKWARD = stepper.BACKWARD

# Global variables for data storage
baseline_data = []
scan_data = []
baseline_level = 0.0
baseline_std = 0.0

# SDR Configuration
SDR_CONFIG = {
    'sample_rate': 2.048e6,
    'center_freq': 1.42040575e9,  # Hydrogen line frequency
    'gain': 'auto',
    'samples_per_measurement': 256*1024
}

def measure_power_rtlsdr(measurement_time=1.0):
    """
    Measure power using RTL-SDR over a specified time window

    Args:
        measurement_time: Time in seconds to collect samples
    """
    if not RTL_SDR_AVAILABLE:
        # Simulate realistic measurements with more samples = better SNR
        num_sim_samples = int(measurement_time * 1000)  # Simulate 1000 samples per second
        samples = np.random.normal(-45, 2, num_sim_samples)  # Base noise level

        # Add occasional hydrogen line signals
        if np.random.random() < 0.1:  # 10% chance of detecting signal
            signal_strength = np.random.normal(3, 0.5)  # 3dB above noise
            samples += signal_strength * 0.1  # Add weak signal

        # Better SNR with more samples
        power = np.mean(samples) - np.std(samples) / np.sqrt(len(samples))
        return power

    try:
        sdr = RtlSdr()
        sdr.sample_rate = SDR_CONFIG['sample_rate']
        sdr.center_freq = SDR_CONFIG['center_freq']
        sdr.gain = SDR_CONFIG['gain']

        # Calculate how many samples to collect
        total_samples = int(measurement_time * sdr.sample_rate)
        samples_per_read = SDR_CONFIG['samples_per_measurement']

        all_samples = []
        samples_collected = 0

        while samples_collected < total_samples:
            # Read samples in chunks
            chunk_size = min(samples_per_read, total_samples - samples_collected)
            chunk = sdr.read_samples(chunk_size)
            all_samples.extend(chunk)
            samples_collected += len(chunk)

        # Calculate power in dB from all collected samples
        power = 10 * np.log10(np.mean(np.abs(all_samples)**2))

        sdr.close()
        return power

    except Exception as e:
        print(f"SDR measurement error: {e}")
        # Fallback to simulation
        num_sim_samples = int(measurement_time * 1000)
        samples = np.random.normal(-45, 2, num_sim_samples)
        return np.mean(samples) - np.std(samples) / np.sqrt(len(samples))


def move_x(direction):
    kit.stepper1.onestep(direction=direction)
    time.sleep(0.01)


def move_y(direction):
    kit.stepper2.onestep(direction=direction)
    time.sleep(0.05)


def position_motor(stdscr, additional_text):
    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.addstr(f"{additional_text}Use arrow keys to move motors. Press 'q' to quit.\n")
    stdscr.addstr("Current position will be used for measurements.\n")
    stdscr.refresh()

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP:
            move_y(BACKWARD)
        elif key == curses.KEY_DOWN:
            move_y(FORWARD)
        elif key == curses.KEY_LEFT:
            move_x(BACKWARD)
        elif key == curses.KEY_RIGHT:
            move_x(FORWARD)
        elif key == ord("q"):
            break
        time.sleep(0.001)


def calibrate_baseline(num_samples=50, sample_interval=0.1, measurement_time=1.0):
    """
    Perform baseline calibration by taking multiple measurements

    Args:
        num_samples: Number of baseline measurements to take
        sample_interval: Time between measurements (seconds)
        measurement_time: Time to spend collecting samples for each measurement
    """
    global baseline_data, baseline_level, baseline_std

    print(f"\nStarting baseline calibration...")
    print(f"Taking {num_samples} measurements")
    print(f"Each measurement collects data for {measurement_time}s")
    print(f"Total calibration time: ~{(num_samples * (measurement_time + sample_interval)):.1f}s")
    print("Point antenna to cold sky region (away from galactic plane)")
    print("Press Enter to start calibration...")
    input()

    baseline_data = []
    timestamps = []

    print("Calibrating", end="", flush=True)

    for i in range(num_samples):
        timestamp = time.time()
        power = measure_power_rtlsdr(measurement_time)

        baseline_data.append(power)
        timestamps.append(timestamp)

        print(".", end="", flush=True)

        # Short pause between measurements
        if i < num_samples - 1:  # Don't pause after last measurement
            time.sleep(sample_interval)

    print(" Done!")

    # Calculate statistics
    baseline_level = np.mean(baseline_data)
    baseline_std = np.std(baseline_data)

    print(f"\nBaseline Calibration Results:")
    print(f"  Mean power: {baseline_level:.2f} dB")
    print(f"  Std deviation: {baseline_std:.2f} dB")
    print(f"  Min power: {np.min(baseline_data):.2f} dB")
    print(f"  Max power: {np.max(baseline_data):.2f} dB")
    print(f"  Measurement time per sample: {measurement_time}s")

    # Plot baseline data
    plot_baseline_data()

    return baseline_level, baseline_std


def plot_baseline_data():
    """
    Plot the baseline calibration data
    """
    if not baseline_data:
        print("No baseline data to plot")
        return

    plt.figure(figsize=(12, 8))

    # Time series plot
    plt.subplot(2, 2, 1)
    plt.plot(baseline_data, 'b-', alpha=0.7)
    plt.axhline(y=baseline_level, color='r', linestyle='--', label=f'Mean: {baseline_level:.2f} dB')
    plt.axhline(y=baseline_level + baseline_std, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {baseline_level + baseline_std:.2f} dB')
    plt.axhline(y=baseline_level - baseline_std, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {baseline_level - baseline_std:.2f} dB')
    plt.xlabel('Sample Number')
    plt.ylabel('Power (dB)')
    plt.title('Baseline Calibration Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Histogram
    plt.subplot(2, 2, 2)
    plt.hist(baseline_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=baseline_level, color='r', linestyle='--', label=f'Mean: {baseline_level:.2f} dB')
    plt.xlabel('Power (dB)')
    plt.ylabel('Frequency')
    plt.title('Baseline Power Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Statistics text
    plt.subplot(2, 2, 3)
    plt.axis('off')
    stats_text = f"""
    Baseline Calibration Statistics:
    
    Mean Power: {baseline_level:.3f} dB
    Std Deviation: {baseline_std:.3f} dB
    Min Power: {np.min(baseline_data):.3f} dB
    Max Power: {np.max(baseline_data):.3f} dB
    Range: {np.max(baseline_data) - np.min(baseline_data):.3f} dB
    
    Samples: {len(baseline_data)}
    Frequency: {SDR_CONFIG['center_freq']/1e9:.6f} GHz
    Sample Rate: {SDR_CONFIG['sample_rate']/1e6:.3f} MHz
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)

    # Running average
    plt.subplot(2, 2, 4)
    running_avg = np.cumsum(baseline_data) / np.arange(1, len(baseline_data) + 1)
    plt.plot(running_avg, 'g-', linewidth=2, label='Running Average')
    plt.axhline(y=baseline_level, color='r', linestyle='--', label=f'Final Mean: {baseline_level:.2f} dB')
    plt.xlabel('Sample Number')
    plt.ylabel('Running Average (dB)')
    plt.title('Baseline Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_calibration_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Baseline calibration plot saved as: {filename}")

    plt.show()


def run_scan(x_steps, y_steps, steps_per_point, step_delay, measurement_time):
    """
    Run the sky scan with baseline subtraction

    Args:
        x_steps: Number of steps in X direction
        y_steps: Number of steps in Y direction
        steps_per_point: Motor steps between measurement points
        step_delay: Delay between motor steps
        measurement_time: Time to collect samples at each point
    """
    global scan_data

    if baseline_level == 0:
        print("ERROR: No baseline calibration performed!")
        return None

    print(f"\nStarting sky scan...")
    print(f"Baseline level: {baseline_level:.2f} ± {baseline_std:.2f} dB")
    print(f"Measurement time per point: {measurement_time}s")
    print(f"Estimated scan time: {(x_steps * y_steps * measurement_time / 60):.1f} minutes")

    # Initialize scan data array
    sky_map = np.zeros((y_steps, x_steps))
    scan_data = []

    try:
        total_points = x_steps * y_steps
        current_point = 0
        start_time = time.time()

        for y in range(y_steps):
            print(f"\nScanning row {y+1}/{y_steps}")

            for x in range(x_steps):
                # Take measurement over specified time window
                print(f"  Point ({x+1},{y+1}) - measuring for {measurement_time}s...", end="", flush=True)
                raw_power = measure_power_rtlsdr(measurement_time)
                calibrated_power = raw_power - baseline_level

                # Store data
                sky_map[y, x] = calibrated_power
                scan_data.append({
                    'x': x, 'y': y,
                    'raw_power': raw_power,
                    'calibrated_power': calibrated_power,
                    'timestamp': time.time(),
                    'measurement_time': measurement_time
                })

                current_point += 1
                progress = (current_point / total_points) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / current_point) * (total_points - current_point)

                print(f" = {raw_power:.2f} dB (raw) / {calibrated_power:.2f} dB (cal) [{progress:.1f}%] ETA: {eta/60:.1f}min")

                # Move to next position
                if x < x_steps - 1:
                    for _ in range(steps_per_point):
                        move_x(FORWARD)

            # Move to next row
            if y < y_steps - 1:
                print("  Moving to next row...")
                # Return to start of row
                for _ in range((x_steps - 1) * steps_per_point):
                    move_x(BACKWARD)

                # Move up one row
                for _ in range(steps_per_point):
                    move_y(FORWARD)

        total_time = time.time() - start_time
        print(f"\nScan complete! Total time: {total_time/60:.1f} minutes")
        return sky_map

    except KeyboardInterrupt:
        print("\nScan interrupted by user")
        return sky_map
    except Exception as e:
        print(f"\nError during scan: {e}")
        return sky_map

def plot_results(sky_map, x_steps, y_steps, steps_per_point):
    """
    Plot the scan results
    """
    if sky_map is None:
        print("No scan data to plot")
        return

    plt.figure(figsize=(15, 10))

    # Main sky map
    plt.subplot(2, 3, (1, 4))
    im = plt.imshow(sky_map, origin='lower', cmap='plasma',
                   extent=[0, x_steps, 0, y_steps], aspect='auto')
    plt.colorbar(im, label="Signal Above Background (dB)")
    plt.xlabel(f"Azimuth (points, {steps_per_point} steps each)")
    plt.ylabel(f"Elevation (points, {steps_per_point} steps each)")
    plt.title("1.42 GHz Hydrogen Line Sky Map")
    plt.grid(True, alpha=0.3)

    # Signal statistics
    plt.subplot(2, 3, 2)
    plt.hist(sky_map.flatten(), bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='Background Level')
    plt.axvline(x=np.mean(sky_map), color='orange', linestyle='--', label=f'Mean: {np.mean(sky_map):.2f} dB')
    plt.xlabel('Signal Above Background (dB)')
    plt.ylabel('Frequency')
    plt.title('Signal Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Scan statistics
    plt.subplot(2, 3, 3)
    plt.axis('off')
    stats_text = f"""
    Scan Statistics:
    
    Background Level: {baseline_level:.2f} ± {baseline_std:.2f} dB
    
    Max Signal: {np.max(sky_map):.2f} dB
    Min Signal: {np.min(sky_map):.2f} dB
    Mean Signal: {np.mean(sky_map):.2f} dB
    Std Deviation: {np.std(sky_map):.2f} dB
    
    Detections > 3σ: {np.sum(sky_map > 3 * baseline_std)}
    Detections > 5σ: {np.sum(sky_map > 5 * baseline_std)}
    
    Grid Size: {x_steps} × {y_steps}
    Total Points: {x_steps * y_steps}
    """
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10)

    # Signal vs position plots
    plt.subplot(2, 3, 5)
    x_profile = np.mean(sky_map, axis=0)
    plt.plot(x_profile, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Azimuth Position')
    plt.ylabel('Mean Signal (dB)')
    plt.title('Azimuth Profile')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    y_profile = np.mean(sky_map, axis=1)
    plt.plot(y_profile, 'g-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Elevation Position')
    plt.ylabel('Mean Signal (dB)')
    plt.title('Elevation Profile')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sky_map_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Sky map plot saved as: {filename}")

    plt.show()

def save_scan_data(sky_map, x_steps, y_steps, steps_per_point):
    """
    Save all scan data to files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save sky map array
    np.save(f"sky_map_{timestamp}.npy", sky_map)

    # Save detailed scan data
    scan_array = np.array([(d['x'], d['y'], d['raw_power'], d['calibrated_power'], d['timestamp'])
                          for d in scan_data])
    np.save(f"scan_data_{timestamp}.npy", scan_array)

    # Save baseline data
    baseline_array = np.array(baseline_data)
    np.save(f"baseline_data_{timestamp}.npy", baseline_array)

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'x_steps': x_steps,
        'y_steps': y_steps,
        'steps_per_point': steps_per_point,
        'baseline_level': baseline_level,
        'baseline_std': baseline_std,
        'sdr_config': SDR_CONFIG
    }
    np.save(f"metadata_{timestamp}.npy", metadata)

    print(f"All data saved with timestamp: {timestamp}")


if __name__ == "__main__":
    x_steps, y_steps = 20, 20  # Reduced for testing
    steps_per_point = 2
    step_delay = 0.01
    measurement_time = 2.0  # Time to collect samples at each point (seconds)

    print("Hydrogen Line Sky Scanner")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Grid size: {x_steps} x {y_steps}")
    print(f"  Steps per point: {steps_per_point}")
    print(f"  Step delay: {step_delay}s")
    print(f"  Measurement time per point: {measurement_time}s")
    print(f"  Total points: {x_steps * y_steps}")
    print(f"  Estimated scan time: {(x_steps * y_steps * measurement_time / 60):.1f} minutes")
    print(f"  Frequency: {SDR_CONFIG['center_freq']/1e9:.6f} GHz")

    # Step 1: Background calibration
    print("\n" + "=" * 50)
    print("STEP 1: BASELINE CALIBRATION")
    print("=" * 50)
    curses.wrapper(position_motor, "Position antenna for baseline calibration (cold sky region). ")
    calibrate_baseline(num_samples=30, sample_interval=0.1, measurement_time=measurement_time)

    # Step 2: Position for scan
    print("\n" + "=" * 50)
    print("STEP 2: POSITION FOR SCAN")
    print("=" * 50)
    curses.wrapper(position_motor, "Position antenna for scan start. ")

    # Step 3: Run scan
    print("\n" + "=" * 50)
    print("STEP 3: SKY SCAN")
    print("=" * 50)
    print("Press Enter to start scan...")
    input()

    sky_map = run_scan(x_steps, y_steps, steps_per_point, step_delay, measurement_time)

    # Step 4: Plot and save results
    print("\n" + "=" * 50)
    print("STEP 4: RESULTS")
    print("=" * 50)
    if sky_map is not None:
        plot_results(sky_map, x_steps, y_steps, steps_per_point)
        save_scan_data(sky_map, x_steps, y_steps, steps_per_point)
    else:
        print("No scan data to process")
