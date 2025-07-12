import time
from src.hydrogen_scanner import HydrogenScanner


def run_headless_scan():
    print("Starting headless Hydrogen Line Scan (no web)...")

    scanner = HydrogenScanner()

    print("Calibrating baseline...")
    scanner.calibrating = True
    scanner.calibrate_baseline_web(num_samples=10, measurement_time=0.5)
    if not scanner.web_data['baseline_calibrated']:
        print("Calibration failed or interrupted.")
        return

    print("Baseline calibrated:")
    print(f"  Mean: {scanner.baseline_level:.2f} dB")
    print(f"  Std : {scanner.baseline_std:.2f} dB")

    print("Starting scan...")
    scanner.run_scan_web(x_steps=5, y_steps=5, measurement_time=0.3, x_step_size=1, y_step_size=1)

    print("Scan complete.")
    print(f"Collected {len(scanner.scan_data)} data points.")

    # Save data
    saved_files = scanner.save_data()
    print("Saved output files:")
    for file in saved_files:
        print(f" - {file}")

    # Save individual plots as images
    print("Generating plots...")
    for plot_type in ['skymap', 'hydrogen_line', 'distribution', 'spectrum']:
        img_b64 = scanner.create_plot_image(plot_type)
        if img_b64:
            with open(f"{plot_type}.png", "wb") as f:
                f.write(base64.b64decode(img_b64.split(",")[1]))
            print(f"Saved {plot_type}.png")
        else:
            print(f"Could not generate {plot_type} plot.")

if __name__ == "__main__":
    run_headless_scan()
