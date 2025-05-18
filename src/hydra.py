import time
import csv
from rtlsdr import RtlSdr
from src.positioning.planetary_drive import PlanetaryDrive

# SDR parameters
SAMPLE_RATE = 2.048e6
FREQ_MHZ = 1420.40575
GAIN = "auto"


def wait_for_settle():
    """
    Wait for motors to settle before recording.
    Adjust or extend this to check motor status if available.
    """
    time.sleep(2)


def record_samples(telescope_id, sdr, duration_sec, filename, az, el):
    print(f"Telescope {telescope_id} recording at Az: {az}°, El: {el} for {duration_sec}s")
    samples = sdr.read_samples(int(sdr.sample_rate * duration_sec))

    with open(filename, 'wb') as f:
        samples.tofile(f)

    meta_filename = filename.replace('.bin', '_meta.csv')
    with open(meta_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Telescope', 'Azimuth', 'Elevation', 'FrequencyMHz', 'SampleRate', 'DurationSec', 'Timestamp'])
        writer.writerow([telescope_id, az, el, sdr.center_freq / 1e6, sdr.sample_rate, duration_sec,
                         time.strftime("%Y-%m-%d %H:%M:%S")])


def generate_pointings(az_start, az_end, el_start, el_end, az_steps, el_steps):
    azimuths = [az_start + i * (az_end - az_start) / max(az_steps - 1, 1) for i in range(az_steps)]
    elevations = [el_start + i * (el_end - el_start) / max(el_steps - 1, 1) for i in range(el_steps)]
    for el in elevations:
        for az in azimuths:
            yield az, el


class Hydra:
    def __init__(self, telescope_id=1, freq_mhz=1420.40575, sample_rate=2.048e6, gain="auto"):
        self.telescope_id = telescope_id
        self.freq_mhz = freq_mhz
        self.sample_rate = sample_rate
        self.gain = gain
        self.sdr = RtlSdr()
        self.sdr.sample_rate = sample_rate
        self.sdr.center_freq = freq_mhz * 1e6
        self.sdr.gain = gain

        self.planetary_drive = PlanetaryDrive()

    def close(self):
        self.sdr.close()

    def move_to(self, azimuth, elevation):
        """
        Move telescope axes to requested azimuth and elevation using planetary drive.
        This method should block until movement completes.
        """
        print(f"Moving to Az: {azimuth}°, El: {elevation}°")
        self.planetary_drive.move_to_azimuth(azimuth)
        self.planetary_drive.move_to_elevation(elevation)

        # Optionally add wait or feedback here if planetary_drive supports it
        while not self.planetary_drive.at_target_position():
            time.sleep(0.1)

        print(f"Arrived at Az: {azimuth}°, El: {elevation}°")

    def run_job(self, job):
        print(f"Starting job: {job}")
        for az, el in generate_pointings(job['az_start'], job['az_end'], job['el_start'], job['el_end'],
                                         job['az_steps'], job['el_steps']):
            self.move_to(az, el)
            wait_for_settle()

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"telescope{self.telescope_id}_az{int(az)}_el{int(el)}_{timestamp}.bin"
            record_samples(self.telescope_id, self.sdr, job['record_duration_sec'], filename, az, el)

        print("Job complete.")

    def wait_for_jobs(self):
        print("Node waiting for jobs...")
        while True:
            time.sleep(5)
            # Replace this with your real job receiving method
            demo_job = {
                "az_start": 0,
                "az_end": 30,
                "az_steps": 3,
                "el_start": 45,
                "el_end": 60,
                "el_steps": 2,
                "record_duration_sec": 3
            }
            print("Received a demo job!")
            self.run_job(demo_job)
            # For continuous waiting remove the break
            # break
