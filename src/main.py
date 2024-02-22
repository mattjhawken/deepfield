# from adafruit_motorkit import MotorKit
# from adafruit_motor import stepper
import matplotlib.pyplot as plt
from rtlsdr import *
import time
import csv

# SDR parameters
SAMPLE_RATE = 2.048e6
CENTER_FREQ = 950e6
FREQ_CORRECTION = 60
GAIN = "auto"
FREQ_RANGE = list(range(920, 935)) + list(range(1220, 1235))


if __name__ == "__main__":
	sdr = RtlSdr()
	# kit = MotorKit()

	sdr.sample_rate = SAMPLE_RATE
	sdr.freq_correction = FREQ_CORRECTION
	sdr.gain = GAIN

	while True:
		# Target range for 12.178GHz
		for freq in FREQ_RANGE:
			dt = time.strftime("%Y-%m-%d %H:%M:%S")
			sdr.center_freq = freq * 1e6
			samples = sdr.read_samples(SAMPLE_RATE)
			psd_values, frequencies = plt.psd(samples, Fs=SAMPLE_RATE/1e6, Fc=freq/1e6)

			with open(f"data_{freq}.csv", "a", newline='') as csvfile:
				csvwriter = csv.writer(csvfile)
				csvwriter.writerow([dt, psd_values.tolist(), frequencies.tolist()])

	sdr.close()
