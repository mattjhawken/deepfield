import time
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
import matplotlib.pyplot as plt
import board
import random
from rtlsdr import RtlSdr


sdr = RtlSdr()
#kit = MotorKit(i2c=board.I2C())

sdr.sample_rate = 2.048e6
sdr.center_freq = 1020e6
sdr.freq_correction = 60
sdr.gain = "auto"


samples = sdr.read_samples(sdr.sample_rate * 1)

psd_values, frequencies = plt.psd(samples, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
print(len(psd_values))
#plt.show()

#for z in range(100):
#	time.sleep(1)
#	for i in range(100):
#		for _ in range(50):
#			kit.stepper2.onestep(direction=stepper.BACKWARD)
#			time.sleep(0.02)
#		time.sleep(3)
#	time.sleep(1)
#	for i in range(100):
#		for _ in range(5):
#			kit.stepper2.onestep(direction=stepper.FORWARD)
#			time.sleep(0.2)
#		time.sleep(3)
