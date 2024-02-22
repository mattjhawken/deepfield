import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

dir = os.getcwd()

# Lists to store all frequency data
all_data = {}

for file in os.listdir():
    if all(n in "0123456789" for n in file):
        file_path = os.path.join(dir, file)
        all_data[file] = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            string_data = ""
            for line in lines:
                string_data += line
            parsed_data = string_data.split(",,,")
            frequencies = parsed_data[2::2]
            psds = parsed_data[3::2]

            # Extract frequency data from current file
            for freq_data, psd_data in zip(frequencies, psds):
                freq = freq_data.split("[")[1].split("]")[0].split()
                psd = psd_data.split("[")[1].split("]")[0].split()
                formatted_freq = [float(n) for n in freq]
                formatted_psd = [float(n) for n in psd]
                
                all_data[file].append((formatted_freq, formatted_psd))

all_data = dict(sorted(all_data.items(), key=lambda x: int(x[0])))

for frequency, data in all_data.items():
    frequencies = [d[0] for d in data] # Frequency readings over time
    psds = [d[1] for d in data] # PSD readings over time
    
    plt.ion()
    fig, ax = plt.subplots()
    
    # Find the maximum frequency  value across all instances
    min_freq = min([min(freqs) for freqs in frequencies])
    max_freq = max([max(freqs) for freqs in frequencies])
    min_psd = min([min(ps) for ps in psds])
    max_psd = max([max(ps) for ps in psds])

    ax.set_ylim(min_psd, max_psd)  # Set the x-axis limit
    ax.set_xlim(min_freq, max_freq)  # Set the x-axis limit

    for i in range(len(psds)):  # plot one psd-freq-time data and animate between time data
        ax.clear()  # Clear the previous plot
        formatted_psd = psds[i]
        formatted_freq = frequencies[i]
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_title(f"{frequency}MHz")

        ax.plot(formatted_freq, formatted_psd)
        plt.pause(1)
