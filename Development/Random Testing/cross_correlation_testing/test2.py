import numpy as np

# Create the time axis (seconds)
num_samples = 1001
samples_per_second = 1000
freq_Hz = 0.5
t = np.linspace(0.0, ((num_samples - 1) / samples_per_second), num_samples)
# Create a sine wave, a(t), with a frequency of 1 Hz
a = np.sin((2.0 * np.pi) * freq_Hz * t)
# Create b(t), a (pi / 2.0) phase-shifted replica of a(t)
b_shift = (np.pi / 2.0)
b = np.sin((2.0 * np.pi) * freq_Hz * t + b_shift)

# Cross-correlate the signals, a(t) & b(t)
ab_corr = np.correlate(a, b, "full")
dt = np.linspace(-t[-1], t[-1], (2 * num_samples) - 1)
# Calculate time & phase shifts
t_shift_alt = (1.0 / samples_per_second) * ab_corr.argmax() - t[-1]
t_shift = dt[ab_corr.argmax()]
# Limit phase_shift to [-pi, pi]
phase_shift = ((2.0 * np.pi) * ((t_shift / (1.0 / freq_Hz)) % 1.0)) - np.pi

manual_t_shift = (b_shift / (2.0 * np.pi)) / freq_Hz

# Print out applied & calculated shifts
print("Manual time shift: {}".format(manual_t_shift))
print("Alternate calculated time shift: {}".format(t_shift_alt))
print("Calculated time shift: {}".format(t_shift))                    
print("Manual phase shift: {}".format(b_shift))                           
print("Calculated phase shift: {}".format(phase_shift)) 

import matplotlib.pyplot as plt

plt.figure()
plt.plot(dt, ab_corr)
# plt.plot(t, a)
# plt.plot(t, b)
plt.show()