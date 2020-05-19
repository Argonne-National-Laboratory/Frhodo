import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def lag_finder(x1, y1, x2, y2):
    n = len(y1)
    if len(y1) > len(y2):
        i_match = np.argwhere(np.in1d(x1, x2)).flatten()
        print(i_match)
        y2 = np.append(np.zeros([1, i_match[0]]), y2)
        y2 = np.append(y2, np.zeros([1, len(x1)-i_match[-1]-1]))
    
    corr = signal.correlate(y1, y2, mode='full')
    # corr = np.convolve(y1, y2[::-1], mode='full')
    
    dt = np.linspace(-x1[-1], x1[-1], 2*x1.size-1)
    # delay = dt[corr.argmax()]
    plt.figure()
    plt.plot(dt, corr)
    plt.show()
    
    print(x2[0], x2[-1])
    print(x1[0], x1[-1])
    delay = np.mean(np.diff(x1))*corr.argmax() - x1[-1]
    
    return delay

time_shift = 2
x1 = np.arange(0, 2*np.pi, np.pi/2**9)
x2 = x1[int(len(x1)/4):int(len(x1)*1/2)]
x2 = x1
y1 = np.sin(x1)
y2 = np.sin(x2-time_shift)
# y1 *= np.random.normal(0.95, 1.05, y1.shape)
# y1 += np.random.normal(0, 0.025, y1.shape)

delay = lag_finder(x1, y1, x2, y2)
print(delay)

plt.figure()
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x2+delay, y2)
plt.show()