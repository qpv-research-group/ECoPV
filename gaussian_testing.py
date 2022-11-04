import numpy as np
import matplotlib.pyplot as plt

interval = 0.1
wl = np.arange(380, 780, interval)

centres = [500, 700]
widths = [100, 50]

height = 1

wider_peak = np.argmax(widths)

spec = height*np.exp(-(wl-centres[wider_peak])**2/(2*widths[wider_peak]**2))

plt.figure()
plt.plot(wl, spec)

height_of_2 = 1-height*np.exp(-(centres[wider_peak-1]-centres[wider_peak])**2/(2*widths[wider_peak]**2))

spec += height_of_2*np.exp(-(wl-centres[wider_peak-1])**2/(2*widths[wider_peak-1]**2))

plt.plot(wl, spec)
plt.show()

return height*spec/max(spec)
