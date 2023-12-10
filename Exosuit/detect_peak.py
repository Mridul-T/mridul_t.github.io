#%%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks

x = np.loadtxt("data_slow.txt", skiprows=1, delimiter=",")
x_2 = np.loadtxt("data_fast.txt", skiprows=1, delimiter=",")
print(x_2.shape)
#%%)
x=x[:,1]*-1
x_2=x_2[:,1]
# peaks, _ = find_peaks(x, distance=20)
peaks, _ = find_peaks(x_2,width = 40)
peaks2, _ = find_peaks(x, prominence=8)      # BEST!
peaks3, _ = find_peaks(x, width=40)
peaks4, _ = find_peaks(x, threshold=0.4)     # Required vertical distance to its direct neighbouring samples, pretty useless
plt.figure(figsize=(16,8))
plt.subplot(2, 2, 1)
plt.plot(peaks, x_2[peaks], "xr"); plt.plot(x_2); plt.legend(['fast_width'])
plt.subplot(2, 2, 2)
plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['prominence'])
plt.subplot(2, 2, 3)
plt.plot(peaks3, x[peaks3], "vg"); plt.plot(x); plt.legend(['width'])
plt.subplot(2, 2, 4)
plt.plot(peaks4, x[peaks4], "xk"); plt.plot(x); plt.legend(['threshold'])
plt.show()
# %%
