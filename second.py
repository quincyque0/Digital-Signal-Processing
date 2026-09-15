import matplotlib.pyplot as plt
import numpy as np

f = 2  
t = np.linspace(0, 5 / f, 1000)

x = (4 / np.pi) * np.cos(2 * np.pi * f * t - np.pi / 2) + (4 / (3 * np.pi)) * np.cos(2 * np.pi * 3 * f * t - np.pi / 2)

plt.figure(figsize=(9, 4))
plt.plot(t, x)
plt.title('Сумма двух гармоник (Формула 4)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()