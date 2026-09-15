import matplotlib.pyplot as plt
import numpy as np

f0 = 5  
t = np.linspace(0, 1, 1000)

A1, phi1 = 3.0, 0
A2, phi2 = 4.0, np.pi / 2
x1 = A1 * np.cos(2 * np.pi * f0 * t + phi1)
x2 = A2 * np.cos(2 * np.pi * f0 * t + phi2)
x_sum2 = x1 + x2

A3, phi3 = 2.0, np.pi
x3 = A3 * np.cos(2 * np.pi * f0 * t + phi3)
x_sum3 = x1 + x2 + x3

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, x_sum2, color='b')
plt.title('Сумма двух колебаний (N=2)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, x_sum3, color='r')
plt.title('Сумма трех колебаний (N=3)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()