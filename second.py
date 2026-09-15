import matplotlib.pyplot as plt
import numpy as np

# Определяем ось времени
t = np.linspace(0, 1, 1000)

# Определяем параметры гармонического колебания
A = 5   # амплитуда колебания
f = 5   # частота колебания
ph = 0  # фаза колебания

# Записываем выражение для сигнала
x = A * np.sin(2 * np.pi * f * t + ph)

# Строим график колебания
plt.plot(t, x)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title(r'$A={}V, F={} Hz, \phi={}^\circ$'.format(A, f, ph))
plt.grid(True)
plt.show()