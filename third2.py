import numpy as np
import matplotlib.pyplot as plt

T = 0.1
tau = 0.025
f0 = 1.0 / T
dt = 1e-4
t = np.arange(0, T, dt)


x = np.where((t <= tau/2) | (t >= T - tau/2), 1.0, 0.0)

n_max = 6
n_vals = np.arange(n_max + 1)
a = np.zeros(n_max + 1)
b = np.zeros(n_max + 1)

a[0] = (2 / T) * np.sum(x) * dt
for n in range(1, n_max + 1):
    a[n] = (2 / T) * np.sum(x * np.cos(2 * np.pi * n * f0 * t)) * dt
    b[n] = (2 / T) * np.sum(x * np.sin(2 * np.pi * n * f0 * t)) * dt

A_n = np.zeros(n_max + 1)
A_n[0] = a[0] / 2
A_n[1:] = np.sqrt(a[1:]**2 + b[1:]**2)
phi_n = -np.arctan2(b, a)

def synthesize(harmonics, t_arr):
    res = a[0] / 2
    for n in range(1, harmonics + 1):
        res += a[n] * np.cos(2 * np.pi * n * f0 * t_arr) + b[n] * np.sin(2 * np.pi * n * f0 * t_arr)
    return res

plt.figure(figsize=(10, 5))
plt.plot(t, x, 'k--', label='Исходный сигнал')
for k in [2, 4, 6]:
    plt.plot(t, synthesize(k, t), label=f'Синтез ({k} гармоник)')
plt.title('Синтез прямоугольного сигнала гармониками ряда Фурье')
plt.xlabel('Время, с')
plt.grid(True)
plt.legend()
plt.show()