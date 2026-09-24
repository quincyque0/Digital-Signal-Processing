import numpy as np
import matplotlib.pyplot as plt

# ----------------- ЗАДАНИЕ 1 -----------------
t1 = np.linspace(0, 1, 1000)
A1, f1, ph1 = 5, 5, 0
x1 = A1 * np.sin(2 * np.pi * f1 * t1 + np.radians(ph1))

plt.figure(figsize=(8, 3.5))
plt.plot(t1, x1, 'b-', linewidth=1.5)
plt.title(f'Задание 1: A = {A1} V, F = {f1} Hz, $\\phi = {ph1}^\\circ$')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда (В)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('1.png', dpi=300)
plt.close()

# ----------------- ЗАДАНИЕ 2 -----------------
t2 = np.linspace(0, 1, 1000)
A2, f2 = 1, 5
phi_list = [0, 90, 180, 270]

plt.figure(figsize=(9, 5.5))
for i, phi_deg in enumerate(phi_list):
    x2 = A2 * np.sin(2 * np.pi * f2 * t2 + np.radians(phi_deg))
    plt.subplot(2, 2, i + 1)
    plt.plot(t2, x2, 'navy', linewidth=1.3)
    plt.title(f'$\\Phi = {phi_deg}^\\circ$')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда (В)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-1.2, 1.2)
plt.tight_layout()
plt.savefig('2.png', dpi=300)
plt.close()

# ----------------- ЗАДАНИЕ 3 -----------------
f0 = 5
t3 = np.linspace(0, 1, 1000)
s1 = 3.0 * np.cos(2 * np.pi * f0 * t3 + 0.0)
s2 = 4.0 * np.cos(2 * np.pi * f0 * t3 + np.pi / 2.0)
s3 = 2.0 * np.cos(2 * np.pi * f0 * t3 + np.pi)

x_sum2 = s1 + s2
x_sum3 = s1 + s2 + s3

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t3, x_sum2, 'b-', linewidth=1.8, label=r'$N=2: A_{рез}=5.0$ В')
plt.title('Сложение двух колебаний (N=2)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда (В)')
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(1, 2, 2)
plt.plot(t3, x_sum3, 'r-', linewidth=1.8, label=r'$N=3: A_{рез}\approx 4.12$ В')
plt.title('Сложение трех колебаний (N=3)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда (В)')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('3.png', dpi=300)
plt.close()

# ----------------- ЗАДАНИЕ 4 -----------------
f4 = 2
t4 = np.linspace(0, 5 / f4, 1500)
h1 = (4 / np.pi) * np.cos(2 * np.pi * f4 * t4 - np.pi / 2)
h3 = (4 / (3 * np.pi)) * np.cos(2 * np.pi * 3 * f4 * t4 - np.pi / 2)
x4 = h1 + h3

plt.figure(figsize=(8.5, 3.5))
plt.plot(t4, h1, ':', color='gray', label='1-я гармоника ($f$)')
plt.plot(t4, h3, ':', color='orange', label='3-я гармоника ($3f$)')
plt.plot(t4, x4, 'm-', linewidth=2.0, label='Сумма')
plt.title('Сложение кратных гармоник (5 периодов)')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда (В)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('4.png', dpi=300)
plt.close()

# ----------------- ЗАДАНИЕ 5 -----------------
f5 = 2
t5 = np.linspace(0, 5 / f5, 2000)
N_harmonics = [1, 2, 5, 20]

plt.figure(figsize=(9, 6))
for idx, N in enumerate(N_harmonics):
    x_meander = np.zeros_like(t5)
    for n in range(1, N + 1):
        k = 2 * n - 1
        x_meander += (4 / (k * np.pi)) * np.cos(2 * np.pi * k * f5 * t5 - np.pi / 2)
    plt.subplot(2, 2, idx + 1)
    plt.plot(t5, x_meander, 'darkred', linewidth=1.2)
    plt.title(f'N = {N} слагаемых')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда (В)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-1.4, 1.4)
plt.tight_layout()
plt.savefig('5.png', dpi=300)
plt.close()