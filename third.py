import matplotlib.pyplot as plt
import numpy as np


f0 = 5.0
T = 1.0 / f0
Ts = 0.01  
A = 6.0


t = np.arange(0, T, Ts)


def analyze_harmonic(phi):
    x = A * np.cos(2 * np.pi * f0 * t + phi)

    harmonics = 5 
    a = np.zeros(harmonics)
    b = np.zeros(harmonics)

    
    a[0] = (1 / T) * np.sum(x) * Ts
    b[0] = 0.0

    
    for n in range(1, harmonics):
        sc = np.cos(2 * np.pi * n * f0 * t)
        ss = np.sin(2 * np.pi * n * f0 * t)
        a[n] = (2 / T) * np.sum(x * sc) * Ts
        b[n] = (2 / T) * np.sum(x * ss) * Ts

    An = np.sqrt(a**2 + b**2)
    An[0] = a[0]

    phin = np.zeros(harmonics)
    for n in range(1, harmonics):
        if An[n] > 1e-4:  
            phin[n] = -np.arctan2(b[n], a[n])

    return a, b, An, phin


a_0, b_0, An_0, phin_0 = analyze_harmonic(phi=0.0)
print("Коэффициенты a_n:", np.round(a_0, 4))
print("Коэффициенты b_n:", np.round(b_0, 4))
print("Амплитуды A_n:   ", np.round(An_0, 4))
print("Фазы phi_n:      ", np.round(phin_0, 4))


phi_new = np.pi / 3
a_pi, b_pi, An_pi, phin_pi = analyze_harmonic(phi=phi_new)
print(f"\nНачальная фаза phi = pi/3 ({phi_new:.4f} рад)")
print("Коэффициенты a_n:", np.round(a_pi, 4))
print("Коэффициенты b_n:", np.round(b_pi, 4))
print("Амплитуды A_n:   ", np.round(An_pi, 4))
print("Фазы phi_n:      ", np.round(phin_pi, 4))


n_vals = np.arange(5)
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

axs[0, 0].stem(n_vals, An_0)
axs[0, 0].set_title(r"Амплитудный спектр $A_n$ ($\phi = 0$)")
axs[0, 0].set_xlabel("n")
axs[0, 0].grid(True)

axs[0, 1].stem(n_vals, phin_0)
axs[0, 1].set_title(r"Фазовый спектр $\phi_n$ ($\phi = 0$)")
axs[0, 1].set_xlabel("n")
axs[0, 1].grid(True)

axs[1, 0].stem(n_vals, An_pi)
axs[1, 0].set_title(rf"Амплитудный спектр $A_n$ ($\phi = \pi/3$)")
axs[1, 0].set_xlabel("n")
axs[1, 0].grid(True)

axs[1, 1].stem(n_vals, phin_pi)
axs[1, 1].set_title(rf"Фазовый спектр $\phi_n$ ($\phi = \pi/3$)")
axs[1, 1].set_xlabel("n")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()