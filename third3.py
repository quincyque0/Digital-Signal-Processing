import numpy as np
import matplotlib.pyplot as plt


T = 1.0        
f1 = 1.0 / T         
fs = 10000            
t = np.linspace(0, T, int(fs * T), endpoint=False)
dt = t[1] - t[0]


N_max = 10
corr_sin = np.zeros((N_max, N_max))

for k in range(1, N_max + 1):
    sk = np.sin(2 * np.pi * k * f1 * t)
    for n in range(1, N_max + 1):
        sn = np.sin(2 * np.pi * n * f1 * t)
        corr_sin[k - 1, n - 1] = np.trapz(sk * sn, t)


indices = np.arange(-5, 6)  
corr_exp = np.zeros((len(indices), len(indices)), dtype=complex)

for i, k in enumerate(indices):
    sk = np.exp(1j * 2 * np.pi * k * f1 * t)
    for m, n in enumerate(indices):
        sn = np.exp(1j * 2 * np.pi * n * f1 * t)
        corr_exp[i, m] = np.trapz(sk * np.conj(sn), t)

delta_f_vec = np.linspace(-f1, f1, 200)
leakage_freq = []
s1 = np.sin(2 * np.pi * 1 * f1 * t)


for df in delta_f_vec:
    s2_shifted = np.sin(2 * np.pi * (2 * f1 + df) * t)
    leakage_freq.append(np.trapz(s1 * s2_shifted, t))


fraction_T = np.linspace(0.8, 1.2, 200)
leakage_time = []
s2_fixed = np.sin(2 * np.pi * 2 * f1 * t)

for frac in fraction_T:
    t_sub = np.linspace(0, T * frac, int(fs * T * frac), endpoint=False)
    sk_sub = np.sin(2 * np.pi * 1 * f1 * t_sub)
    sn_sub = np.sin(2 * np.pi * 2 * f1 * t_sub)
    leakage_time.append(np.trapz(sk_sub * sn_sub, t_sub))

plt.figure(figsize=(14, 10))


plt.subplot(2, 2, 1)
plt.imshow(corr_sin, cmap='Blues', extent=[1, N_max, N_max, 1])
plt.colorbar(label='Энергия интеграла')
plt.title(r'Ортогональность $\sin$: $\int_0^T s_k(t)s_n(t)dt$')
plt.xlabel('Гармоника n')
plt.ylabel('Гармоника k')


plt.subplot(2, 2, 2)
plt.imshow(np.abs(corr_exp), cmap='Greens', extent=[-5, 5, 5, -5])
plt.colorbar(label='|Интеграл|')
plt.title(r'Ортогональность $\exp$: $|\int_0^T s_k(t)s_n^*(t)dt|$')
plt.xlabel('Гармоника n')
plt.ylabel('Гармоника k')


plt.subplot(2, 2, 3)
plt.plot(delta_f_vec / f1, np.abs(leakage_freq), 'r', lw=2)
plt.axvline(0, color='gray', linestyle='--')
plt.grid(True)
plt.title(r'Нарушение ортогональности при сдвиге частоты $\Delta f$')
plt.xlabel(r'$\Delta f / f_1$')
plt.ylabel('Абсолютное значение интеграла')

plt.subplot(2, 2, 4)
plt.plot(fraction_T, np.abs(leakage_time), 'm', lw=2)
plt.axvline(1.0, color='gray', linestyle='--')
plt.grid(True)
plt.title('Нарушение ортогональности при изменении пределов $[0, T_{int}]$')
plt.xlabel(r'$T_{int} / T$')
plt.ylabel('Абсолютное значение интеграла')

plt.tight_layout()
plt.show()