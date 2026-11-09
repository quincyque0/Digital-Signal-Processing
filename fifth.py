import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import matplotlib.pyplot as plt

def trapz(y, x=None, dx=1.0, axis=-1):
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)

def rect_pulse(t, t_start=0.0, t_end=1.0, A=1.0):
    return np.where((t >= t_start) & (t <= t_end), A, 0.0)

def harmonic_pulse(t, t_start=0.0, t_end=4.0, f0=1.0, A=1.0):
    return np.where((t >= t_start) & (t <= t_end), A * np.sin(2.0 * np.pi * f0 * t), 0.0)

def y_rect_analytical(t):
    res = np.zeros_like(t, dtype=float)
    c1 = (t >= 0.0) & (t < 1.0)
    c2 = (t >= 1.0) & (t < 2.0)
    c3 = (t >= 2.0) & (t <= 3.0)
    res[c1] = t[c1]
    res[c2] = 1.0
    res[c3] = 3.0 - t[c3]
    return res

def fourier_integral(t_grid, s, f_grid):
    dt = t_grid[1] - t_grid[0]
    kernel = np.exp(-1j * 2.0 * np.pi * np.outer(f_grid, t_grid))
    return trapz(s * kernel, dx=dt, axis=1)

print("ПУНКТ 1.2")

dtau = 0.001
tau = np.arange(-5.0, 5.0 + dtau/2, dtau)

s1_tau = rect_pulse(tau, 0.0, 1.0)
s2_tau = rect_pulse(tau, 0.0, 2.0)

s1_neg_tau = rect_pulse(-tau, 0.0, 1.0)

t_offsets = [-1.0, 0.0, 2.0]
s1_shifted = {t_val: rect_pulse(t_val - tau, 0.0, 1.0) for t_val in t_offsets}

prod_shifted = {t_val: s2_tau * s1_shifted[t_val] for t_val in t_offsets}

test_t = np.array([0.0, 0.2, 1.0])
conv_test_num = []
conv_test_ana = y_rect_analytical(test_t)

for t_val in test_t:
    s1_t_minus_tau = rect_pulse(t_val - tau, 0.0, 1.0)
    integrand = s2_tau * s1_t_minus_tau
    integral_val = trapz(integrand, dx=dtau)
    conv_test_num.append(integral_val)
conv_test_num = np.array(conv_test_num)

print(f"{'Момент t, с':^12} | {'Численное y(t)':^16} | {'Аналитическое':^16} | {'Погрешность':^14}")
print("-" * 64)
for t_val, num_val, ana_val in zip(test_t, conv_test_num, conv_test_ana):
    err = np.abs(num_val - ana_val)
    print(f"{t_val:^12.2f} | {num_val:^16.6f} | {ana_val:^16.6f} | {err:^14.2e}")

t_grid_conv = np.arange(-1.0, 4.0 + 0.05/2, 0.05)
y_num = np.zeros_like(t_grid_conv)

for i, t_val in enumerate(t_grid_conv):
    s1_cur = rect_pulse(t_val - tau, 0.0, 1.0)
    y_num[i] = trapz(s2_tau * s1_cur, dx=dtau)

y_ana = y_rect_analytical(t_grid_conv)
max_err_conv = np.max(np.abs(y_num - y_ana))
print(f"\nМаксимальная абсолютная погрешность свертки на сетке: {max_err_conv:.2e}")

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs[0].plot(tau, s2_tau, 'b', lw=2, label=r'$s_2(\tau)$ ($\tau \in [0, 2]$)')
axs[0].plot(tau, s1_tau, 'g--', lw=1.5, label=r'$s_1(\tau)$ ($\tau \in [0, 1]$)')
axs[0].plot(tau, s1_neg_tau, 'r-.', lw=2, label=r'$s_1(-\tau)$ ($\tau \in [-1, 0]$)')
axs[0].set_ylabel('Амплитуда')
axs[0].set_title('1. Замена переменной и инверсия аргумента')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc='upper right')

colors = {-1.0: 'purple', 0.0: 'orange', 2.0: 'teal'}
for t_val in t_offsets:
    axs[1].plot(tau, s1_shifted[t_val], color=colors[t_val], lw=1.8, 
                label=f'$s_1({int(t_val)} - \\tau)$')
axs[1].set_ylabel('Амплитуда')
axs[1].set_title(r'2. Смещение функции $s_1(t - \tau)$ для моментов $t = -1, 0, 2$ с')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc='upper right')

for t_val in t_offsets:
    axs[2].plot(tau, prod_shifted[t_val], color=colors[t_val], lw=2, 
                label=f'$s_2(\\tau)s_1({int(t_val)} - \\tau)$')
    axs[2].fill_between(tau, prod_shifted[t_val], color=colors[t_val], alpha=0.2)
axs[2].set_ylabel('Произведение')
axs[2].set_xlabel(r'Вспомогательная переменная интегрирования $\tau$, с')
axs[2].set_title(r'3. Подынтегральное произведение $s_2(\tau)s_1(t - \tau)$ в интервале $[-5 \leq \tau \leq 5]$')
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].legend(loc='upper right')
axs[2].set_xlim(-5, 5)

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, 
                        gridspec_kw={'height_ratios': [3, 1]})

axs[0].plot(t_grid_conv, y_num, 'bo-', ms=4, lw=1.5, label='Численная свертка (шаг 0.05 с)')
axs[0].plot(t_grid_conv, y_ana, 'r--', lw=2, label='Аналитическая трапеция')
axs[0].plot(test_t, conv_test_num, 'go', ms=8, label='Контрольные точки (t = 0, 0.2, 1.0 с)')
axs[0].set_ylabel('y(t)')
axs[0].set_title('Пункт 1.2: Результат вычисления интеграла свертки $y(t) = s_1(t) * s_2(t)$')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc='upper right')

axs[1].plot(t_grid_conv, np.abs(y_num - y_ana), 'm-', lw=1.5, label=r'Погрешность $|y_{числ} - y_{аналит}|$')
axs[1].set_ylabel('Погрешность')
axs[1].set_xlabel('Время t, с')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc='upper right')
axs[1].set_xlim(-1, 4)

plt.tight_layout()
plt.show()


print("ПУНКТ 1.3")


f0_harm = 1.0
t_grid_harm = np.arange(-1.0, 8.0 + 0.05/2, 0.05)

tau_harm = np.arange(-5.0, 10.0 + dtau/2, dtau)
s2_harm = harmonic_pulse(tau_harm, 0.0, 4.0, f0=f0_harm, A=1.0)

y_harm_num = np.zeros_like(t_grid_harm)
for i, t_val in enumerate(t_grid_harm):
    s1_cur = rect_pulse(t_val - tau_harm, 0.0, 1.0)
    y_harm_num[i] = trapz(s2_harm * s1_cur, dx=dtau)

f0_half = 0.5
s2_harm_half = harmonic_pulse(tau_harm, 0.0, 4.0, f0=f0_half, A=1.0)
y_harm_half = np.zeros_like(t_grid_harm)
for i, t_val in enumerate(t_grid_harm):
    s1_cur = rect_pulse(t_val - tau_harm, 0.0, 1.0)
    y_harm_half[i] = trapz(s2_harm_half * s1_cur, dx=dtau)

print(f"Частота f0 = {f0_harm} Гц (период T0 = 1 с):")
print(f"  Максимум свертки (переходный процесс t in [0, 1]): {np.max(y_harm_num):.4f}")
print(f"  Значение в стационарном режиме (t in [1, 4]):      {y_harm_num[np.argmin(np.abs(t_grid_harm - 2.5))]:.4f} (полное подавление)")
print(f"  Минимум свертки (переходный процесс t in [4, 5]):  {np.min(y_harm_num):.4f}")

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

t_disp = np.linspace(-1, 8, 2000)
axs[0].plot(t_disp, rect_pulse(t_disp, 0, 1), 'b', lw=2, label=r'$s_1(t)$ — импульс ($\tau_1 = 1$ с)')
axs[0].plot(t_disp, harmonic_pulse(t_disp, 0, 4, f0=f0_harm), 'r', lw=1.5, 
            label=rf'$s_2(t) = \sin(2\pi f_0 t)$ ($\tau_2 = 4$ с, $f_0 = {f0_harm}$ Гц)')
axs[0].set_ylabel('Амплитуда')
axs[0].set_title('Пункт 1.3: Исходные сигналы $s_1(t)$ и $s_2(t)$')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc='upper right')

axs[1].plot(t_grid_harm, y_harm_num, 'b.-', lw=1.8, label=rf'Свертка $y(t) = s_1(t) * s_2(t)$ ($f_0 = {f0_harm}$ Гц)')
axs[1].plot(t_grid_harm, y_harm_half, 'g--', lw=1.5, alpha=0.7, label=rf'Сравнение: свертка при $f_0 = {f0_half}$ Гц')
axs[1].axvspan(0, 1, color='orange', alpha=0.15, label=r'Переходный режим входа ($0 \leq t < 1$)')
axs[1].axvspan(1, 4, color='green', alpha=0.10, label=r'Квазистационарный режим ($1 \leq t < 4$)')
axs[1].axvspan(4, 5, color='purple', alpha=0.15, label=r'Переходный режим выхода ($4 \leq t \leq 5$)')
axs[1].set_ylabel('y(t)')
axs[1].set_xlabel('Время t, с')
axs[1].set_title('Результат вычисления интеграла свертки на интервале (-1, 8) с шагом 0.05 с')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc='upper right', fontsize=9)
axs[1].set_xlim(-1, 8)

plt.tight_layout()
plt.show()

print("ПУНКТ 1.4")

dt_spec = 0.0005
t_sig = np.arange(-2.0, 5.0 + dt_spec/2, dt_spec)

s1_sig = rect_pulse(t_sig, 0.0, 1.0)
s2_sig = rect_pulse(t_sig, 0.0, 2.0)

y_sig = np.zeros_like(t_sig)
tau_dense = np.arange(-4.0, 4.0 + dt_spec/2, dt_spec)
s2_dense = rect_pulse(tau_dense, 0.0, 2.0)
for i, t_val in enumerate(t_sig):
    if -0.1 <= t_val <= 3.1:
        s1_cur = rect_pulse(t_val - tau_dense, 0.0, 1.0)
        y_sig[i] = trapz(s2_dense * s1_cur, dx=dt_spec)

f_grid = np.linspace(0.0, 2.5, 500)

S1_f = fourier_integral(t_sig, s1_sig, f_grid)
S2_f = fourier_integral(t_sig, s2_sig, f_grid)
Y_f = fourier_integral(t_sig, y_sig, f_grid)

Prod_f = S1_f * S2_f

test_freqs = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
S1_test = fourier_integral(t_sig, s1_sig, test_freqs)
S2_test = fourier_integral(t_sig, s2_sig, test_freqs)
Prod_test = S1_test * S2_test
Y_test = fourier_integral(t_sig, y_sig, test_freqs)

print(f"{'Частота f, Гц':^13} | {'|S1(f)|':^10} | {'|S2(f)|':^10} | {'|S1*S2|':^10} | {'|Y(f)|':^10} | {'Погрешность':^12}")
print("-" * 75)
for f_v, s1_v, s2_v, p_v, y_v in zip(test_freqs, S1_test, S2_test, Prod_test, Y_test):
    err = np.abs(y_v - p_v)
    print(f"{f_v:^13.2f} | {np.abs(s1_v):^10.4f} | {np.abs(s2_v):^10.4f} | {np.abs(p_v):^10.4f} | {np.abs(y_v):^10.4f} | {err:^12.2e}")

max_diff_fourier = np.max(np.abs(Y_f - Prod_f))
print(f"\nМаксимальное отклонение |Y(f) - S1(f)*S2(f)|: {max_diff_fourier:.2e}")

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(f_grid, np.abs(Y_f), 'b', lw=2.5, label=r'$|Y(f)| = |\mathcal{F}\{s_1(t) * s_2(t)\}|$')
axs[0].plot(f_grid, np.abs(Prod_f), 'r--', lw=1.8, label=r'$|S_1(f) \cdot S_2(f)|$')
axs[0].plot(test_freqs, np.abs(Y_test), 'go', ms=6, label='Контрольные точки')
axs[0].set_ylabel(r'$|S(f)|$, В$\cdot\text{с}^2$')
axs[0].set_title('Пункт 1.4: Амплитудно-частотные спектры')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc='upper right')

axs[1].plot(f_grid, np.angle(Y_f), 'b', lw=2, label=r'$\arg Y(f)$')
axs[1].plot(f_grid, np.angle(Prod_f), 'r--', lw=1.5, label=r'$\arg [S_1(f) \cdot S_2(f)]$')
axs[1].set_ylabel('Фаза, рад')
axs[1].set_yticks([-np.pi, 0, np.pi])
axs[1].set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
axs[1].set_title('Фазо-частотные спектры')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc='upper right')

axs[2].plot(f_grid, np.abs(Y_f - Prod_f), 'm', lw=1.5, label=r'Разность $|Y(f) - S_1(f)S_2(f)|$')
axs[2].set_ylabel('Разность, В$\cdot\text{с}^2$')
axs[2].set_xlabel('Частота f, Гц')
axs[2].set_title('Абсолютное расхождение теоремы о свертке')
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].legend(loc='upper right')
axs[2].set_xlim(0, 2.5)

plt.tight_layout()
plt.show()
