import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import matplotlib.pyplot as plt

RC = 0.001
f_c = 1.0 / (2.0 * np.pi * RC)
f_max = 3.0 / RC

f0 = 1000.0
omega0 = 2.0 * np.pi * f0
T0 = 1.0 / f0
A_in = 1.0

print("ПРАКТИЧЕСКАЯ РАБОТА № 6: ЧАСТОТНЫЕ ХАРАКТЕРИСТИКИ И ПРЕОБРАЗОВАНИЕ СИГНАЛОВ")
print(f"Параметры цепи:")
print(f"  Постоянная времени RC = {RC*1e3:.2f} мс ({RC} с)")
print(f"  Частота среза f_ср = 1/(2*pi*RC) = {f_c:.2f} Гц")
print(f"  Верхняя граница диапазона 3/RC = {f_max:.1f} Гц")
print(f"  Рабочая частота входного сигнала f0 = {f0:.1f} Гц (период T0 = {T0*1e3:.2f} мс)")

print("ПУНКТ 1.1: ВЫЧИСЛЕНИЕ ИМПУЛЬСНОЙ ХАРАКТЕРИСТИКИ И СВЕРТКИ")

dt = 1e-6
t_sig = np.arange(0.0, 0.012 + dt/2, dt)

s1 = A_in * np.cos(omega0 * t_sig)

t_h = np.arange(0.0, 0.006 + dt/2, dt)
h = (1.0 / RC) * np.exp(-t_h / RC)

s_out_conv = np.convolve(s1, h) * dt
s_out_conv = s_out_conv[:len(t_sig)]

y_exact = (A_in / (1.0 + (omega0 * RC)**2)) * (
    np.cos(omega0 * t_sig) + (omega0 * RC) * np.sin(omega0 * t_sig) - np.exp(-t_sig / RC)
)

err_conv = np.max(np.abs(s_out_conv - y_exact))
print(f"Максимальное отклонение численной свертки от точного решения: {err_conv:.2e} В")

idx_steady = (t_sig >= 0.008) & (t_sig <= 0.012)
A_out_conv = (np.max(s_out_conv[idx_steady]) - np.min(s_out_conv[idx_steady])) / 2.0
print(f"Амплитуда выходного сигнала в установившемся режиме (свертка): {A_out_conv:.4f} В")
print(f"Коэффициент ослабления сигнала цепью: {A_in / A_out_conv:.2f} раз")

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(t_sig * 1e3, s1, 'b', lw=1.5, label=rf'Входной сигнал $s_1(t) = \cos(2\pi f_0 t)$ ($f_0 = {f0:.0f}$ Гц)')
axs[0].set_ylabel('Амплитуда, В')
axs[0].set_title('Пункт 1.1: Преобразование гармонического сигнала RC-цепью (временная свертка)')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc='upper right')

axs[1].plot(t_sig * 1e3, s_out_conv, 'r', lw=2, label=r'Выходной сигнал $s_{вых}(t) = s_1(t) * h(t)$')
axs[1].axvspan(0, 3*RC*1e3, color='orange', alpha=0.15, label=r'Переходный процесс ($t < 3RC = 3$ мс)')
axs[1].axvspan(3*RC*1e3, 12, color='green', alpha=0.10, label='Установившийся режим ($t \geq 3RC$)')
axs[1].set_ylabel('Амплитуда, В')
axs[1].set_xlabel('Время t, мс')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc='lower right')
axs[1].set_xlim(0, 12)

plt.tight_layout()
plt.show()

print("ПУНКТ 1.2: ВЫЧИСЛЕНИЕ АЧХ И ФЧХ RC-ЦЕПИ В ДИАПАЗОНЕ (0 ... 3/RC) Гц")


f_grid = np.linspace(0.0, f_max, 1000)
omega_grid = 2.0 * np.pi * f_grid

H_complex = 1.0 / (1.0 + 1j * omega_grid * RC)

ACH_mag = 1.0 / np.sqrt(1.0 + (omega_grid * RC)**2)
FCH_phi = -np.arctan(omega_grid * RC)
FCH_deg = np.degrees(FCH_phi)

check_freqs = np.array([0.0, f_c / 2.0, f_c, 500.0, 1000.0, 2000.0, 3000.0])
omega_chk = 2.0 * np.pi * check_freqs
H_chk_mag = 1.0 / np.sqrt(1.0 + (omega_chk * RC)**2)
H_chk_phi_rad = -np.arctan(omega_chk * RC)
H_chk_phi_deg = np.degrees(H_chk_phi_rad)
H_chk_db = 20.0 * np.log10(H_chk_mag)

print(f"{'Частота f, Гц':^14} | {'АЧХ |H(jf)|':^14} | {'АЧХ, дБ':^12} | {'ФЧХ, рад':^12} | {'ФЧХ, град':^12}")
print("-" * 75)
for f_v, m_v, db_v, r_v, d_v in zip(check_freqs, H_chk_mag, H_chk_db, H_chk_phi_rad, H_chk_phi_deg):
    print(f"{f_v:^14.2f} | {m_v:^14.4f} | {db_v:^12.2f} | {r_v:^12.4f} | {d_v:^12.2f}")

fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

axs[0].plot(f_grid, ACH_mag, 'b', lw=2, label=r'АЧХ $|H(j\omega)| = \frac{1}{\sqrt{1 + (\omega RC)^2}}$')
axs[0].axvline(f_c, color='r', linestyle='--', lw=1.5, label=rf'Частота среза $f_с = {f_c:.1f}$ Гц ($|H| = 0.707$)')
axs[0].axvline(f0, color='g', linestyle=':', lw=1.8, label=rf'Рабочая частота $f_0 = {f0:.0f}$ Гц ($|H| = {1/np.sqrt(1+(omega0*RC)**2):.4f}$)')
axs[0].axhline(1.0 / np.sqrt(2.0), color='gray', linestyle='-.', lw=1.0)
axs[0].set_ylabel('Коэффициент передачи |H|')
axs[0].set_title(rf'Пункт 1.2: Частотные характеристики RC-цепи ($RC = {RC*1e3:.0f}$ мс, $f \in [0; 3000]$ Гц)')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc='upper right')

axs[1].plot(f_grid, FCH_deg, 'm', lw=2, label=r'ФЧХ $\phi(\omega) = -\operatorname{arctg}(\omega RC)$')
axs[1].axvline(f_c, color='r', linestyle='--', lw=1.5, label=r'Фазовый сдвиг на срезе: $-45^\circ$')
axs[1].axvline(f0, color='g', linestyle=':', lw=1.8, label=rf'Фазовый сдвиг на $f_0$: {np.degrees(-np.arctan(omega0*RC)):.2f}$^\circ$')
axs[1].set_ylabel('Фазовый сдвиг, град')
axs[1].set_xlabel('Частота f, Гц')
axs[1].set_yticks([0, -30, -45, -60, -80.96, -90])
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc='lower left')
axs[1].set_xlim(0, f_max)

plt.tight_layout()
plt.show()

print("ПУНКТ 1.3: РАСЧЕТ ВЫХОДНОГО СИГНАЛА ПО ЧАСТОТНОМУ КОЭФФИЦИЕНТУ ПЕРЕДАЧИ")

H_f0_complex = 1.0 / (1.0 + 1j * omega0 * RC)
H_f0_mod = np.abs(H_f0_complex)
H_f0_phi = np.angle(H_f0_complex)
H_f0_deg = np.degrees(H_f0_phi)
delay_t = -H_f0_phi / omega0

print(f"Комплексный коэффициент передачи H(j*omega0): {H_f0_complex.real:+.4f} {H_f0_complex.imag:+.4f}j")
print(f"Модуль коэффициента передачи |H(j*omega0)|:     {H_f0_mod:.6f} (-16.07 дБ)")
print(f"Фазовый сдвиг phi(omega0):                     {H_f0_phi:.6f} рад ({H_f0_deg:.2f} град)")
print(f"Временное запаздывание delta_t:                {delay_t*1e3:.4f} мс ({delay_t/T0:.3f} периода)")

s_out_spectral = A_in * H_f0_mod * np.cos(omega0 * t_sig + H_f0_phi)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_sig * 1e3, s1, 'b--', lw=1.5, alpha=0.7, label=rf'Входной сигнал $s_{{вх}}(t) = \cos(2\pi \cdot {f0:.0f} t)$')
ax.plot(t_sig * 1e3, s_out_spectral, 'r', lw=2, 
        label=rf'Выходной сигнал $s_{{вых,уст}}(t) = {H_f0_mod:.4f}\cos(2\pi \cdot {f0:.0f} t - {abs(H_f0_phi):.3f})$')
ax.axvline(0, color='gray', linestyle=':')
ax.axvline(delay_t * 1e3, color='teal', linestyle=':', lw=1.5, 
           label=rf'Временное запаздывание $\Delta t = {delay_t*1e3:.3f}$ мс ({abs(H_f0_deg):.1f}$^\circ$)')
ax.set_title('Пункт 1.3: Преобразование сигнала в установившемся режиме по частотной характеристике')
ax.set_ylabel('Амплитуда, В')
ax.set_xlabel('Время t, мс')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right')
ax.set_xlim(0, 6)

plt.tight_layout()
plt.show()

print("ПУНКТ 1.4: СРАВНЕНИЕ РЕЗУЛЬТАТОВ ВРЕМЕННОЙ СВЕРТКИ И ЧАСТОТНОГО РАСЧЕТА")

transient_component = s_out_conv - s_out_spectral
exact_transient = -(A_in / (1.0 + (omega0 * RC)**2)) * np.exp(-t_sig / RC)

print("Сопоставление амплитуд и характеристик:")
print(f"  Амплитуда по свертке (t >= 8 мс):          {A_out_conv:.6f} В")
print(f"  Амплитуда по частотной характеристике:     {H_f0_mod:.6f} В")
print(f"  Погрешность согласования амплитуд:         {abs(A_out_conv - H_f0_mod):.2e} В")
print(f"  Длительность переходного процесса t_пп:    ~ 3*RC = {3*RC*1e3:.2f} мс")

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(t_sig * 1e3, s_out_conv, 'b', lw=2, label=r'Временной метод: свертка $s_1(t) * h(t)$')
axs[0].plot(t_sig * 1e3, s_out_spectral, 'r--', lw=1.5, label='Частотный метод: установившийся режим')
axs[0].set_ylabel('Амплитуда, В')
axs[0].set_title('Сравнение временного и частотного методов расчета отклика RC-цепи')
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc='lower right')

axs[1].plot(t_sig * 1e3, transient_component, 'g', lw=2, label=r'Разность $(s_{conv} - s_{spectral})$ — переходный процесс')
axs[1].plot(t_sig * 1e3, exact_transient, 'k--', lw=1.2, label=r'Аналитическая экспонента $-\frac{A}{1+(\omega_0 RC)^2}e^{-t/RC}$')
axs[1].axvline(3*RC*1e3, color='orange', linestyle='--', lw=1.5, label=r'$t = 3RC$ (затухание на 95%)')
axs[1].set_ylabel('Амплитуда, В')
axs[1].set_title('Свободная затухающая составляющая переходного процесса')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc='lower right')

axs[2].plot(t_sig * 1e3, s_out_conv, 'b', lw=2.5, label='Свертка')
axs[2].plot(t_sig * 1e3, s_out_spectral, 'r--', lw=1.5, label='Частотный расчет')
axs[2].set_ylabel('Амплитуда, В')
axs[2].set_xlabel('Время t, мс')
axs[2].set_title('Точное совпадение сигналов в установившемся режиме ($t \geq 3RC$)')
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].legend(loc='upper right')
axs[2].set_xlim(0, 12)

plt.tight_layout()
plt.show()