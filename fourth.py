    import numpy as np
    import matplotlib.pyplot as plt

    def fourier_integral(t, s, f_grid):
        dt = t[1] - t[0]
        kernel = np.exp(-1j * 2 * np.pi * np.outer(f_grid, t))
        S = np.trapz(s * kernel, dx=dt, axis=1)
        return S

    def rect_pulse(t, tau, t0=0.0):
        return np.where(np.abs(t - t0) <= tau / 2.0, 1.0, 0.0)

    dt = 0.001
    t = np.arange(-5.0, 5.0, dt)    

    tau_base = 2.0
    f_max_base = 3.0 / tau_base 
    f_grid_base = np.linspace(0, f_max_base, 600)


    s2 = rect_pulse(t, tau=tau_base)
    S2 = fourier_integral(t, s2, f_grid_base)

    test_freqs = np.array([0.0, 0.25, 0.5]) 
    S_test_num = fourier_integral(t, s2, test_freqs)
    S_test_analytical = tau_base * np.sinc(test_freqs * tau_base)

    for f_val, num_val, ana_val in zip(test_freqs, S_test_num, S_test_analytical):
        print(f"f = {f_val:4.2f} Гц | Численно: {num_val.real:+.4f} + {num_val.imag:+.4f}j "
            f"(Модуль: {np.abs(num_val):.4f}) | Аналитически: {ana_val:+.4f}")

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(f_grid_base, np.abs(S2), 'b', label='Численный |S(f)|')
    ax[0].plot(test_freqs, np.abs(S_test_num), 'ro', label='Точки проверки (числ.)')
    ax[0].set_title('Пункт 2: ПФ прямоугольного импульса (tau = 2 c)')
    ax[0].set_ylabel('|S(f)|, В·с')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(f_grid_base, np.angle(S2), 'm', label='arg S(f)')
    ax[1].set_ylabel('Фаза, рад')
    ax[1].set_xlabel('Частота f, Гц')
    ax[1].set_yticks([-np.pi, 0, np.pi])
    ax[1].set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax[1].grid(True)
    ax[1].legend()
    plt.tight_layout()
    plt.show()


    tau_half = tau_base / 2.0  
    tau_double = tau_base * 2.0  

    s3_base = rect_pulse(t, tau_base)
    s3_half = rect_pulse(t, tau_half)
    s3_double = rect_pulse(t, tau_double)


    f_grid_scale = np.linspace(0, 3.0, 800)

    S3_base = fourier_integral(t, s3_base, f_grid_scale)
    S3_half = fourier_integral(t, s3_half, f_grid_scale)
    S3_double = fourier_integral(t, s3_double, f_grid_scale)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(f_grid_scale, np.abs(S3_base), 'b', label=f'tau = {tau_base} c')
    ax[0].plot(f_grid_scale, np.abs(S3_half), 'g--', label=f'tau = {tau_half} c (сжатие во времени)')
    ax[0].plot(f_grid_scale, np.abs(S3_double), 'r-.', label=f'tau = {tau_double} c (растяжение во времени)')
    ax[0].set_title('Пункт 3: Свойство изменения длительности')
    ax[0].set_ylabel('|S(f)|, В·с')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(f_grid_scale, np.angle(S3_base), 'b', label=f'tau = {tau_base} c')
    ax[1].plot(f_grid_scale, np.angle(S3_half), 'g--', label=f'tau = {tau_half} c')
    ax[1].plot(f_grid_scale, np.angle(S3_double), 'r-.', label=f'tau = {tau_double} c')
    ax[1].set_ylabel('Фаза, рад')
    ax[1].set_xlabel('Частота f, Гц')
    ax[1].grid(True)
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    tau_0 = 1.0  
    s4_orig = rect_pulse(t, tau=tau_base, t0=0.0)
    s4_delayed = rect_pulse(t, tau=tau_base, t0=tau_0)

    S4_orig = fourier_integral(t, s4_orig, f_grid_base)
    S4_delayed = fourier_integral(t, s4_delayed, f_grid_base)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(f_grid_base, np.abs(S4_orig), 'b', label='Исходный t0=0')
    ax[0].plot(f_grid_base, np.abs(S4_delayed), 'r--', label=f'Задержанный tau0={tau_0} с')
    ax[0].set_title('Пункт 4: Свойство сдвига (запаздывания) во времени')
    ax[0].set_ylabel('|S(f)|, В·с')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(f_grid_base, np.unwrap(np.angle(S4_orig)), 'b', label='arg S_исх(f)')
    ax[1].plot(f_grid_base, np.unwrap(np.angle(S4_delayed)), 'r--', label='arg S_зад(f) = arg S_исх - 2*pi*f*tau0')
    ax[1].set_ylabel('Фаза (unwrap), рад')
    ax[1].set_xlabel('Частота f, Гц')
    ax[1].grid(True)
    ax[1].legend()
    plt.tight_layout()
    plt.show()


    f0 = 2.0  
    carrier = np.cos(2 * np.pi * f0 * t)
    s5_rect = rect_pulse(t, tau=tau_base)
    s5_mod = s5_rect * carrier
    f_grid_mod = np.linspace(0, f0 + 2.0, 1000)

    S5_rect = fourier_integral(t, s5_rect, f_grid_mod)
    S5_mod = fourier_integral(t, s5_mod, f_grid_mod)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(f_grid_mod, np.abs(S5_rect), 'b', label='Исходный видеоимпульс')
    ax[0].plot(f_grid_mod, np.abs(S5_mod), 'r', label=f'Радиоимпульс (f0 = {f0} Гц)')
    ax[0].set_title('Пункт 5: Свойство модуляции (теорема смещения)')
    ax[0].set_ylabel('|S(f)|, В·с')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(f_grid_mod, np.angle(S5_rect), 'b', alpha=0.6, label='Фаза видеоимпульса')
    ax[1].plot(f_grid_mod, np.angle(S5_mod), 'r', alpha=0.8, label='Фаза радиоимпульса')
    ax[1].set_ylabel('Фаза, рад')
    ax[1].set_xlabel('Частота f, Гц')
    ax[1].grid(True)
    ax[1].legend()
    plt.tight_layout()
    plt.show()