import numpy as np
import matplotlib.pyplot as plt


A = 4.0             
omega_0 = 0.4 * np.pi    
f0 = omega_0 / (2 * np.pi) 
phi = 0.25 * np.pi  


T = 1.0 / f0       
delta_t = -phi / omega_0 

print(f"Период T = {T:.3f} с")
print(f"Частота f0 = {f0:.3f} Гц")
print(f"Временной сдвиг максимума delta_t = {delta_t:.3f} с (сдвиг влево на {-delta_t:.3f} с)\n")

t = np.linspace(-1, 2.5 * T, 1000)
x_shifted = A * np.cos(omega_0 * t + phi)
x_zero_phase = A * np.cos(omega_0 * t)


plt.figure(figsize=(10, 5))
plt.plot(t, x_shifted, label=r'$x(t) = 4\cos(0.4\pi t + 0.25\pi)$ (со сдвигом)', color='blue', linewidth=2)
plt.plot(t, x_zero_phase, label=r'$x_0(t) = 4\cos(0.4\pi t)$ (нулевая фаза)', color='orange', linestyle='--', linewidth=2)

plt.axvline(delta_t, color='blue', linestyle=':', alpha=0.7, label=f'Максимум x(t): t = {delta_t:.3f} с')
plt.axvline(0, color='orange', linestyle=':', alpha=0.7, label='Максимум x0(t): t = 0.000 с')




test_times = [-1.0, 3.0, 7.0]
for t_val in test_times:
    phi_rad = omega_0 * t_val
    phi_deg = np.degrees(phi_rad)
    phi_wrapped_rad = (phi_rad + np.pi) % (2 * np.pi) - np.pi
    phi_wrapped_deg = np.degrees(phi_wrapped_rad)
    print(f"t = {t_val:4.1f} с -> Фаза: {phi_rad:6.3f} рад ({phi_deg:6.1f}°) "
          f"| В интервале [-π, π]: {phi_wrapped_rad:6.3f} рад ({phi_wrapped_deg:6.1f}°)")


a, b = 3.0, 4.0
z1 = complex(a, b)
z2 = complex(a, -b)

r1, theta1 = abs(z1), np.angle(z1)
r2, theta2 = abs(z2), np.angle(z2)

print(f"z1 = {z1.real:.0f} + {z1.imag:.0f}j -> r = {r1:.2f}, theta = {theta1:.3f} рад ({np.degrees(theta1):.2f}°)")
print(f"z2 = {z2.real:.0f} - {-z2.imag:.0f}j -> r = {r2:.2f}, theta = {theta2:.3f} рад ({np.degrees(theta2):.2f}°)")


r_polar = 6.0
theta_polar = np.pi / 3  
z_cart = r_polar * np.exp(1j * theta_polar)

print(f"Полярная форма: r = {r_polar}, theta = π/3 ({np.degrees(theta_polar):.1f}°)")
print(f"Алгебраическая форма: z = {z_cart.real:.3f} + {z_cart.imag:.3f}j")
plt.title('Сравнение гармонических сигналов', fontsize=14)
plt.xlabel('Время t, с', fontsize=12)
plt.ylabel('Амплитуда x(t)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()