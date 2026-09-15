import matplotlib.pyplot as plt
import numpy as np

f = 2  
t = np.linspace(0, 5 / f, 1000)  

plt.figure(figsize=(12, 8))
N_values = [1, 2, 3, 6] 

for idx, N in enumerate(N_values):
    x = np.zeros_like(t)
    for n in range(1, N + 1):
        k = 2 * n - 1  
        x += (4 / (k * np.pi)) * np.cos(2 * np.pi * k * f * t - np.pi / 2)
    
    plt.subplot(2, 2, idx + 1)
    plt.plot(t, x)
    plt.title(f'Число гармоник: {N}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

plt.tight_layout()
plt.show()