import numpy as np 
import math 
import matplotlib.pyplot as plt


A = 16
fe = math.pi/2
f = 10

t2 = np.linspace(0,1,15)
t = np.linspace(0,1,100)

y = A * np.sin(math.pi*t * f + fe)  + A*2 * np.sin(math.pi*t * f*2 + fe) + np.random.normal(size = 100,scale=15,loc = 0)
y2 = A * np.sin(math.pi*t2 * f + fe)  + A*2 * np.sin(math.pi*t2 * f*2 + fe) + np.random.normal(size = 15,scale = 15,loc = 0)

yd= []
yd2 =[0,0,0,0,0,0,1,1,1,1,0,0,0,0,0]
for i in y:
    if i<sum(y)/len(y):
        yd.append(0)
    else:
        yd.append(1)

# for i in y2:
#     if i<sum(y2)/len(y2):
#         yd2.append(0)
#     else:
#         yd2.append(1)
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(16, 10))

ax1.plot(t, yd, label="signal", color="black", linewidth=2)
ax1.legend()
ax1.grid(True)

ax2.stem(t, yd, label="signal", markerfmt="blue", basefmt="blue")
ax2.legend()
ax2.grid(True)

ax3.plot(t2, yd2, label="signal", color="black", linewidth=2)
ax3.legend()
ax3.grid(True)

ax4.stem(t2, yd2, label="signal", markerfmt="blue", basefmt="blue")
ax4.legend()
ax4.grid(True)
plt.tight_layout()

plt.show()
