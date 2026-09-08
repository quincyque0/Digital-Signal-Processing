import numpy as np 
import math 
import matplotlib.pyplot as plt

x = np.arange(0,10,0.1)
# x = np.linspace(0,10,100)
y = np.sin(x)

plt.plot(x,y,label="signal",color="black",linewidth=2)
plt.xlabel("x")
plt.ylabel("y")

plt.legend()
plt.grid(True)

plt.show()
