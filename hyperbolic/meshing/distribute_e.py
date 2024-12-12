import matplotlib.pyplot as plt
import numpy as np

N = 100
R = 1.0
i = np.arange(N)

t = np.pi * (3 - np.sqrt(5)) * i
r = np.sqrt(i / N) * R

plt.scatter(r * np.cos(t), r * np.sin(t), s=.5)

t = np.linspace(0, 2 * np.pi)
plt.plot(R * np.cos(t), R * np.sin(t), "--", color="gray", linewidth=.5)
plt.axis("equal")
plt.show()
