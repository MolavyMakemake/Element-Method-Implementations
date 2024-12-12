import matplotlib.pyplot as plt
import numpy as np

N = 100
i = np.arange(N)

t = np.pi * (3 - np.sqrt(5)) * i
r = np.sqrt(i / N)

plt.scatter(r * np.cos(t), r * np.sin(t), s=.5)
plt.axis("equal")
plt.show()
