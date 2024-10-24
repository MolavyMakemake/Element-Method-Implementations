import numpy as np
import matplotlib.pyplot as plt

def radius(p, q):
    a = np.tan(np.pi * (0.5 - 1.0 / q))
    b = np.tan(np.pi / p)
    return np.sqrt((a - b) / (a + b))

# must satisfy (p - 2) * (q - 2) > 4
p = 3
q = 7

r = radius(p, q)

angle = np.linspace(0, 2 * np.pi * (1 - 1 / p), p)
X = r * np.cos(angle)
Y = r * np.sin(angle)

plt.scatter(X, Y)
plt.axis("equal")
plt.show()
