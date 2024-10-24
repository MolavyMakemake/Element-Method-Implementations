import numpy as np
import matplotlib.pyplot as plt

def d(x, y, w):
    a = x - w[0]
    b = y - w[1]
    return 1 + 2 * (a*a + b*b) / ((1 - x*x - y*y) * (1 - w @ w))

def A_tri(x, y, u, v):
    d0 = d(u[0], u[1], v)
    d1 = d(x, y, u)
    d2 = d(x, y, v)

    A = 1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2
    B = 1 + d0 + d1 + d2

    return 2 * np.atan(np.sqrt(A) / B)


e = np.array([1, 0])
T = 0.5
t = np.linspace(0, T)

u0 = np.array([0, 0])
v1 = np.array([.1, .7])
v2 = np.array([-.3, .5])

plt.plot(t, A_tri(t, 0, u0, v1) / A_tri(T, 0, u0, v1))
plt.plot(t, A_tri(t, 0, u0, v2) / A_tri(T, 0, u0, v2))
plt.show()