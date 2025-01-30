import numpy as np
import matplotlib.pyplot as plt

def norm(u, x):
    v0 = (1 - x[1] * x[1]) * u[0] + x[0] * x[1] * u[1]
    v1 = (1 - x[0] * x[0]) * u[1] + x[0] * x[1] * u[0]
    return np.sqrt(v0 * u[0] + v1 * u[1]) / (1 - x @ x)

def trace(a, b, step=0.00001):
    f = lambda x: x[0] * np.power((1 - x[1] * x[1]) / ((1 - x[0] * x[0]) * (1 - x @ x)), .25)

    T = []
    F = []

    t = 0
    while True:
        x = a + t * (b - a)

        T.append(t)
        F.append(f(x))
        print(t)

        t += step / norm(b - a, x)
        if t > 1:
            return T, F

x0 = np.array([0, -0.1])
x1 = np.array([0, 0.99])
x2 = np.array([0.5, 0])


t1, f1 = trace(x1, x2)
t2, f2 = trace(x0, x2)

plt.plot(t1, f1)
plt.plot(t2, f2)
plt.show()