import numpy as np
import matplotlib.pyplot as plt
import Integrator

def klein_to_hyperboloid(x):
    t = 1 / np.sqrt(1 - np.sum(x*x, axis=0))
    return np.concatenate((t[np.newaxis, :], t * x))

def hyperboloid_to_klein(x):
    return x[1:, :] / x[0, :]

def translate(a):
    return np.array([
        [a[0], -a[1], -a[2]],
        [-a[1], a[1] * a[1] / (a[0] + 1) + 1, a[1] * a[2] / (a[0] + 1)],
        [-a[2], a[1] * a[2] / (a[0] + 1), a[2] * a[2] / (a[0] + 1) + 1]
    ])

def rotate(a):
    s = 1.0 / np.sqrt(a[1] * a[1] + a[2] * a[2])
    return np.array([
        [1, 0, 0],
        [0, a[2] * s, -a[1] * s],
        [0, a[1] * s, a[2] * s]
    ])

def shift(a):
    s = 1.0 / np.sqrt(1 + a[1] * a[1])
    return np.array([
        [a[0] * s, 0, -a[2] * s],
        [0, 1, 0],
        [-a[2] * s, 0, a[0] * s]
    ])

def orthomap(a):
    A = translate(a[:, 0])
    A = rotate(A @ a[:, n - 1]) @ A
    return shift(A @ x[:, -1]) @ A

def _V(x, a):
    y = klein_to_hyperboloid(x)
    b = klein_to_hyperboloid(a)

    A = orthomap(b)
    y = hyperboloid_to_klein(A @ y)


def distance(x, y):
    z = x / (1 + np.sqrt(1 - x @ x))
    w = y / (1 + np.sqrt(1 - y @ y))
    return np.arccosh(1.0 + 2.0 * (z - w) @ (z - w) / ((1 - z @ z) * (1 - w @ w)))

a0 = np.array([-0.5, -0.5])
a1 = np.array([0.5, -0.5])
a2 = np.array([0.2, 0.5])

n = 10
integrator = Integrator.Integrator(n)
x = a0[:, np.newaxis] + np.array([a1 - a0, a2 - a0]).T @ integrator.vertices
y = map_triangle(x, n)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(y[0, :], y[1, :], 0, s=.4)
ax.scatter(y[0, 0], y[1, 0], 0, "r")
ax.scatter(y[0, n-1], y[1, n-1], 0, "g")
ax.scatter(y[0, -1], y[1, -1], 0, "b")
ax.legend()
plt.show()
