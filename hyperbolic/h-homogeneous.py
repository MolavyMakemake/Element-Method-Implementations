import numpy as np
import matplotlib.pyplot as plt
import Integrator

def _midpoint_bkdisk(x, y):
    _a = (x - y) @ (x - y)
    _b = x @ x - x @ y
    _c = 1 - x @ x

    _d = np.sqrt(_b * _b + _a * _c)
    t0 = (_b + _d) / _a
    t1 = (_b - _d) / _a

    A = x + t0 * (y - x)
    B = x + t1 * (y - x)

    Ax = np.sqrt((x - A) @ (x - A))
    Ay = np.sqrt((y - A) @ (y - A))
    AB = np.sqrt((B - A) @ (B - A))

    ab = np.sqrt(Ax * Ay)
    t = ab / (np.sqrt((AB - Ax) * (AB - Ay)) + ab)

    return A + t * (B - A)

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
    A = translate(a[:, 1])
    A = rotate(A @ a[:, 2]) @ A
    return shift(A @ a[:, 0]) @ A

def centroid(a):
    m1 = _midpoint_bkdisk(a[:, 0], a[:, 2])
    m2 = _midpoint_bkdisk(a[:, 0], a[:, 1])

    return a[:, 1] + np.linalg.solve(
        np.array([m1 - a[:, 1], m2 - a[:, 2]]).T,
        a[:, 2] - a[:, 1]
    )[0] * (m1 - a[:, 1])

def _V(x, a):
    z = klein_to_hyperboloid(x)
    b = klein_to_hyperboloid(a)

    A = orthomap(b)
    z = A @ z
    y = hyperboloid_to_klein(z)

    r = y * y
    S = np.power((1 - r[1, :]) / ((1 - r[0, :]) * (1 - r[0, :] - r[1, :])), .25)

    f = y[0, :] * S
    m = np.min(f)
    M = np.max(f)
    M = M if M > -m else m

    return f / M

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
y1 = _V(x, np.roll(np.array([a0, a1, a2]).T, 0, axis=1))
y2 = _V(x, np.roll(np.array([a0, a1, a2]).T, 1, axis=1))
y3 = _V(x, np.roll(np.array([a0, a1, a2]).T, 2, axis=1))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(x[0, :], x[1, :], y1 + y2 + y3, s=.4)
ax.scatter(x[0, 0], x[1, 0], y1[0], color="r")
ax.scatter(x[0, n-1], x[1, n-1], y1[n-1], color="g")
ax.scatter(x[0, -1], x[1, -1], y1[-1], color="b")
ax.legend()
plt.show()
