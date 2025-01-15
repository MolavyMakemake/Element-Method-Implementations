import numpy as np
import matplotlib.pyplot as plt
from hyperbolic import Integrator

def _vec_delta(x, u):
    w = x - u[:, np.newaxis]
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * w2 / ((1 - x2) * (1 - u @ u))

def _V(u, v, x):
    d0 = 1 + 2 * ((u - v) @ (u - v)) / ((1 - u @ u) * (1 - v @ v))
    d1 = _vec_delta(x, u)
    d2 = _vec_delta(x, v)

    xu = x - u[:, np.newaxis]
    xv = x - v[:, np.newaxis]

    xu2 = np.sum(xu * xu, axis=0)
    xv2 = np.sum(xv * xv, axis=0)
    x2 = np.sum(x * x, axis=0)

    s1 = 4.0 / ((1 - x2) * (1 - u @ u) * np.sqrt(d1 * d1 - 1))
    s2 = 4.0 / ((1 - x2) * (1 - v @ v) * np.sqrt(d2 * d2 - 1))

    Dd1 = (s1 * (xu[0, :] + x[0, :] * xu2 / (1 - x2)),
           s1 * (xu[1, :] + x[1, :] * xu2 / (1 - x2)))

    Dd2 = (s2 * (xv[0, :] + x[0, :] * xv2 / (1 - x2)),
           s2 * (xv[1, :] + x[1, :] * xv2 / (1 - x2)))

    A = 1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2
    B = 1 + d0 + d1 + d2

    a1 = -d1 * Dd1[0] - d2 * Dd2[0] + d0 * d1 * Dd2[0] + d0 * Dd1[0] * d2
    a2 = -d1 * Dd1[1] - d2 * Dd2[1] + d0 * d1 * Dd2[1] + d0 * Dd1[1] * d2
    b1 = Dd1[0] + Dd2[0]
    b2 = Dd1[1] + Dd2[1]

    V = np.sqrt(A) / B
    DV = (2 * V / (1 + V * V) * (a1 / A - b1 / B),
       2 * V / (1 + V * V) * (a2 / A - b2 / B))

    vol = 2 * np.arctan(V)
    M = np.max(vol)

    return vol / M, DV[0] / M, DV[1] / M

_int = Integrator.Integrator(50, open=True)

p0 = np.array([0, 0])
p1 = np.array([.5, 0])
p2 = np.array([-.5, .5])

v0 = 2 * p0 / (1 + p0 @ p0)
v1 = 2 * p1 / (1 + p1 @ p1)
v2 = 2 * p2 / (1 + p2 @ p2)

A = np.array([v1 - v0, v2 - v0]).T
F = lambda x: v0[:, np.newaxis] + A @ x
phi = lambda x: x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

X = phi(F(_int.vertices))
a, b, c = _V(p1, p2, X)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(X[0, :], X[1, :], b, s=.4)
ax.legend()
plt.show()
