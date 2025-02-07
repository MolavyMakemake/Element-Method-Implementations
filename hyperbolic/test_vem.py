import numpy as np
from Integrator import Integrator
import matplotlib.pyplot as plt

def _vec_delta(x, u):
    w = x - u[:, np.newaxis]
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * w2 / ((1 - x2) * (1 - u @ u))

def _V(u):
    r = u * u
    t = np.sqrt(1.0 - np.sum(r, axis=0))

    x = u[0, :] / (1 + t)
    dy = np.array([
        u[0, :] * u[1, :] / ((1 + t) * (1 + t) * t),
        1 / (1 + t) + r[1, :] / ((1 + t) * (1 + t) * t)
    ])

    return x, dy

u = np.array([0.0, 0.0])
w = np.array([.99999999, 0.0])
v = np.array([.0, .99999999])

N_samples = []
J_arr = []

for N in [100, 200, 300]:
    print("N =", N)

    N_t = N * (N + 1) // 4
    dt = 1 / N_t
    t = np.linspace(0, 1 - dt, N_t)

    X_uv = u[:, np.newaxis] + np.outer(v - u, t)
    X_vw = v[:, np.newaxis] + np.outer(w - v, t)
    X_wu = w[:, np.newaxis] + np.outer(u - w, t)

    x_uv, dy_uv = _V(X_uv)
    x_vw, dy_vw = _V(X_vw)
    x_wu, dy_wu = _V(X_wu)

    _J_v = (v - u) @ dy_uv * x_uv
    _J_v += (w - v) @ dy_vw * x_vw
    _J_v += (u - w) @ dy_wu * x_wu

    _J = np.sum(_J_v * dt) - .5 * (_J_v[0] + _J_v[-1]) * dt

    N_samples.append(N * (N + 1) // 2)
    J_arr.append(-_J)

A = 1.0 - np.pi / 4
plt.plot([N_samples[0], N_samples[-1]], [A, A])
plt.plot(N_samples, J_arr, "o--", label="path integral")

plt.xlabel("nr. samples")
plt.ylabel("out")
plt.legend()

plt.show()
