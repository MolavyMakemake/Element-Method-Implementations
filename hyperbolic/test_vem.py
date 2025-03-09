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

def compute_area(v, t, dt):
    dv = np.roll(v, -1, axis=1) - v

    I = 0
    for i in range(np.size(v, axis=1)):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)
        r = X * X
        c = np.sqrt(1.0 - np.sum(r, axis=0))

        x = X[0, :] / (1 + c)
        dy = np.array([
            X[0, :] * X[1, :] / ((1 + c) * (1 + c) * c),
            1 / (1 + c) + r[1, :] / ((1 + c) * (1 + c) * c)
        ])

        I_v = dv[:, i] @ dy * x
        I += np.sum(I_v * dt) - .5 * (I_v[0] + I_v[-1]) * dt

    return I

def proj_RHS(v, t, dt):
    v_1 = np.roll(v, -1, axis=1)
    dv = v_1 - v

    N_v = np.size(v, axis=1)
    I = np.zeros(shape=(2, N_v), dtype=float)

    for i in range(N_v):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)
        r = X * X
        c = np.sqrt(1.0 - np.sum(r, axis=0))

        Y = X / (1 + c)

        dx = np.array([
            1 / (1 + c) + Y[0, :] * Y[0, :] / c,
            Y[0, :] * Y[1, :] / c
        ])
        dy = np.array([
            Y[0, :] * Y[1, :] / c,
            1 / (1 + c) + Y[1, :] * Y[1, :] / c
        ])

        e = np.arccosh(_vec_delta(Y, v[:, i] / (1 + np.sqrt(1 - v[:, i] @ v[:, i]))))
        e /= e[-1]

        I0_v = dv[:, i] @ dy * (1 - e)
        I1_v = -dv[:, i] @ dx * (1 - e)

        I[0, i] += np.sum(I0_v * dt) - .5 * (I0_v[0] + I0_v[-1]) * dt
        I[1, i] += np.sum(I1_v * dt) - .5 * (I1_v[0] + I1_v[-1]) * dt

        j = (i + 1) % N_v
        J0_v = dv[:, i] @ dy * e
        J1_v = -dv[:, i] @ dx * e

        I[0, j] += np.sum(J0_v * dt) - .5 * (J0_v[0] + J0_v[-1]) * dt
        I[1, j] += np.sum(J1_v * dt) - .5 * (J1_v[0] + J1_v[-1]) * dt

    return I

def compute_stabilizer(v, v0, v1, a0, a1, t, dt):
    v_1 = np.roll(v, -1, axis=1)
    dv = v_1 - v

    N_v = np.size(v, axis=1)
    I = np.zeros(shape=(2, N_v), dtype=float)

    for i in range(N_v):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)
        r = X * X
        c = np.sqrt(1.0 - np.sum(r, axis=0))

        Y = X / (1 + c)

        dx = np.array([
            1 / (1 + c) + Y[0, :] * Y[0, :] / c,
            Y[0, :] * Y[1, :] / c
        ])
        dy = np.array([
            Y[0, :] * Y[1, :] / c,
            1 / (1 + c) + Y[1, :] * Y[1, :] / c
        ])

        e = np.arccosh(_vec_delta(Y, v[:, i] / (1 + np.sqrt(1 - v[:, i] @ v[:, i]))))
        e /= e[-1]

        I0_v = dv[:, i] @ dy * e
        I1_v = -dv[:, i] @ dx * e

        I[0, i] += np.sum(I0_v * dt) - .5 * (I0_v[0] + I0_v[-1]) * dt
        I[1, i] += np.sum(I1_v * dt) - .5 * (I1_v[0] + I1_v[-1]) * dt

        j = (i + 1) % N_v
        J0_v = dv[:, i] @ dy * (1 - e)
        J1_v = -dv[:, i] @ dx * (1 - e)

        I[0, j] += np.sum(J0_v * dt) - .5 * (J0_v[0] + J0_v[-1]) * dt
        I[1, j] += np.sum(J1_v * dt) - .5 * (J1_v[0] + J1_v[-1]) * dt

    return I

u = np.array([
    [-0.5, 0],
    [0, 0.5],
    [0.5, 0.2]
]).T

N_samples = []
J_arr = []

for N in [100]:
    print("N =", N)

    N_t = N ** 2
    dt = 1 / N_t
    t = np.linspace(0, 1, N_t + 1)

    N_samples.append(N)
    J_arr.append(compute_area(u, t, dt))

    X = Integrator(100).vertices
    X = u[:, 0, np.newaxis] + np.array([
        u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]
    ]).T @ X

    area = compute_area(u, t, dt)
    a = proj_RHS(u, t, dt) / np.sqrt(2) / area

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for i in range(np.size(u, axis=1)):
        ax.scatter(u[0, i], u[1, i], s=0.6)

    ax.scatter(u[0, 2], u[1, 2], s=5.6)

    Y = a[0, 2] * X[0, :] + a[1, 2] * X[1, :]
    ax.scatter(X[0, :], X[1, :], Y - np.average(Y), color="b", alpha=0.5, s=.4)
    ax.legend()
    plt.show()