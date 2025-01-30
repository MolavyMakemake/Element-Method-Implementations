import numpy as np
from Integrator import Integrator
import matplotlib.pyplot as plt

def _vec_delta(x, u):
    w = x - u[:, np.newaxis]
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * w2 / ((1 - x2) * (1 - u @ u))

def V(u, v, x):
    u = u / (1.0 + np.sqrt(1 - u @ u))
    v = v / (1.0 + np.sqrt(1 - v @ v))

    d0 = 1 + 2 * ((u - v) @ (u - v)) / ((1 - u @ u) * (1 - v @ v))
    d1 = _vec_delta(x, u)
    d2 = _vec_delta(x, v)

    xu = x - u[:, np.newaxis]
    xv = x - v[:, np.newaxis]

    x2 = np.sum(x * x, axis=0)

    Dd1 = (2 * (d1 - 1) * x[0, :] / (1 - x2) + 4 * xu[0, :] / ((1 - x2) * (1 - u @ u)),
           2 * (d1 - 1) * x[1, :] / (1 - x2) + 4 * xu[1, :] / ((1 - x2) * (1 - u @ u)))

    Dd2 = (2 * (d2 - 1) * x[0, :] / (1 - x2) + 4 * xv[0, :] / ((1 - x2) * (1 - v @ v)),
           2 * (d2 - 1) * x[1, :] / (1 - x2) + 4 * xv[1, :] / ((1 - x2) * (1 - v @ v)))

    A = np.abs(1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2)
    B = 1 + d0 + d1 + d2

    print(np.min(A))
    a1 = -d1 * Dd1[0] - d2 * Dd2[0] + d0 * d1 * Dd2[0] + d0 * Dd1[0] * d2
    a2 = -d1 * Dd1[1] - d2 * Dd2[1] + d0 * d1 * Dd2[1] + d0 * Dd1[1] * d2
    b1 = Dd1[0] + Dd2[0]
    b2 = Dd1[1] + Dd2[1]

    V = np.sqrt(A) / B
    DV = (2 * V / (1 + V * V) * (a1 / A - b1 / B),
       2 * V / (1 + V * V) * (a2 / A - b2 / B))

    vol = 2 * np.arctan(V)
    M = 1.0

    return vol / M, np.array(DV) / M

u = np.array([-.5, .6])
v = np.array([-.6, -0.1])
w = np.array([.5, .2])

N_samples = []
I_arr = []
J_arr = []

for N in [100, 200, 300, 500, 1000]:
    print("N =",N)

    N_t = N * (N + 1) // 4
    dt = 1 / N_t
    t = np.linspace(0, 1 - dt, N_t)

    integrator = Integrator(N, open=True)
    A = np.array([v - u, w - u]).T
    X = u[:, np.newaxis] + A @ integrator.vertices
    Jac_A = np.abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

    Y = X / (1.0 + np.sqrt(1 - np.sum(X * X, axis=0)))

    path_uv = np.array([u[0] + t * (v[0] - u[0]), u[1] + t * (v[1] - u[1])])
    path_uw = np.array([u[0] + t * (w[0] - u[0]), u[1] + t * (w[1] - u[1])])
    path_vw = np.array([v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1])])

    V0, DV0 = V(v, w, Y)
    V1, DV1 = V(u, w, Y)
    V2, DV2 = V(u, v, Y)

    g = np.square(1 - np.sum(Y * Y, axis=0)) / 4.0
    dv = np.power(1 - np.sum(X * X, axis=0), -1.5)

    I = Jac_A * integrator.integrate_vector(
        g * np.sum(DV0 * DV0, axis=0) * dv
    )

    ###

    X = path_uv
    X2 = X * X

    c = np.sqrt(1 - np.sum(X2, axis=0))
    Y = X / (1.0 + c)
    V0, DV0 = V(v, w, Y)

    g11 = 1 - X2[0, :]
    g12 = -X[0, :] * X[1, :]
    g22 = 1 - X2[1, :]
    g = 1.0 / np.sqrt(1 - np.sum(X2, axis=0))

    s = (1 + c) * (1 + c) * c
    dV0 = np.array([
        DV0[0, :] * (1 / (1 + c) + X2[0, :] / s) + DV0[1, :] * X[0, :] * X[1, :] / s,
        DV0[1, :] * (1 / (1 + c) + X2[1, :] / s) + DV0[0, :] * X[0, :] * X[1, :] / s,
    ])
    s_dV0 = g * np.array([
        -g12 * dV0[0, :] - g22 * dV0[1, :],
        g11 * dV0[0, :] + g12 * dV0[1, :]
    ])

    J_v = V0 * ((v - u) @ s_dV0)
    J = np.sum(J_v * dt) - .5 * J_v[0] * dt
    plt.plot(t, J_v)
    plt.show()

    ###

    X = path_uw
    X2 = X * X

    c = np.sqrt(1 - np.sum(X2, axis=0))
    Y = X / (1.0 + c)
    V0, DV0 = V(v, w, Y)

    g11 = 1 - X2[0, :]
    g12 = -X[0, :] * X[1, :]
    g22 = 1 - X2[1, :]
    g = 1.0 / np.sqrt(1 - np.sum(X2, axis=0))

    s = (1 + c) * (1 + c) * c
    dV0 = np.array([
        DV0[0, :] * (1 / (1 + c) + X2[0, :] / s) + DV0[1, :] * X[0, :] * X[1, :] / s,
        DV0[1, :] * (1 / (1 + c) + X2[1, :] / s) + DV0[0, :] * X[0, :] * X[1, :] / s,
    ])
    s_dV0 = g * np.array([
        -g12 * dV0[0, :] - g22 * dV0[1, :],
        g11 * dV0[0, :] + g12 * dV0[1, :]
    ])

    J_v = V0 * ((w - u) @ s_dV0)
    J -= np.sum(J_v * dt) - .5 * J_v[0] * dt

    ###

    N_samples.append(N * (N + 1) // 2)
    I_arr.append(I)
    J_arr.append(J)

plt.plot(N_samples, I_arr, "o--", label="interior integral")
plt.plot(N_samples, J_arr, "o--", label="path integral")

plt.xlabel("nr. samples")
plt.ylabel("out")
plt.legend()

plt.show()
