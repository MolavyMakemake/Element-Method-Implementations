import numpy as np
from Integrator import Integrator
import matplotlib.pyplot as plt

def _vec_delta(x, u):
    w = x - u[:, np.newaxis]
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * w2 / ((1 - x2) * (1 - u @ u))

def _V(u, v, x):
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

    a1 = -d1 * Dd1[0] - d2 * Dd2[0] + d0 * d1 * Dd2[0] + d0 * Dd1[0] * d2
    a2 = -d1 * Dd1[1] - d2 * Dd2[1] + d0 * d1 * Dd2[1] + d0 * Dd1[1] * d2
    b1 = Dd1[0] + Dd2[0]
    b2 = Dd1[1] + Dd2[1]

    V = np.sqrt(A) / B
    DV = (2 * V / (1 + V * V) * (a1 / A - b1 / B),
       2 * V / (1 + V * V) * (a2 / A - b2 / B))

    vol = 2 * np.arctan(V)
    M = np.max(vol)

    return vol / M, np.array(DV) / M

def dot(u, v, x):
    r = x * x

    g11 = 1 - r[1, :]
    g12 = x[0, :] * x[1, :]
    g22 = 1 - r[0, :]

    g_u = np.array([g11 * u[0, :] + g12 * u[1, :], g12 * u[0, :] + g22 * u[1, :]])
    g_v = np.array([g11 * v[0, :] + g12 * v[1, :], g12 * v[0, :] + g22 * v[1, :]])

    V = np.sum(u * g_v, axis=0)
    d1V =  -g_u[0, :] - g_v[0, :] \
           + u[0, :] * x[1, :] * v[1, :] \
           + u[1, :] * (x[1, :] * v[0, :] - 2 * x[0, :] * v[1, :])
    d2V = -g_u[1, :] - g_v[1, :] \
          - u[0, :] * (x[0, :] * v[1, :] - 2 * x[1, :] * v[0, :]) \
          + u[1, :] * x[0, :] * v[0, :]

    return V, d1V, d2V

def _V_ideal(A, B, x):
    u = A[:, np.newaxis] - x
    v = B[:, np.newaxis] - x

    uv, d1uv, d2uv = dot(u, v, x)
    uu, d1uu, d2uu = dot(u, u, x)
    vv, d1vv, d2vv = dot(v, v, x)

    dnm = np.sqrt(uu * vv)
    p = uv / dnm
    d1p = d1uv / dnm - .5 * p / dnm * (d1uu * vv + uu * d1vv)
    d2p = d2uv / dnm - .5 * p / dnm * (d2uu * vv + uu * d2vv)

    V = np.pi - np.arccos(p)
    c = np.sqrt(1 - p * p) * V[0]
    return V / V[0], np.array([d1p, d2p]) / c


def _RHS_V(Y, d0, v1, v2, b1, b2):
    Y2 = np.sum(Y * Y, axis=0)
    print(np.max(Y2))
    d1 = 1.0 + 2.0 * np.sum(v1 * v1, axis=0) / ((1.0 - Y2) * b1)
    d2 = 1.0 + 2.0 * np.sum(v2 * v2, axis=0) / ((1.0 - Y2) * b2)

    Dd1 = (2 * (d1 - 1) * Y[0, :] / (1 - Y2) + 4 * v1[0] / ((1 - Y2) * b1),
           2 * (d1 - 1) * Y[1, :] / (1 - Y2) + 4 * v1[1] / ((1 - Y2) * b1))

    Dd2 = (2 * (d2 - 1) * Y[0, :] / (1 - Y2) + 4 * v2[0] / ((1 - Y2) * b2),
           2 * (d2 - 1) * Y[1, :] / (1 - Y2) + 4 * v2[1] / ((1 - Y2) * b2))

    A = np.abs(1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2)
    B = 1 + d0 + d1 + d2

    a1 = -d1 * Dd1[0] - d2 * Dd2[0] + d0 * d1 * Dd2[0] + d0 * Dd1[0] * d2
    a2 = -d1 * Dd1[1] - d2 * Dd2[1] + d0 * d1 * Dd2[1] + d0 * Dd1[1] * d2
    b1 = Dd1[0] + Dd2[0]
    b2 = Dd1[1] + Dd2[1]

    V = np.sqrt(A) / B
    DV = (2 * V / (1 + V * V) * (a1 / A - b1 / B),
          2 * V / (1 + V * V) * (a2 / A - b2 / B))

    vol = 2 * np.arctan(V)
    M = vol[0]

    return vol / M, np.array(DV) / M

def _RHS_bdry_int(a, b, t, dt):
    u1 = a[:, 1] - a[:, 0]
    u2 = a[:, 2] - a[:, 0]

    b1 = 1.0 - b[:, 1] @ b[:, 1]
    b2 = 1.0 - b[:, 2] @ b[:, 2]

    d0 = 1.0 + 2.0 * (b[:, 2] - b[:, 1]) @ (b[:, 2] - b[:, 1]) / (b1 * b2)

    X = a[:, 0, np.newaxis] + np.outer(u1, t)
    X2 = X * X

    c = np.sqrt(1.0 - np.sum(X2, axis=0))
    Y = X / (1.0 + c)

    v1 = Y - b[:, 1, np.newaxis]
    v2 = Y - b[:, 2, np.newaxis]

    #V, DV = _RHS_V(Y, d0, v1, v2, b1, b2)
    V, DV = _V_ideal(a[:, 1], a[:, 2], X)

    s = (1 + c) * (1 + c) * c
    g11 = 1.0 - X2[0, :]
    g12 = -X[0, :] * X[1, :]
    g22 = 1.0 - X2[1, :]

    dV = np.array([
        DV[0, :] * (1 / (1 + c) + X2[0, :] / s) - DV[1, :] * g12 / s,
        DV[1, :] * (1 / (1 + c) + X2[1, :] / s) - DV[0, :] * g12 / s,
    ])
    s_dV = np.array([
        -g12 * dV[0, :] - g22 * dV[1, :],
        g11 * dV[0, :] + g12 * dV[1, :]
    ]) / c

    I_v = V * (u1 @ s_dV)
    I = np.sum(I_v * dt) - .5 * I_v[0] * dt

    plt.plot(t, I_v)
    plt.show()
    ###

    X = a[:, 0, np.newaxis] + np.outer(u2, t)
    X2 = X * X

    c = np.sqrt(1.0 - np.sum(X2, axis=0))
    Y = X / (1.0 + c)

    v1 = Y - b[:, 1, np.newaxis]
    v2 = Y - b[:, 2, np.newaxis]

    V, DV = _V_ideal(a[:, 1], a[:, 2], X)

    s = (1 + c) * (1 + c) * c
    g11 = 1.0 - X2[0, :]
    g12 = -X[0, :] * X[1, :]
    g22 = 1.0 - X2[1, :]

    dV = np.array([
        DV[0, :] * (1 / (1 + c) + X2[0, :] / s) - DV[1, :] * g12 / s,
        DV[1, :] * (1 / (1 + c) + X2[1, :] / s) - DV[0, :] * g12 / s,
    ])
    s_dV = np.array([
        -g12 * dV[0, :] - g22 * dV[1, :],
        g11 * dV[0, :] + g12 * dV[1, :]
    ]) / c

    I_v = V * (u2 @ s_dV)
    I -= np.sum(I_v * dt) - .5 * I_v[0] * dt

    return np.abs(I)

def RHS(a, N):
    dt = 1.0 / N
    t = np.linspace(0, 1 - dt, N)

    b = a / (1.0 + np.sqrt(1.0 - np.sum(a * a, axis=0)))

    I0 = _RHS_bdry_int(np.roll(a, -0, axis=1), np.roll(b, -0, axis=1), t, dt)
    #I1 = _RHS_bdry_int(np.roll(a, -1, axis=1), np.roll(b, -1, axis=1), t, dt)
    #I2 = _RHS_bdry_int(np.roll(a, -2, axis=1), np.roll(b, -2, axis=1), t, dt)

    return I0, 0, 0

u = np.array([-0.8, .0])
w = np.array([.2, -0.6])
v = np.array([.0, -0.9])

N_samples = []
I_arr = []
J_arr = []

for N in [100, 200, 400, 1000]:
    print("N =", N)

    N_t = N * (N + 1) // 4
    dt = 1 / N_t
    t = np.linspace(0, 1 - dt, N_t)

    integrator = Integrator(N, open=True)
    A = np.array([v - u, w - u]).T
    X = u[:, np.newaxis] + A @ integrator.vertices
    Jac_A = np.abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

    Y = X / (1.0 + np.sqrt(1 - np.sum(X * X, axis=0)))

    path_uv = u[:, np.newaxis] + np.outer(v - u, t)
    path_uw = np.array([u[0] + t * (w[0] - u[0]), u[1] + t * (w[1] - u[1])])
    path_vw = np.array([v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1])])

    #V0, DV0 = _V(v, w, Y)
    #V1, DV1 = _V(u, w, Y)
    #V2, DV2 = _V(u, v, Y)

    g = np.square(1 - np.sum(Y * Y, axis=0)) / 4.0
    dv = np.power(1 - np.sum(X * X, axis=0), -1.5)

    #I = Jac_A * integrator.integrate_vector(
    #    g * np.sum(DV0 * DV0, axis=0) * dv
    #)

    J0, J1, J2 = RHS(np.array([u, v, w]).T, N_t)

    N_samples.append(N * (N + 1) // 2)
    #I_arr.append(I)
    J_arr.append(J0)

#plt.plot(N_samples, I_arr, "o--", label="interior integral")
plt.plot(N_samples, J_arr, "o--", label="path integral")

plt.xlabel("nr. samples")
plt.ylabel("out")
plt.legend()

plt.show()
