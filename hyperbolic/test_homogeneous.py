import numpy as np
from Integrator import Integrator
import matplotlib.pyplot as plt

def V1(x):
    r = x * x
    S = np.power((1 - r[1, :]) / ((1 - r[0, :]) * (1 - r[0, :] - r[1, :])), .25)

    return S * x[0, :], S * np.array([
        1 + .5 * r[0, :] * (1.0 / (1 - r[0, :] - r[1, :]) + 1.0 / (1 - r[0, :])),
        .5 * x[0, :] * x[1, :] * (1 / (1 - r[0, :] - r[1, :]) - 1 / (1 - r[1, :]))
    ])

def V2(x):
    r = x * x
    S = np.power((1 - r[0, :]) / ((1 - r[1, :]) * (1 - r[0, :] - r[1, :])), .25)

    return S * x[0, :], S * np.array([
        1 + .5 * r[0, :] * (1.0 / (1 - r[0, :] - r[1, :]) + 1.0 / (1 - r[0, :])),
        .5 * x[0, :] * x[1, :] * (1 / (1 - r[0, :] - r[1, :]) - 1 / (1 - r[1, :]))
    ])

def W_vec(x):
    r = x * x
    return (1 - r[0, :] - r[1, :]) * np.array([
        x[0, :] / (1 - r[1, :]), x[1, :] / (1 - r[0, :])
    ])

def W(x):
    r = x * x
    return np.array([
        x[0, :] * (1 + r[1, :] / (1 - r[0, :])), x[1, :] * (1 + r[0, :] / (1 - r[1, :]))
    ]) / (1 - r[0, :] - r[1, :])

def star(v, x):
    r = x * x
    g11 = 1 - r[0, :]
    g12 = -x[0, :] * x[1, :]
    g22 = 1 - r[1, :]
    return np.array([
        -g12 * v[0, :] - g22 * v[1, :],
        g11 * v[0, :] + g12 * v[1, :]
    ]) / np.sqrt(1 - r[0, :] - r[1, :])


u = np.array([-.5, .8])
v = np.array([-.8, 0.3])
w = np.array([.5, .2])

N_samples = []
I = []
J = []

for N in [100, 200]:
    print("N =",N)

    N_t = N * (N + 1) // 6
    dt = 1 / N
    t = np.linspace(0, 1, N+1)

    ###

    _int = Integrator(N)
    A = np.array([v - u, w - u]).T
    X = u[:, np.newaxis] + A @ _int.vertices

    f1, df1 = V1(X)
    f2, df2 = V1(X)
    dv = np.power(1 - np.sum(X * X, axis=0), -.5)

    g11 = 1 - X[0, :] * X[0, :]
    g12 = X[0, :] * X[1, :]
    g22 = 1 - X[1, :] * X[1, :]
    p = np.sum(df2 * np.array([
        g11 * df1[0, :] + g12 * df1[1, :],
        g12 * df1[0, :] + g22 * df1[1, :]
    ]), axis=0)

    _I = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]) \
         * _int.integrate_vector(p * dv)

    ###

    X_uv = u[:, np.newaxis] + np.outer(v - u, t)
    X_vw = v[:, np.newaxis] + np.outer(w - v, t)
    X_wu = w[:, np.newaxis] + np.outer(u - w, t)

    f1_uv, df1_uv = V1(X_uv)
    f1_vw, df1_vw = V1(X_vw)
    f1_wu, df1_wu = V1(X_wu)

    f2_uv, df2_uv = V2(X_uv)
    f2_vw, df2_vw = V2(X_vw)
    f2_wu, df2_wu = V2(X_wu)

    _J_v = (v - u) @ star(df1_uv, X_uv) * f1_uv
    _J_v += (w - v) @ star(df1_vw, X_vw) * f1_vw
    _J_v += (u - w) @ star(df1_wu, X_wu) * f1_wu

    _J_v -= ((v - u) @ df1_uv) * np.sum(W_vec(X_uv) * star(df1_uv, X_uv), axis=0)
    _J_v -= ((w - v) @ df1_vw) * np.sum(W_vec(X_vw) * star(df1_vw, X_vw), axis=0)
    _J_v -= ((u - w) @ df1_wu) * np.sum(W_vec(X_wu) * star(df1_wu, X_wu), axis=0)

    _J = np.sum(_J_v * dt) - .5 * (_J_v[0] + _J_v[-1]) * dt

    N_samples.append(N * (N + 1) // 2)
    I.append(_I)
    J.append(_J / 2)

plt.plot(N_samples, I, "o--", label="interior integral")
plt.plot(N_samples, J, "o--", label="path integral")

plt.xlabel("nr. samples")
plt.ylabel("out")
plt.legend()

plt.show()
