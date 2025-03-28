import numpy as np
from Integrator import Integrator
import matplotlib.pyplot as plt

def V1(x):
    r = x * x
    t = np.sqrt(1 - r[0, :] - r[1, :])
    s = np.sqrt(t * (1 + t))

    return x[0, :] / s, np.array([
        t * (1 + t) + r[0, :] * (.5 / t + 1),
        x[0, :] * x[1, :] * (.5 / t + 1)]) / (s * s * s)

def V2(x):
    r = x * x
    t = np.sqrt(1 - r[0, :] - r[1, :])
    s = np.sqrt(t * (1 + t))

    return x[1, :] / s, np.array([
        x[0, :] * x[1, :] * (.5 / t + 1),
        t * (1 + t) + r[1, :] * (.5 / t + 1)]) / (s * s * s)

def xy_d(x):
    t = np.sqrt(1.0 - np.sum(x * x, axis=0))

    y = x / (1 + t)
    r = y * y

    f = y[1, :] / (1 - r[0, :] - r[1, :])
    dr = np.array([
        y[0, :] * y[1, :] / t,
        1 / (1 + t) + r[1, :] / t
    ])

    return f, dr

def W_vec(x):
    r = x * x
    t = np.sqrt(1 - r[0, :] - r[1, :])

    return 2 * t * t / (1 + t) * x

def W(x):
    r = x * x
    t = np.sqrt(1 - r[0, :] - r[1, :])

    return 2 / (t * t * (1 + t)) * x

def dst(x0, x1):
    y0 = x0 / (1 + np.sqrt(1 - np.sum(x0 * x0, axis=0)))
    y1 = x1 / (1 + np.sqrt(1 - np.sum(x1 * x1, axis=0)))

    delta = 1 + 2 * np.sum((y0 - y1) * (y0 - y1), axis=0) \
            / ((1 - np.sum(y0 * y0, axis=0)) * (1 - np.sum(y1 * y1, axis=0)))

    return np.arccosh(delta)

def compute_hyperbolic_area(v):
    N_v = np.size(v, axis=1)

    I = (N_v - 2) * np.pi
    for i1 in range(N_v):
        i0 = i1 - 1
        i2 = (i1 + 1) % N_v

        u1 = v[:, i2] - v[:, i1]
        u2 = v[:, i0] - v[:, i1]

        r = v[:, i1] * v[:, i1]
        g = np.array([
            [1 - r[1], v[0, i1] * v[1, i1]],
            [v[0, i1] * v[1, i1], 1 - r[0]]
        ]) / np.square(1 - r[0] - r[1])

        I -= np.arccos(np.dot(u1, g @ u2) / np.sqrt(
            np.dot(u1, g @ u1) * np.dot(u2, g @ u2)
        ))

    return I

def Int_PP(v, t, dt):
    dv = np.roll(v, -1, axis=1) - v

    area = compute_hyperbolic_area(v)

    I11 = area / 2
    I12 = 0
    I22 = area / 2
    for i in range(np.size(v, axis=1)):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)

        p1, dp1 = V1(X)
        p2, dp2 = V2(X)

        sW = star(W(X), X)

        c = np.sqrt(1.0 - np.sum(X * X, axis=0))
        Y = X / (1 + c)
        r = Y * Y

        dx = np.array([1 / (1 + c) + r[0, :] / c, Y[0, :] * Y[1, :] / c])
        dy = np.array([Y[0, :] * Y[1, :] / c, 1 / (1 + c) + r[1, :] / c])

        I11_v = (1 / 16) * dv[:, i] @ sW * p1 * p1
        I12_v = (1 / 16) * dv[:, i] @ sW * p1 * p2
        I22_v = (1 / 16) * dv[:, i] @ sW * p2 * p2

        s = 1.0 / np.sqrt(1 - r)
        I11_v += .5 * dv[:, i] @ dx * (X[1, :] / c - 2 * s[0, :] * np.arctanh(s[0, :] * Y[1, :]))
        I12_v += .5 * dv[:, i] @ dy * X[1, :] / c
        I22_v -= .5 * dv[:, i] @ dy * (X[0, :] / c - 2 * s[1, :] * np.arctanh(s[1, :] * Y[0, :]))

        I11 += np.sum(I11_v * dt) - .5 * (I11_v[0] + I11_v[-1]) * dt
        I12 += np.sum(I12_v * dt) - .5 * (I12_v[0] + I12_v[-1]) * dt
        I22 += np.sum(I22_v * dt) - .5 * (I22_v[0] + I22_v[-1]) * dt

    return np.array([[I11, I12], [I12, I22]])

def Int_WP(v, t, dt):
    dv = np.roll(v, -1, axis=1) - v

    I = np.zeros(shape=(2, np.size(v, axis=1)), dtype=float)
    for i in range(np.size(v, axis=1)):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)

        p1, dp1 = V1(X)
        p2, dp2 = V2(X)



    return I

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
v = np.array([-.8, 0.1])
w = np.array([.5, -.2])

N_samples = []
I = []
J = []

for N in [100, 200, 300, 400]:
    print("N =",N)

    N_t = N * (N + 1) // 6
    dt = 1 / N_t
    t = np.linspace(0, 1, N_t+1)

    ###

    _int = Integrator(N)
    A = np.array([v - u, w - u]).T
    X = u[:, np.newaxis] + A @ _int.vertices

    f1, df1 = V1(X)
    f2, df2 = V2(X)
    dv = np.power(1 - np.sum(X * X, axis=0), -.5)

    g11 = 1 - X[0, :] * X[0, :]
    g12 = -X[0, :] * X[1, :]
    g22 = 1 - X[1, :] * X[1, :]
    p = np.sum(df1 * np.array([
        g11 * df1[0, :] + g12 * df1[1, :],
        g12 * df1[0, :] + g22 * df1[1, :]
    ]), axis=0)

    _Y = X / (1 + np.sqrt(1 - np.sum(X * X, axis=0)))

    _I = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]) \
         * _int.integrate_vector(p * dv)

    ###


    _J = Int_PP(np.array([u, v, w]).T, t, dt)

    N_samples.append(N * (N + 1) // 2)
    I.append(_I)
    J.append(_J[0, 0])

plt.plot(N_samples, I, "o--", label="interior integral")
plt.plot(N_samples, J, "o--", label="path integral")

plt.xlabel("nr. samples")
plt.ylabel("out")
plt.legend()

plt.show()
