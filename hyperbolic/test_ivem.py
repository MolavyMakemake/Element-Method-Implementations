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

def Int_WP(v, t):
    dv = np.roll(v, -1, axis=1) - v
    h = dst(v, np.roll(v, -1, axis=1))

    N_v = np.size(v, axis=1)
    I = np.zeros(shape=(2, N_v), dtype=float)
    for i in range(N_v):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)

        ds = dst(X[:, :-1], X[:, 1:])

        p1, dp1 = V1(X)
        p2, dp2 = V2(X)

        I0 = np.sum(ds * (p1[:-1] + p1[1:])) / 2
        I1 = np.sum(ds * (p2[:-1] + p2[1:])) / 2

        j = (i + 1) % N_v

        I[0, i] += I0 * h[i-1]
        I[1, i] += I1 * h[i-1]

        I[0, j] -= I0 * h[j]
        I[1, j] -= I1 * h[j]

    return I[:, :-1]

def Int_VW(v):
    N_v = np.size(v, axis=1)
    h = dst(v, np.roll(v, -1, axis=1))

    I = np.zeros(shape=(N_v - 1, N_v), dtype=float)
    i = np.arange(0, N_v - 1)

    I[i[1:], i[:-1]] = -h[i[:-1]] * h[i[1:]] / 2
    I[i, i+1] = h[i] * h[i-1] / 2
    I[0, -1] = -h[0] * h[-1] / 2
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


v = np.array([
    [-.3, .8],
    [-.8, 0.1],
    [.3, -.2]
]).T

N = 1000
dt = 1 / N
t = np.linspace(0, 1, N+1)

I_wp = Int_WP(v, t)
I_pp = Int_PP(v, t, dt)
I_vw = Int_VW(v)

X = Integrator(100).vertices
X = v[:, 0, np.newaxis] + np.array([
    v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
]).T @ X

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")

proj_P = np.linalg.inv(I_pp) @ I_wp

h = dst(v, np.roll(v, -1, axis=1)) / np.sqrt(2)
a = np.linalg.lstsq(proj_P.T @ I_wp, I_vw)[0]

print(I_vw)
print(np.array([
    [0, -h[0] * h[1]],
    [h[0] * h[2], 0],
    [-h[0] * h[2], h[0] * h[1]]
]).T)

I = 1
a = proj_P @ a[:, I]


for i in range(np.size(v, axis=1)):
    ax.scatter(v[0, i], v[1, i], s=0.6)

ax.scatter(v[0, I], v[1, I], s=5.6)

Y = a[0] * X[0, :] + a[1] * X[1, :]

ax.scatter(X[0, :], X[1, :], Y - np.average(Y), color="b", alpha=0.5, s=.4)
ax.legend()
plt.show()
