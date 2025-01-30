import numpy as np
import matplotlib.pyplot as plt

res = 100
size = 1

_s = np.tanh(size / 2) / np.sqrt(2)
dx = dy = 2 * _s / (res - 1)

X = np.zeros(shape=(2, res * res), dtype=float)
_X0, _X1 = np.meshgrid(np.linspace(-_s, _s, res), np.linspace(-_s, _s, res))

X[0, :] = _X0.flatten()
X[1, :] = _X1.flatten()

#X, Y = np.meshgrid(np.linspace(-size, size, res), np.linspace(-size, size, res))
#_R = np.sqrt(X * X + Y * Y); _r = np.tanh(_R / 2)
#X *= _r / _R; Y *= _r / _R


u = np.array([-.2, .8])
v = np.array([0, 0])

d_uv = 1 + 2 * ((u - v) @ (u - v)) / ((1 - u@u) * (1 - v@v))

def d(x, w):
    u = np.zeros_like(x)
    u[0, :] = x[0, :] - w[0]
    u[1, :] = x[1, :] - w[1]

    u2 = np.sum(u * u, axis=0)
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * u2 / ((1 - x2) * (1 - w2))

def A_tri(x):
    d0 = d_uv
    d1 = d(x, u)
    d2 = d(x, v)

    A = 1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2
    B = 1 + d0 + d1 + d2

    sgn = 2 * (x[0, :] > 0) - 1
    sgn = 1
    return 2 * np.atan(np.sqrt(np.abs(A)) / B)

def A_htri(x, y):
    def sign (A, B):
        return (x - B[0]) * (A[1] - B[1]) - (A[0] - B[0]) * (y - B[1])

    A = np.zeros_like(x)
    I = (2 / (1 - x * x - y * y)) ** 2
    #I = np.ones_like(x)
    for i in range(np.size(x, axis=0)):
        for j in range(np.size(y, axis=1)):
            w = np.array([x[i, j], y[i, j]])
            d1 = sign(u, v)
            d2 = sign(v, w)
            d3 = sign(w, u)

            has_neg = np.logical_or(np.logical_or(d1 < 0, d2 < 0), d3 < 0)
            has_pos = np.logical_or(np.logical_or(d1 > 0, d2 > 0), d3 > 0)

            A[i, j] = np.sum(I, where=np.logical_not(np.logical_and(has_neg, has_pos))) * dx * dy

    return A * np.sign(x)

def A_ihtri(x, y):
    def _int(x0, y0):
        x = np.linspace(0, x0)
        dx = x[1] - x[0]

        y0 = x / x0 * y0
        y1 = y0 + v[1] * (1 - x / x0)
        u0 = y0 / np.sqrt(1 - x * x)
        u1 = y1 / np.sqrt(1 - x * x)
        v0 = np.atanh(u0) + u0 / (1 - u0 * u0)
        v1 = np.atanh(u1) + u1 / (1 - u1 * u1)

        return 2 * np.sum((v1 - v0) / (1 - x * x) ** 1.5) * dx

    A = np.zeros_like(x)
    for i in range(np.size(x, axis=0)):
        for j in range(np.size(y, axis=1)):
            A[i, j] = _int(x[i, j], y[i, j])

    return A


def D1V(x):
    d0 = d_uv
    d1 = d(x, u)
    d2 = d(x, v)

    xu = x - u[:, np.newaxis]
    xv = x - v[:, np.newaxis]

    x2 = np.sum(x * x, axis=0)

    D1d1 = 2 * (d1 - 1) * x[1, :] / (1 - x2) + 4 * xu[1, :] / ((1 - u @ u) * (1 - x2))
    D1d2 = 2 * (d2 - 1) * x[1, :] / (1 - x2) + 4 * xv[1, :] / ((1 - v @ v) * (1 - x2))

    A = 1 - d0*d0 - d1*d1 - d2*d2 + 2 * d0 * d1 * d2
    B = 1 + d0 + d1 + d2

    V = np.sqrt(A) / B
    a = (-d1 * D1d1 - d2 * D1d2 + d0 * d1 * D1d2 + d0 * D1d1 * d2)
    b = (D1d1 + D1d2)

    M = np.max(2 * np.atan(V))
    return 2 * V / (1 + V*V) * (a / A - b / B)
    #return D1d1

F = A_tri(X)
DF = np.gradient(np.reshape(F, (res, res)), dx, dy)

k = .5 * (1 - _X0 * _X0 - _X1 * _X1)
DgF1 = np.gradient(DF[0], dx, axis=0)
DgF2 = np.gradient(DF[1], dy, axis=1)
LgF = k * k * (DgF1 + DgF2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.plot_surface(_X0, _X1, np.clip(LgF, -0.01, 0.01), label="f_1")
#ax.scatter(X[0, :], X[1, :], D1V(X), s=0.1, color="yellow")
#ax.plot_surface(_X0, _X1, np.clip(np.reshape(D1V(X), (res, res)), -1, 1), label="f_1")
#ax.plot_surface(X, Y, k * k * DF[1], label="f_2")
#ax.plot_surface(X, Y, np.clip(LgF, -1, 1), label="Lf")
ax.legend()
plt.show()
