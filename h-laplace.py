import numpy as np
import matplotlib.pyplot as plt

res = 100
size = 10

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
v = np.array([-.4, -.5])

d_uv = 1 + 2 * ((u - v) @ (u - v)) / ((1 - u@u) * (1 - v@v))

def d(x, w):
    u = np.zeros_like(x)
    u[0, :] = x[0, :] - w[0]
    u[1, :] = x[1, :] - w[1]

    u2 = np.sum(u * u, axis=0)
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * u2 / ((1 - x2) * (1 - w2))

def nrm_dst(b, y):
    a = 2 * b / (1 + np.sum(b * b, axis=0))
    x = 2 * y / (1 + np.sum(y * y, axis=0))

    det = (x[0, :] - a[0, 0]) * (a[1, 2] - a[1, 1]) \
            - (x[1, :] - a[1, 0]) * (a[0, 2] - a[0, 1])
    t = ((a[1, 2] - a[1, 1]) * (a[0, 1] - a[0, 0])
         - (a[0, 2] - a[0, 1]) * (a[1, 1] - a[1, 0])) / det

    x_A = a[:, 0, np.newaxis] + t * (x - a[:, 0, np.newaxis])

    d0 = d(y, b[:, 0])
    d1 = d(x_A / (1 + np.sqrt(1 - np.sum(x_A * x_A, axis=0))), b[:, 0])

    f = 1 - np.acosh(d0) / np.acosh(d1)
    f[np.isnan(f)] = 1
    return f


def A_tri(x):
    d0 = d_uv
    d1 = d(x, u)
    d2 = d(x, v)

    A = 1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2
    B = 1 + d0 + d1 + d2

    return 2 * np.atan(np.sqrt(np.abs(A)) / B)

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

F = nrm_dst(np.array([[0, 0], u, v]).T, X)
DF = np.gradient(np.reshape(F, (res, res)), dx, dy)

k = .5 * (1 - _X0 * _X0 - _X1 * _X1)
DgF1 = np.gradient(DF[0], dx, axis=0)
DgF2 = np.gradient(DF[1], dy, axis=1)
LgF = k * k * (DgF1 + DgF2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")

print(np.sum(np.isnan(F)))

ax.plot_surface(_X0, _X1, LgF, label="f_1")
ax.scatter(u[0], u[1])
ax.scatter(v[0], v[1])
#ax.scatter(X[0, :], X[1, :], D1V(X), s=0.1, color="yellow")
#ax.plot_surface(_X0, _X1, np.clip(np.reshape(D1V(X), (res, res)), -1, 1), label="f_1")
#ax.plot_surface(X, Y, k * k * DF[1], label="f_2")
#ax.plot_surface(X, Y, np.clip(LgF, -1, 1), label="Lf")
ax.legend()
plt.show()
