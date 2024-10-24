import numpy as np
import matplotlib.pyplot as plt

res = 100
size = 10

_s = np.tanh(size / 2) / np.sqrt(2)
dx = dy = 2 * _s / (res - 1)
X, Y = np.meshgrid(np.linspace(-_s, _s, res), np.linspace(-_s, _s, res))

#X, Y = np.meshgrid(np.linspace(-size, size, res), np.linspace(-size, size, res))
#_R = np.sqrt(X * X + Y * Y); _r = np.tanh(_R / 2)
#X *= _r / _R; Y *= _r / _R


u = np.array([0, .5])
v = np.array([0, 0])

def d(x, y, w):
    a = x - w[0]
    b = y - w[1]
    return 1 + 2 * (a*a + b*b) / ((1 - x*x - y*y) * (1 - w @ w))

def A_tri(x, y):
    w = np.array([x, y])
    d0 = d(u[0], u[1], v)
    d1 = d(x, y, u)
    d2 = d(x, y, v)

    A = 1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2
    B = 1 + d0 + d1 + d2

    return 2 * np.atan(np.sqrt(A) / B) * np.sign((u[0] - X) * (v[1] - u[1]) - (u[1] - Y) * (v[0] - u[0]))

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


F = A_tri(X, Y)
DF = np.gradient(F, dx, dy)

k = .5 * (1 - X * X - Y * Y)
DgF1 = np.gradient(DF[0], dx, axis=0)
DgF2 = np.gradient(DF[1], dy, axis=1)
LgF = k * k * (DgF1 + DgF2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.plot_surface(X, Y, DgF1, label="f_1")
ax.plot_surface(X, Y, DgF2, label="f_2")
#ax.plot_surface(X, Y, np.clip(LgF, -1, 1), label="Lf")
ax.legend()
plt.show()
