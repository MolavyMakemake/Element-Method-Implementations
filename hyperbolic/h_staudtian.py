import numpy as np
import matplotlib.pyplot as plt
import Integrator

def klein_to_poincare(x):
    return x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

def klein_to_hyperboloid(x):
    t = 1 / np.sqrt(1 - np.sum(x*x, axis=0))
    return np.concatenate((t[np.newaxis, :], t * x))

def hyperboloid_to_klein(x):
    return x[1:, :] / x[0, :]

def translate(x, a):
    return np.array([
        [a[0], -a[1], -a[2]],
        [-a[1], a[1] * a[1] / (a[0] + 1) + 1, a[1] * a[2] / (a[0] + 1)],
        [-a[2], a[1] * a[2] / (a[0] + 1), a[2] * a[2] / (a[0] + 1) + 1]
    ]) @ x

def rotate(x, a):
    s = 1.0 / np.sqrt(a[1] * a[1] + a[2] * a[2])
    return np.array([
        [1, 0, 0],
        [0, a[2] * s, -a[1] * s],
        [0, a[1] * s, a[2] * s]
    ]) @ x

def shift(x, a):
    s = 1.0 / np.sqrt(1 + a[1] * a[1])
    return np.array([
        [a[0] * s, 0, -a[2] * s],
        [0, 1, 0],
        [-a[2] * s, 0, a[0] * s]
    ]) @ x

def map_triangle(x, n):
    x = klein_to_hyperboloid(x)
    x = translate(x, x[:, 0])
    x = rotate(x, x[:, n - 1])
    x = shift(x, x[:, -1])
    x = hyperboloid_to_klein(x)
    return x

def distance(x, y, A):
    u0 = x - A[0]
    u1 = y - A[1]

    a = 1 - x * x - y * y
    b = u0 * u0 + u1 * u1
    d = 1.0 + 2.0 * b / (a * (1 - A @ A))

    dst = np.arccosh(d)

    s = 4.0 / (a * (1 - A @ A) * np.sqrt(d * d - 1))
    D1dst = s * (u0 + x * b / a)
    D2dst = s * (u1 + y * b / a)
    return dst, D1dst, D2dst

def staudtian(x, A, B):
    x = klein_to_poincare(x)
    A = klein_to_poincare(A)
    B = klein_to_poincare(B)

    a, D1a, D2a = distance(x[0, :], x[1, :], A)
    b, D1b, D2b = distance(x[0, :], x[1, :], B)
    c, _, _ = distance(A[0], A[1], B)
    s = (a + b + c) / 2.0

    S = np.sqrt(np.sinh(s) * np.sinh(s - a) * np.sinh(s - b) * np.sinh(s - c))
    D1S = 0.5 / S * (
        np.cosh(s) * np.sinh(s - a) * np.sinh(s - b) * np.sinh(s - c) * 0.5 * ( D1a + D1b) +
        np.sinh(s) * np.cosh(s - a) * np.sinh(s - b) * np.sinh(s - c) * 0.5 * (-D1a + D1b) +
        np.sinh(s) * np.sinh(s - a) * np.cosh(s - b) * np.sinh(s - c) * 0.5 * ( D1a - D1b) +
        np.sinh(s) * np.sinh(s - a) * np.sinh(s - b) * np.cosh(s - c) * 0.5 * ( D1a + D1b)
    )
    D2S = 0.5 / S * (
        np.cosh(s) * np.sinh(s - a) * np.sinh(s - b) * np.sinh(s - c) * 0.5 * ( D2a + D2b) +
        np.sinh(s) * np.cosh(s - a) * np.sinh(s - b) * np.sinh(s - c) * 0.5 * (-D2a + D2b) +
        np.sinh(s) * np.sinh(s - a) * np.cosh(s - b) * np.sinh(s - c) * 0.5 * ( D2a - D2b) +
        np.sinh(s) * np.sinh(s - a) * np.sinh(s - b) * np.cosh(s - c) * 0.5 * ( D2a + D2b)
    )

    return S, D1S, D2S


a0 = np.array([-0.2, -0.5])
a1 = np.array([0.5, 0.6])
a2 = np.array([-0.3, 0.5])


if False:
    a3 = np.array([0.4, -0.1])

    t = np.linspace(0, 1, 30)
    x = np.zeros(shape=(2, np.size(t)), dtype=float)
    x[0, :] = a0[0] + t * (a1[0] - a0[0])
    x[1, :] = a0[1] + t * (a1[1] - a0[1])

    #plt.plot([a0[0], a1[0], a2[0], a0[0]], [a0[1], a1[1], a2[1], a0[1]])
    #plt.plot([a0[0], a1[0], a3[0], a0[0]], [a0[1], a1[1], a3[1], a0[1]])
    #plt.show()

    y1, D1y1, D2y1 = staudtian(x, a0, a2)
    y2, D1y2, D2y2 = staudtian(x, a0, a3)

    plt.plot(t, y1 / np.max(y1) - y2 / np.max(y2))
    plt.show()

if True:
    integrator = Integrator.Integrator(100, open=True)
    x = a0[:, np.newaxis] + np.array([a1 - a0, a2 - a0]).T @ integrator.vertices

    y0, d1, d2 = staudtian(x, a0, a1)
    y1, _, _ = staudtian(x, a1, a2)
    y2, _, _ = staudtian(x, a0, a2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter(x[0, :], x[1, :], np.arcsinh(y0) + np.arcsinh(y1) + np.arcsinh(y2), s=.4)
    ax.legend()
    plt.show()
