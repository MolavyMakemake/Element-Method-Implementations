import numpy as np
import matplotlib.pyplot as plt
import Integrator

def klein_to_poincare(x):
    return x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

def klein_to_hyperboloid(x):
    t = 1 / np.sqrt(1 - np.sum(x*x, axis=0))
    return np.concatenate((t[np.newaxis, :], t * x))

def _midpoint_bkdisk(x, y):
    _a = (x - y) @ (x - y)
    _b = x @ x - x @ y
    _c = 1 - x @ x

    _d = np.sqrt(_b * _b + _a * _c)
    t0 = (_b + _d) / _a
    t1 = (_b - _d) / _a

    A = x + t0 * (y - x)
    B = x + t1 * (y - x)

    Ax = np.sqrt((x - A) @ (x - A))
    Ay = np.sqrt((y - A) @ (y - A))
    AB = np.sqrt((B - A) @ (B - A))

    ab = np.sqrt(Ax * Ay)
    t = ab / (np.sqrt((AB - Ax) * (AB - Ay)) + ab)

    return A + t * (B - A)

def hyperboloid_to_klein(x):
    return x[1:, :] / x[0, :]

def translate(a):
    return np.array([
        [a[0], -a[1], -a[2]],
        [-a[1], a[1] * a[1] / (a[0] + 1) + 1, a[1] * a[2] / (a[0] + 1)],
        [-a[2], a[1] * a[2] / (a[0] + 1), a[2] * a[2] / (a[0] + 1) + 1]
    ])

def rotate(a):
    s = 1.0 / np.sqrt(a[1] * a[1] + a[2] * a[2])
    return np.array([
        [1, 0, 0],
        [0, a[2] * s, -a[1] * s],
        [0, a[1] * s, a[2] * s]
    ])

def shift(a):
    s = 1.0 / np.sqrt(1 + a[1] * a[1])
    return np.array([
        [a[0] * s, 0, -a[2] * s],
        [0, 1, 0],
        [-a[2] * s, 0, a[0] * s]
    ])

def orthomap(x):
    A = translate(x[:, 0])
    y = A @ x

    y_b = rotate(y[:, 2]) @ y
    t_b = np.arctanh(np.abs(y_b[2, 1] / y_b[0, 1]))

    y_c = rotate(y[:, 1]) @ y
    t_c = np.arctanh(np.abs(y_c[2, 2] / y_c[0, 2]))

    x = hyperboloid_to_klein(y)
    H_b = np.tanh(t_b) * x[:, 2] / np.sqrt(x[:, 2] @ x[:, 2])
    H_c = np.tanh(t_c) * x[:, 1] / np.sqrt(x[:, 1] @ x[:, 1])

    oc = x[:, 1] + np.linalg.solve(np.array([H_b - x[:, 1], H_c - x[:, 2]]).T, x[:, 2] - x[:, 1])[0] * (H_b - x[:, 1])
    oc = klein_to_hyperboloid(oc[:, np.newaxis])[:, 0]

    return translate(oc) @ A

def barymap(a):
    integrator = Integrator.Integrator(100)

    x = a[:, 0, np.newaxis] + np.array([
        a[:, 1] - a[:, 0],
        a[:, 2] - a[:, 0]
    ]).T @ integrator.vertices
    print(x)
    dv = np.power(1 - np.sum(x * x, axis=0), -1.5)

    bc = np.array([
        integrator.integrate_vector(x[0, :] * dv),
        integrator.integrate_vector(x[1, :] * dv)
    ])

    y = klein_to_hyperboloid(bc[:, np.newaxis])[:, 0]
    return translate(y)

def centroidmap(a):
    m1 = _midpoint_bkdisk(a[:, 0], a[:, 2])
    m2 = _midpoint_bkdisk(a[:, 0], a[:, 1])

    M = a[:, 1] + np.linalg.solve(
        np.array([m1 - a[:, 1], m2 - a[:, 2]]).T,
        a[:, 2] - a[:, 1]
    )[0] * (m1 - a[:, 1])

    return translate(
        klein_to_hyperboloid(M[:, np.newaxis])[:, 0]
    )

def f1(x, y):
    return x * np.power((1 - y * y) / ((1 - x * x) * (1 - x * x - y * y)), .25)

def f2(x, y):
    return y * np.power((1 - x * x) / ((1 - y * y) * (1 - x * x - y * y)), .25)

def f(x, a):
    b = klein_to_hyperboloid(a)
    y = klein_to_hyperboloid(x)

    A = translate(b[:, 0])
    b = hyperboloid_to_klein(A @ b)
    y = hyperboloid_to_klein(A @ y)

    F = np.array([
        [1.0, f1(b[0, 0], b[1, 0]), f2(b[0, 0], b[1, 0])],
        [1.0, f1(b[0, 1], b[1, 1]), f2(b[0, 1], b[1, 1])],
        [1.0, f1(b[0, 2], b[1, 2]), f2(b[0, 2], b[1, 2])]
    ])
    F = np.linalg.inv(F)

    z1 = f1(y[0, :], y[1, :])
    z2 = f2(y[0, :], y[1, :])
    return F.T @ np.array([np.ones_like(z1), z1, z2])

a = np.array([
    [-0.5, -0.5],
    [0.1, -0.1],
    [-0.3, 0.5],
]).T

if False:
    b = klein_to_hyperboloid(a)
    A = orthomap(b)

    a = hyperboloid_to_klein(A @ b)

    t = np.linspace(0, 1)
    s = np.linspace(-1, 1)

    s1 = klein_to_poincare(a[:, 0, np.newaxis] + np.outer((a[:, 1] - a[:, 0]), t))
    s2 = klein_to_poincare(a[:, 1, np.newaxis] + np.outer((a[:, 2] - a[:, 1]), t))
    s3 = klein_to_poincare(a[:, 2, np.newaxis] + np.outer((a[:, 0] - a[:, 2]), t))

    b1 = klein_to_poincare(np.outer(a[:, 0], s))
    b2 = klein_to_poincare(np.outer(a[:, 1], s))
    b3 = klein_to_poincare(np.outer(a[:, 2], s))

    for _x in [s1, s2, s3, b1, b2, b3]:
        plt.plot(_x[0, :], _x[1, :], color="black", linewidth=.5)

    plt.axis("equal")
    plt.show()

if False:
    b = klein_to_hyperboloid(a)
    A = orthomap(b)

    a = hyperboloid_to_klein(A @ b)

    integrator = Integrator.Integrator(100)
    x = a[:, 0, np.newaxis] + np.array([a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]]).T @ integrator.vertices
    y = f(x, a)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.scatter(x[0, :], x[1, :], f(x, a)[2, :], s=.4)
    ax.scatter(x[0, 0], x[1, 0], f(x, a)[0, 0], color="red")
    ax.scatter(x[0, 99], x[1, 99], f(x, a)[1, 99], color="green")
    ax.scatter(x[0, -1], x[1, -1], f(x, a)[2, -1], color="blue")
    ax.legend()
    plt.show()

if True:
    a2 = np.array([a[:, 0], a[:, 1], np.array([-0.1, -0.5])]).T
    t = np.linspace(0, 1)

    s1 = a[:, 0, np.newaxis] + np.outer((a[:, 1] - a[:, 0]), t)
    s2 = a[:, 1, np.newaxis] + np.outer((a[:, 2] - a[:, 1]), t)
    s3 = a[:, 2, np.newaxis] + np.outer((a[:, 0] - a[:, 2]), t)
    s4 = a2[:, 2, np.newaxis] + np.outer((a2[:, 0] - a2[:, 2]), t)
    s5 = a2[:, 2, np.newaxis] + np.outer((a2[:, 1] - a2[:, 2]), t)

    y1 = f(s1, a)[0, :]
    y2 = f(s1, a2)[0, :]
    plt.plot(t, y1, color="r")
    plt.plot(t, y2, color="g")
    plt.show()

    #for _x in [s1, s2, s3, s4, s5]:
    #    plt.plot(_x[0, :], _x[1, :], color="black", linewidth=.5)

    #plt.show()


