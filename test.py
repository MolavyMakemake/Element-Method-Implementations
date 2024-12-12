import numpy as np
import matplotlib.pyplot as plt
from hyperbolic.Integrator import Integrator

_int = Integrator(100)

f = lambda x, y: x * np.power((1 - y*y) / ((1 - x*x) * (1 - x*x - y*y)), .25)
g = lambda x, y: y * np.power((1 - x*x) / ((1 - y*y) * (1 - x*x - y*y)), .25)

v1 = np.array([.5, -.2])
v2 = np.array([.4, .7])
v3 = np.array([-.5, -.5])


A = np.array([
    [f(v2[0], v2[1]), g(v2[0], v2[1]), 1],
    [f(0, 0), g(0, 0), 1],
    [f(v1[0], v1[1]), g(v1[0], v1[1]), 1]
])
a = np.linalg.solve(A, np.array([1, 0, 0]))

B = np.array([
    [f(v2[0], v2[1]), g(v2[0], v2[1]), 1],
    [f(v3[0], v3[1]), g(v3[0], v3[1]), 1],
    [f(0, 0), g(0, 0), 1]
])
b = np.linalg.solve(B, np.array([1, 0, 0]))

t = np.linspace(0, 1)
X = t * v2[0]
Y = t * v2[1]

plt.plot(t, a[0] * f(X, Y) + a[1] * g(X, Y) + a[2])
plt.plot(t, b[0] * f(X, Y) + b[1] * g(X, Y) + b[2])
plt.show()

X0 = _int.vertices[0, :]
Y0 = _int.vertices[1, :]

X = v2[0] * X0 + v3[0] * Y0
Y = v2[1] * X0 + v3[1] * Y0

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.scatter(X, Y, b[0] * f(X, Y) + b[1] * g(X, Y) + b[2], s=.4)
ax.legend()
plt.show()