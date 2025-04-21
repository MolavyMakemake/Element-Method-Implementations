import numpy as np
import matplotlib.pyplot as plt
from hyperbolic.triangulate import save
from hyperbolic.Integrator import Integrator

_BC_INTEGRATOR = Integrator(20)
def barycenter(v):
    v = 2 * v / (1 + np.sum(v * v, axis=0))
    X = v[:, 0, np.newaxis] + np.array([
        v[:, 1] - v[:, 0],
        v[:, 2] - v[:, 0]
    ]).T @ _BC_INTEGRATOR.vertices

    dv = np.power(1 - np.sum(X * X, axis=0), -1.5)
    bc = np.array([
        _BC_INTEGRATOR.integrate_vector(X[0, :] * dv),
        _BC_INTEGRATOR.integrate_vector(X[1, :] * dv)
    ]) / _BC_INTEGRATOR.integrate_vector(dv)

    return bc / (1 + np.sqrt(1 - bc @ bc))

def distance(x, y):
    _d = 2 * (x - y) @ (x - y) / ((1 - x @ x) * (1 - y @ y))
    return np.arccosh(1 + _d)

def cos_a(u, v, x):
    z0 = ((1 - x[1] * x[1]) * u[0] + x[0] * x[1] * u[1])
    z1 = ((1 - x[0] * x[0]) * u[1] + x[0] * x[1] * u[0])
    w0 = ((1 - x[1] * x[1]) * v[0] + x[0] * x[1] * v[1])
    w1 = ((1 - x[0] * x[0]) * v[1] + x[0] * x[1] * v[0])

    return (z0 * v[0] + z1 * v[1]) / np.sqrt((z0 * u[0] + z1 * u[1]) * (w0 * v[0] + w1 * v[1]))

def angles(x0, x1, x2):
    y0 = 2.0 / (1 + x0 @ x0) * x0
    y1 = 2.0 / (1 + x1 @ x1) * x1
    y2 = 2.0 / (1 + x2 @ x2) * x2

    u0 = y1 - y0
    u1 = y2 - y1
    u2 = y0 - y2

    a0 = np.arccos(-cos_a(u0, u1, y1))
    a1 = np.arccos(-cos_a(u1, u2, y2))
    a2 = np.arccos(-cos_a(u2, u0, y0))
    return [a0, a1, a2]

N_v = 1203
N_bdry = 284

vertices = []
triangles = []
file = open(f"../meshgen/output/triangulation_square_hyp_{N_v}({N_bdry}).txt")
exec(file.read())
file.close()

R = np.tanh(1.5)
boundary = [i for i in range(N_v - N_bdry, N_v, 1)]

is_boundary = lambda i: triangles[i] >= N_v - N_bdry

N = N_v
_triangles = []
for i in range(0, len(triangles), 3):
    if not (is_boundary(i) and is_boundary(i+1) and is_boundary(i+2)):
        _triangles.append([triangles[i], triangles[i + 1], triangles[i + 2]])
        continue

    v = np.array([
        [vertices[2 * triangles[i + 0] + 0], vertices[2 * triangles[i + 1] + 0], vertices[2 * triangles[i + 2] + 0]],
        [vertices[2 * triangles[i + 0] + 1], vertices[2 * triangles[i + 1] + 1], vertices[2 * triangles[i + 2] + 1]]
    ])
    bc = barycenter(v)
    vertices.append(bc[0])
    vertices.append(bc[1])
    _triangles.append([N, triangles[i + 0], triangles[i + 1]])
    _triangles.append([N, triangles[i + 1], triangles[i + 2]])
    _triangles.append([N, triangles[i + 2], triangles[i + 0]])
    N += 1

vertices = np.array(vertices).reshape((len(vertices) // 2, 2)).T

plt.scatter(vertices[0, :], vertices[1, :], color="black", s=.2)
plt.scatter(vertices[0, boundary], vertices[1, boundary], color="red", s=1)
plt.show()

save(vertices, _triangles, boundary, f"uniform_rect_{N}")

H = []
T = []
for p_i in _triangles:
    x0 = vertices[:, p_i[0]]
    x1 = vertices[:, p_i[1]]
    x2 = vertices[:, p_i[2]]

    H.append(max([
        distance(x0, x1),
        distance(x1, x2),
        distance(x2, x0)
    ]))

    T.append(min(angles(x0, x1, x2)))

H = np.array(H)
T = np.array(T)

print(f"{np.min(T):.6f}")
print(f"{np.mean(T):.6f}")
print(f"{np.std(T):.6f}\n")
print(f"{np.max(H):.6f}")
print(f"{np.mean(H):.6f}")
print(f"{np.std(H):.6f}\n")