import numpy as np
import matplotlib.pyplot as plt
from hyperbolic.triangulate import save

def distance(x, y):
    return np.sqrt((x - y) @ (x - y))

    _d = 2 * (x - y) @ (x - y) / ((1 - x @ x) * (1 - y @ y))
    return np.arccosh(1 + _d)

def cos_a(u, v, x):
    z0 = ((1 - x[1] * x[1]) * u[0] + x[0] * x[1] * u[1])
    z1 = ((1 - x[0] * x[0]) * u[1] + x[0] * x[1] * u[0])
    w0 = ((1 - x[1] * x[1]) * v[0] + x[0] * x[1] * v[1])
    w1 = ((1 - x[0] * x[0]) * v[1] + x[0] * x[1] * v[0])

    return (z0 * v[0] + z1 * v[1]) / np.sqrt((z0 * u[0] + z1 * u[1]) * (w0 * v[0] + w1 * v[1]))

def angles(x0, x1, x2):
    u0 = x1 - x0
    u1 = x2 - x1
    u2 = x0 - x2

    l0 = np.sqrt(u0 @ u0)
    l1 = np.sqrt(u1 @ u1)
    l2 = np.sqrt(u2 @ u2)

    a0 = np.arccos(-u0 @ u1 / (l0 * l1))
    a1 = np.arccos(-u1 @ u2 / (l1 * l2))
    a2 = np.arccos(-u2 @ u0 / (l2 * l0))
    return [a0, a1, a2]

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

N_v = 8192
N_bdry = 610

vertices = []
triangles = []
file = open(f"../meshgen/output/triangulation_hyp_{N_v}({N_bdry}).txt")
exec(file.read())
vertices = np.array(vertices).reshape((len(vertices) // 2, 2)).T
file.close()

R = np.tanh(1.5)
boundary = [i for i in range(N_v - N_bdry, np.size(vertices, axis=1), 1)]
_triangles = []
for i in range(0, len(triangles), 3):
    _triangles.append([triangles[i], triangles[i + 1], triangles[i + 2]])

save(vertices, _triangles, boundary, f"uniform_disk_hyp_{N_v}")

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