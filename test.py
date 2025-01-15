import numpy as np
import matplotlib.pyplot as plt
from hyperbolic.triangulate import save

def distance(x, y):
    #return np.sqrt((x - y) @ (x - y))

    _d = 2 * (x - y) @ (x - y) / ((1 - x @ x) * (1 - y @ y))
    return np.arccosh(1 + _d)

vertices = []
triangles = []
file = open("./meshgen/output/triangulation_hyp_512(131).txt")
exec(file.read())
vertices = np.array(vertices).reshape((len(vertices) // 2, 2)).T
file.close()

R = np.tanh(1.5)
boundary = []
for i in range(np.size(vertices, axis=1)):
    if np.dot(vertices[:, i], vertices[:, i]) > R * R - 1e-5:
        boundary.append(i)

    #vertices[:, i] *= R

_triangles = []
for i in range(0, len(triangles), 3):
    _triangles.append([triangles[i], triangles[i + 1], triangles[i + 2]])

print(len(_triangles))

H = []
for p_i in _triangles:
    H.append(max([
        distance(vertices[:, p_i[0]], vertices[:, p_i[1]]),
        distance(vertices[:, p_i[1]], vertices[:, p_i[2]]),
        distance(vertices[:, p_i[2]], vertices[:, p_i[0]])
    ]))

N = len(H)

h0 = np.min(H)
w = 0.005

buckets = np.arange(h0, np.max(H), w)
distribution = np.zeros_like(buckets)

s = 1.0 / N
for h in H:
    i = int((h - h0) / w)
    distribution[i] += s

plt.bar(buckets, distribution, width=w, align="edge")
plt.show()