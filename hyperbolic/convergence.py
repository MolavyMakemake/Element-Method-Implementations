import numpy as np
import matplotlib.pyplot as plt
import plot, FEM_BKDISK_O1, FEM_BKDISK_O2, FEM_PDISK_O1, FEM_PDISK_O2, triangulate


def distance(x, y):
    return np.sqrt((x - y) @ (x - y))

    _d = 2 * (x - y) @ (x - y) / ((1 - x @ x) * (1 - y @ y))
    return np.arccosh(1 + _d)

def compute_h(vertices, polygons):
    h = 0
    for p_i in polygons:
        h = max([h,
            distance(vertices[:, p_i[0]], vertices[:, p_i[1]]),
            distance(vertices[:, p_i[1]], vertices[:, p_i[2]]),
            distance(vertices[:, p_i[2]], vertices[:, p_i[0]])
        ])

    return h

R = .95

#g0 = lambda t: -np.log(1 - t)
#c = g0(R * R)
#
#v = lambda z: c - g0(z * np.conj(z))
#f = lambda z: 1

v = lambda z: R * R - z * np.conj(z)
f = lambda z: np.power(1 - z * np.conj(z), 2)

v_k = lambda z: v(z / (1 + np.sqrt(1 - z * np.conj(z))))
f_k = lambda z: f(z / (1 + np.sqrt(1 - z * np.conj(z))))

V = [v, v, v_k, v_k]
F = [f, f, f_k, f_k]

H = []
Y = [[] for _ in range(len(F))]

for triangulation_f in ["rect35__poincare__95.npz",
                        "rect20__poincare__95.npz",
                        "rect08__poincare__95.npz",
                        "rect04__poincare__95.npz"]:

    vertices, polygons, boundary = triangulate.load(triangulation_f)
    vertices_k = triangulate._pdisk_to_bkdisk(vertices)

    models = [
        FEM_PDISK_O1.Model(vertices, polygons, boundary),
        FEM_PDISK_O2.Model(vertices, polygons, boundary),
        FEM_BKDISK_O1.Model(vertices_k, polygons, boundary),
        FEM_BKDISK_O2.Model(vertices_k, polygons, boundary)
    ]

    H.append(compute_h(vertices, polygons))
    for i in range(len(models)):
        u = np.real(models[i].solve_poisson(F[i]))
        Y[i].append(models[i].compare(V[i], "L2"))

print("H =", H)
for y in Y:
    print(y)

plt.loglog(H, Y[0], "o--", color="black", label="Poincaré k=1")
plt.loglog(H, Y[1], "<--", color="black", label="Poincaré k=2")
plt.loglog(H, Y[2], "o--", color="gray", label="Klein k=1")
plt.loglog(H, Y[3], "<--", color="gray", label="Klein k=2")
plt.legend()
plt.show()