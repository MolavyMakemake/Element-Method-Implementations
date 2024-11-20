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
W = R / np.sqrt(2)

def v(z):
    return -np.log(1 - z * np.conj(z))

    x2 = np.real(z) * np.real(z)
    y2 = np.imag(z) * np.imag(z)
    return .5 * (W * W - np.maximum(x2, y2))
def f(z):
    return 1
    x2 = np.real(z) * np.real(z)
    y2 = np.imag(z) * np.imag(z)
    a = np.maximum(x2, y2)
    b = np.minimum(x2, y2)
    return 3 * a*a + 3 * a*b - 4 * b + 1

g0 = lambda t: -np.log(1 - t)
c = g0(R * R)
v = lambda z: c - g0(z * np.conj(z))

#v = lambda z: R * R - z * np.conj(z)
#f = lambda z: np.power(1 - z * np.conj(z), 2)

v_k = lambda z: v(z / (1 + np.sqrt(1 - z * np.conj(z))))
f_p = lambda z: v(2 * z / (1 + z * np.conj(z)))

V = [v_k, v_k]
F = [f, f]

H = []
Y = [[] for _ in range(len(F))]

for triangulation_f in ["./triangulations/rect35__klein__d95.npz",
                        "./triangulations/rect20__klein__d95.npz",
                        "./triangulations/rect08__klein__d95.npz",
                        "./triangulations/rect04__klein__d95.npz"]:

    vertices, polygons, boundary = triangulate.load(triangulation_f)
    #vertices_k = triangulate._pdisk_to_bkdisk(vertices)

    models = [
        #FEM_PDISK_O1.Model(vertices, polygons, boundary),
        #FEM_PDISK_O2.Model(vertices, polygons, boundary),
        FEM_BKDISK_O1.Model(vertices, polygons, boundary),
        FEM_BKDISK_O2.Model(vertices, polygons, boundary)
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
#plt.loglog(H, Y[2], "o--", color="gray", label="Klein k=1")
#plt.loglog(H, Y[3], "<--", color="gray", label="Klein k=2")
plt.legend()
plt.show()