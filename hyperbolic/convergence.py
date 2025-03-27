import numpy as np
import matplotlib.pyplot as plt
import plot, FEM_BKDISK_O1, triangulate
import FEM_SIMPLEX_O1, FEM_STAUDTIAN_O1
#import VEM_PDISK_O1, VEM_HOMOGENEOUS_O1, vem_triangulate


def distance(x, y):
    #return np.sqrt((x - y) @ (x - y))

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

R_k = np.sqrt(2) * np.tanh(.86)
R = np.tanh(1.5)
W = np.tanh(.86)
W2 = W * W

def v(z):
    x2 = np.real(z) * np.real(z)
    y2 = np.imag(z) * np.imag(z)
    return (W2 - x2) * (W2 - y2)
def f(z):
    x2 = np.real(z) * np.real(z)
    y2 = np.imag(z) * np.imag(z)
    L = -2 * x2 * (1 - x2) * (W2 - y2) + 2 * x2 * y2 * (W2 - x2) \
        -2 * y2 * (1 - y2) * (W2 - x2) + 2 * x2 * y2 * (W2 - y2) \
        + (1 - x2 - y2) * (
        -2 * (1 - x2) * (W2 - y2) + 4 * x2 * (W2 - y2) + 2 * y2 * (W2 - x2) - 4 * x2 * y2
        -2 * (1 - y2) * (W2 - x2) + 4 * y2 * (W2 - x2) + 2 * x2 * (W2 - y2) - 4 * x2 * y2
        )
    return -L

g0 = lambda t: -np.log(1 - t)
c = g0(R * R)
v_p = lambda z: c - g0(z * np.conj(z))

#v_p = lambda z: R * R - z * np.conj(z)
#f = lambda z: np.power(1 - z * np.conj(z), 2)

#v = lambda z: v_p(z / (1 + np.sqrt(1 - z * np.conj(z))))
#f_p = lambda z: v(2 * z / (1 + z * np.conj(z)))

#f = lambda z: 1

V = [v, v, v]
F = [f, f, f]

H_dof = [[] for _ in range(len(F))]
H_g = [[] for _ in range(len(F))]
Y_e = [[] for _ in range(len(F))]
Y_g = [[] for _ in range(len(F))]

#for triangulation_f in ["./triangulations/uniform_disk_euc_256.npz",
#                        "./triangulations/uniform_disk_euc_512.npz",
#                        "./triangulations/uniform_disk_euc_1024.npz",
#                        "./triangulations/uniform_disk_euc_2048.npz",
#                        "./triangulations/uniform_disk_euc_4096.npz",
#                        "./triangulations/uniform_disk_euc_8192.npz"]:

for triangulation_f in ["./triangulations/uniform_rect_180.npz",
                        "./triangulations/uniform_rect_560.npz",
                        "./triangulations/uniform_rect_1100.npz"]:
#for N in [256, 512, 1024, 2048, 4096, 8192]:
    #print(N)

    #vertices, polygons, boundary = triangulate.load(f"./triangulations/uniform_disk_hyp_{N}.npz")
    #vertices_vem, polygons_vem, boundary_vem = vem_triangulate.vem_mesh(N)

    print(triangulation_f)
    _vertices, polygons, boundary = triangulate.load(triangulation_f)

    #vertices = np.tanh(1.5) / np.sqrt(np.max(np.sum(_vertices * _vertices, axis=0))) * _vertices
    #vertices_k = np.tanh(3.0) / np.sqrt(np.max(np.sum(_vertices * _vertices, axis=0))) * _vertices

    vertices = _vertices
    vertices_k = triangulate._pdisk_to_bkdisk(_vertices)

    res = 100
    models = [
        FEM_BKDISK_O1.Model(vertices_k, polygons, boundary),
        FEM_SIMPLEX_O1.Model(vertices_k, polygons, boundary),
        FEM_STAUDTIAN_O1.Model(vertices_k, polygons, boundary)
    ]

    h = compute_h(vertices, polygons)
    for i in range(len(models)):
        u = np.real(models[i].solve_poisson(F[i]))
        H_dof[i].append(models[i]._n_elements)
        H_g[i].append(h)
        Y_e[i].append(models[i].compare(V[i], "L2"))
        Y_g[i].append(models[i].compare(V[i], "L2_g"))

print("DOF =", H_dof)
print("Y_g =", Y_g)

if True:

    plt.loglog(H_dof[0], Y_g[0], "o--", color="black", label="VEM k=1")
    plt.loglog(H_dof[1], Y_g[1], "o--", color="red", label="Simplex k=1")
    plt.loglog(H_dof[2], Y_g[2], "o--", color="red", label="Staudtian k=1")
    plt.ylabel("Relative error (hyperbolic metric)")
    plt.xlabel("# DOF")
    plt.legend()
    plt.show()