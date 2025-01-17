import numpy as np
import matplotlib.pyplot as plt
import plot, FEM_BKDISK_O1, FEM_BKDISK_O2, FEM_PDISK_O1, FEM_PDISK_O2, triangulate
import FEM_SIMPLEX_O1, FEM_HOMOGENEOUS_O1, FEM_STAUDTIAN_O1


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

R = np.tanh(1.5)
R_k = np.tanh(3.0)
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
#v = lambda z: c - g0(z * np.conj(z))

#v = lambda z: R * R - z * np.conj(z)
#f = lambda z: np.power(1 - z * np.conj(z), 2)

v_k = lambda z: v(z / (1 + np.sqrt(1 - z * np.conj(z))))
f_p = lambda z: v(2 * z / (1 + z * np.conj(z)))

#f = lambda z: 1

V = [v, v, v, v, v]
F = [f, f, f, f, f]

H_dof = [[] for _ in range(len(F))]
H_g = [[] for _ in range(len(F))]
Y_e = [[] for _ in range(len(F))]
Y_g = [[] for _ in range(len(F))]

#for triangulation_f in ["../meshgen/output/triangulation_euc_256(42).txt",
#                        "../meshgen/output/triangulation_euc_512(68).txt"]
#                        "../meshgen/output/triangulation_euc_1024(89).txt",
#                        "../meshgen/output/triangulation_euc_2048(144).txt",
#                        "../meshgen/output/triangulation_euc_4096(230).txt",
#                        "../meshgen/output/triangulation_euc_8192(288).txt"]:

_bdry_N = [100, 400, 900]
_bdry_i = 0
for triangulation_f in ["../meshgen/output/triangulation_rect_hyp_180(80).txt",
                        "../meshgen/output/triangulation_rect_hyp_560(160).txt",
                        "../meshgen/output/triangulation_rect_hyp_1100(200).txt"]:

    vertices = []
    triangles = []
    file = open(triangulation_f)
    exec(file.read())
    vertices = np.array(vertices).reshape((len(vertices) // 2, 2)).T
    file.close()

    boundary = [i for i in range(_bdry_N[_bdry_i], np.size(vertices, axis=1), 1)]
    _bdry_i += 1

    vertices_k = triangulate._pdisk_to_bkdisk(vertices)
    _triangles = []
    for i in range(0, len(triangles), 3):
        _triangles.append([triangles[i], triangles[i + 1], triangles[i + 2]])

    models = [
        FEM_SIMPLEX_O1.Model(vertices_k, _triangles, boundary, int_res=100),
        FEM_STAUDTIAN_O1.Model(vertices_k, _triangles, boundary, int_res=100),
        FEM_HOMOGENEOUS_O1.Model(vertices_k, _triangles, boundary, int_res=100),
        FEM_BKDISK_O1.Model(vertices_k, _triangles, boundary),
        FEM_BKDISK_O2.Model(vertices_k, _triangles, boundary)
    ]

    h = compute_h(vertices, _triangles)
    for i in range(len(models)):
        u = np.real(models[i].solve_poisson(F[i]))
        H_dof[i].append(models[i]._n_elements)
        H_g[i].append(h)
        Y_e[i].append(models[i].compare(V[i], "L2"))
        Y_g[i].append(models[i].compare(V[i], "L2_g"))

print("H_g =", H_g)
print("Y_e =")
for y in Y_e:
    print(y)

print("Y_g =")
for y in Y_g:
    print(y)

if True:

    plt.loglog(H_dof[0], Y_g[0], "o--", color="black", label="Simplex k=1")
    plt.loglog(H_dof[1], Y_g[1], "<--", color="black", label="Staudtian k=1")
    plt.loglog(H_dof[2], Y_g[2], "x--", color="gray", label="Homogeneous k=1")
    plt.loglog(H_dof[3], Y_g[3], "o--", color="black", label="Klein k=1")
    plt.loglog(H_dof[4], Y_g[4], "o--", color="black", label="Klein k=2")
    plt.ylabel("Relative error (hyperbolic metric)")
    plt.xlabel("# DOF")
    plt.legend()
    plt.show()

    plt.loglog(H_g[0], Y_g[0], "o--", color="black", label="Simplex k=1")
    plt.loglog(H_g[1], Y_g[1], "<--", color="black", label="Staudtian k=1")
    plt.loglog(H_g[2], Y_g[2], "x--", color="gray", label="Homogeneous k=1")
    plt.loglog(H_g[3], Y_g[3], "o--", color="black", label="Klein k=1")
    plt.loglog(H_g[4], Y_g[4], "o--", color="black", label="Klein k=2")
    plt.ylabel("Relative error (hyperbolic metric)")
    plt.xlabel("Mesh size (hyperbolic metric)")
    plt.legend()
    plt.show()