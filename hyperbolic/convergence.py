import numpy as np
import matplotlib.pyplot as plt
import plot, FEM_BKDISK_O1, FEM_BKDISK_O2, FEM_PDISK_O1, FEM_PDISK_O2, triangulate
import FEM_SIMPLEX_O1, FEM_HOMOGENEOUS_O1, FEM_STAUDTIAN_O1
import VEM_PDISK_O1, VEM_HOMOGENEOUS_O1, vem_triangulate


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

#def v(z):
#    x2 = np.real(z) * np.real(z)
#    y2 = np.imag(z) * np.imag(z)
#    return (W2 - x2) * (W2 - y2)
#def f(z):
#    x2 = np.real(z) * np.real(z)
#    y2 = np.imag(z) * np.imag(z)
#    L = -2 * x2 * (1 - x2) * (W2 - y2) + 2 * x2 * y2 * (W2 - x2) \
#        -2 * y2 * (1 - y2) * (W2 - x2) + 2 * x2 * y2 * (W2 - y2) \
#        + (1 - x2 - y2) * (
#        -2 * (1 - x2) * (W2 - y2) + 4 * x2 * (W2 - y2) + 2 * y2 * (W2 - x2) - 4 * x2 * y2
#        -2 * (1 - y2) * (W2 - x2) + 4 * y2 * (W2 - x2) + 2 * x2 * (W2 - y2) - 4 * x2 * y2
#        )
#    return -L

g0 = lambda t: -np.log(1 - t)
c = g0(R * R)
v_p = lambda z: c - g0(z * np.conj(z))

#v_p = lambda z: R * R - z * np.conj(z)
#f = lambda z: np.power(1 - z * np.conj(z), 2)

v = lambda z: v_p(z / (1 + np.sqrt(1 - z * np.conj(z))))
#f_p = lambda z: v(2 * z / (1 + z * np.conj(z)))

f = lambda z: 1

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

_bdry_N = [256-110, 512-131, 1024-185, 2048-288, 4096-377, 8192-610]
_bdry_i = 0
#for triangulation_f in ["../meshgen/output/triangulation_rect_hyp_180(80).txt",
#                        "../meshgen/output/triangulation_rect_hyp_560(160).txt",
#                        "../meshgen/output/triangulation_rect_hyp_1100(200).txt"]:
for N in [256, 512, 1024, 2048, 4096, 8192]:
    vertices, polygons, boundary = triangulate.load(f"./triangulations/uniform_disk_hyp_{N}.npz")
    vertices_vem, polygons_vem, boundary_vem = vem_triangulate.vem_mesh(N)

    vertices_k = triangulate._pdisk_to_bkdisk(vertices)
    vertices_vem = triangulate._pdisk_to_bkdisk(vertices_vem)

    res = 100
    models = [
        FEM_STAUDTIAN_O1.Model(vertices_k, polygons, boundary),
        FEM_SIMPLEX_O1.Model(vertices_k, polygons, boundary),
        VEM_PDISK_O1.Model(vertices_vem, polygons_vem, boundary_vem),
        VEM_HOMOGENEOUS_O1.Model(vertices_k, polygons, boundary)
    ]

    h = compute_h(vertices, polygons)
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

    plt.loglog(H_dof[0], Y_g[0], "o--", color="black", label="VEM k=1")
    plt.loglog(H_dof[1], Y_g[1], "o--", color="red", label="Simplex k=1")
    plt.ylabel("Relative error (hyperbolic metric)")
    plt.xlabel("# DOF")
    plt.legend()
    plt.show()

    plt.loglog(H_g[0], Y_g[0], "o--", color="black", label="VEM k=1")
    plt.loglog(H_g[1], Y_g[1], "o--", color="red", label="Simplex k=1")
    plt.ylabel("Relative error (hyperbolic metric)")
    plt.xlabel("Mesh size (hyperbolic metric)")
    plt.legend()
    plt.show()