import numpy as np
import matplotlib.pyplot as plt
import plot, triangulate
import FEM_PDISK_O1, FEM_PDISK_O2, FEM_BKDISK_O1, FEM_BKDISK_O2
import FEM_SIMPLEX_O1, FEM_STAUDTIAN_O1, FEM_HOMOGENEOUS_O1

vertices_p, triangles, boundary = triangulate.load("./triangulations/uniform_rect_1100.npz")
vertices_k = triangulate._pdisk_to_bkdisk(vertices_p)

models = (
    FEM_HOMOGENEOUS_O1.Model(vertices_k, triangles, boundary),
    FEM_STAUDTIAN_O1.Model(vertices_k, triangles, boundary),
    FEM_SIMPLEX_O1.Model(vertices_k, triangles, boundary),
    FEM_BKDISK_O1.Model(vertices_k, triangles, boundary)
)
print(models[0].area())
#print(models[1].area())

W = np.tanh(.86)
W2 = W * W

def F(z):
    x2 = np.real(z) * np.real(z)
    y2 = np.imag(z) * np.imag(z)
    L = -2 * x2 * (1 - x2) * (W2 - y2) + 2 * x2 * y2 * (W2 - x2) \
        -2 * y2 * (1 - y2) * (W2 - x2) + 2 * x2 * y2 * (W2 - y2) \
        + (1 - x2 - y2) * (
        -2 * (1 - x2) * (W2 - y2) + 4 * x2 * (W2 - y2) + 2 * y2 * (W2 - x2) - 4 * x2 * y2
        -2 * (1 - y2) * (W2 - x2) + 4 * y2 * (W2 - x2) + 2 * x2 * (W2 - y2) - 4 * x2 * y2
        )
    return -L

f = (
    F,
    F,
    F,
    F
)

labels = ["Homogeneous k=1", "Staudtian k=1", "Simplex k=1", "Klein k=1"]

fig = plt.figure(figsize=plt.figaspect(1.0))
for i in range(len(models)):
    u = np.real(models[i].solve_poisson(f[i]))
    r = vertices_k * vertices_k
    #u -= (W2 - r[0, :]) * (W2 - r[1, :])

    ax = fig.add_subplot(2, 2, i+1, projection="3d")
    ax.set_title(labels[i])
    plot.surface(ax, vertices_k, models[i].triangles, u)
    plot.add_wireframe(ax, vertices_k, models[i].triangles, u)

plt.show()