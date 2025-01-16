import numpy as np
import matplotlib.pyplot as plt
import plot, triangulate
import FEM_PDISK_O1, FEM_PDISK_O2, FEM_BKDISK_O1, FEM_BKDISK_O2
import FEM_SIMPLEX_O1, FEM_STAUDTIAN_O1, FEM_HOMOGENEOUS_O1

vertices_p, triangles, boundary = triangulate.load("./triangulations/uniform_disk_hyp_512.npz")
vertices_k = triangulate._pdisk_to_bkdisk(vertices_p)

models = (
    FEM_HOMOGENEOUS_O1.Model(vertices_k, triangles, boundary),
    FEM_STAUDTIAN_O1.Model(vertices_k, triangles, boundary),
    FEM_SIMPLEX_O1.Model(vertices_p, triangles, boundary)
)
print(models[0].area())
#print(models[1].area())

f = (
    lambda z: 1,
    lambda z: 1,
    lambda z: 1,
    lambda z: 1
)

labels = ["Poincaré k=2", "Staudtian k=1", "Simplex k=1"]

fig = plt.figure(figsize=plt.figaspect(1.0))
for i in range(len(models)):
    u = np.real(models[i].solve_poisson(f[i]))

    ax = fig.add_subplot(1, 3, i+1, projection="3d")
    ax.set_title(labels[i])
    plot.surface(ax, vertices_p, models[i].triangles, u)
    plot.add_wireframe(ax, vertices_p, models[i].triangles, u)

plt.show()