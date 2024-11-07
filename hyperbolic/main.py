import numpy as np
import matplotlib.pyplot as plt
import plot, FEM_PDISK_O1, FEM_PDISK_O2, FEM_BKDISK_O1, FEM_BKDISK_O2, triangulate

vertices_p, triangles, boundary = triangulate.load("37_10s1__poincare__95.npz")
vertices_bk = triangulate._pdisk_to_bkdisk(vertices_p)

models = (
    FEM_PDISK_O1.Model(vertices_p, triangles, boundary),
    FEM_PDISK_O2.Model(vertices_p, triangles, boundary),
    FEM_BKDISK_O1.Model(vertices_bk, triangles, boundary),
    FEM_BKDISK_O2.Model(vertices_bk, triangles, boundary)
)

print(models[0].area())
print(models[1].area())

f = (
    lambda z: 1,
    lambda z: 1,
    lambda z: 1,
    lambda z: 1
)

labels = ["Poincaré k=1", "Poincaré k=2", "Klein k=1", "Klein k=2"]

fig = plt.figure(figsize=plt.figaspect(1.0))
for i in range(len(models)):
    u = np.real(models[i].solve_poisson(f[i]))

    ax = fig.add_subplot(2, 2, i+1, projection="3d")
    plot.surface(ax, models[i].vertices, models[i].triangles, u, label=labels[i])
    plot.add_wireframe(ax, models[i].vertices, models[i].triangles, u)

plt.show()