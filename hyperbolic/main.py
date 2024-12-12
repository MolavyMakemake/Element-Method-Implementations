import numpy as np
import matplotlib.pyplot as plt
import plot, FEM_PDISK_O1, FEM_PDISK_O2, FEM_BKDISK_O1, FEM_BKDISK_O2, FEM_SIMPLEX_O1, triangulate

vertices_p, triangles, boundary = triangulate.generate(p=3, q=7, iterations=3, subdivisions=3, model="Poincare", minimal=True)
vertices_bk = triangulate._pdisk_to_bkdisk(vertices_p)

models = (
    FEM_PDISK_O2.Model(vertices_p, triangles, boundary),
    FEM_BKDISK_O2.Model(vertices_bk, triangles, boundary),
    FEM_SIMPLEX_O1.Model(vertices_p, triangles, boundary)
)

#print(models[0].area())
#print(models[1].area())

f = (
    lambda z: 1,
    lambda z: 1,
    lambda z: 1,
    lambda z: 1
)

labels = ["Poincaré k=2", "Klein k=2", "Simplex k=1"]

fig = plt.figure(figsize=plt.figaspect(1.0))
for i in range(len(models)):
    u = np.real(models[i].solve_poisson(f[i]))

    ax = fig.add_subplot(1, 3, i+1, projection="3d")
    ax.set_title(labels[i])
    plot.surface(ax, models[i].vertices, models[i].triangles, u)
    plot.add_wireframe(ax, models[i].vertices, models[i].triangles, u)

plt.show()