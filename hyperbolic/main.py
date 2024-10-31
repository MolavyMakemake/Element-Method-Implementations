import numpy as np
import matplotlib.pyplot as plt
import plot, FEM_BKDISK, FEM_PDISK, triangulate

model_bk = FEM_BKDISK.Model(
    *triangulate.generate(p=3, q=7, iterations=3, subdivisions=3, model="Klein")
)
model_p = FEM_PDISK.Model(
    *triangulate.generate(p=3, q=7, iterations=3, subdivisions=3, model="Poincare")
)

print(model_bk.area())
print(model_p.area())

norm_bk = np.max(np.sum(model_bk.vertices * model_bk.vertices, axis=0))
norm_p = np.max(np.sum(model_p.vertices * model_p.vertices, axis=0))

#f_bk = lambda z: np.atanh(np.abs(z)) < .5
#f_p = lambda z: 2 * np.atanh(np.abs(z)) < .5
f_bk = lambda z: 1
f_p = lambda z: 1

u_bk = np.real(model_bk.solve_poisson(f_bk))
u_p = np.real(model_p.solve_poisson(f_p))

fig = plt.figure(figsize=plt.figaspect(0.5))

#model_bk.vertices = triangulate._bkdisk_to_pdisk(model_bk.vertices)
ax = fig.add_subplot(1, 2, 1, projection="3d")
plot.surface(ax, model_bk.vertices, model_bk.triangles, u_bk, label="Klein")
plot.add_wireframe(ax, model_bk.vertices, model_bk.triangles, u_bk)
plt.legend()

ax = fig.add_subplot(1, 2, 2, projection="3d")
plot.surface(ax, model_p.vertices, model_p.triangles, u_p, label="Poincare")
plot.add_wireframe(ax, model_p.vertices, model_p.triangles, u_p)
plt.legend()
plt.show()