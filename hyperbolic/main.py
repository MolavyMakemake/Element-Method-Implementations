import numpy as np
import matplotlib.pyplot as plt
import plot, FEM, triangulate

vertices, polygons, trace = triangulate.generate(p=3, q=7, iterations=5, model="Klein")
model = FEM.Model(vertices, polygons, trace)

f = lambda z: z * np.conj(z)
u = np.real(model.solve_poisson(f))

ax = plt.figure().add_subplot(projection="3d")
plot.surface(ax, model.vertices, model.triangles, u)
plot.add_wireframe(ax, model.vertices, model.triangles, u)
ax.scatter(vertices[0, trace], vertices[1, trace], u[trace])
plt.show()
