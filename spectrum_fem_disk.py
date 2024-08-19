import matplotlib.pyplot as plt
import numpy as np

import FEM
import plot

N = 21
fem = FEM.Model()

fem.vertices = np.zeros([N*N, 2])

X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

fem.vertices[:, 0] = X.flatten()
fem.vertices[:, 1] = Y.flatten()

for v in fem.vertices:
    if abs(v[0]) > abs(v[1]):
        v /= np.linalg.norm(v / v[0])
    elif v[1] != 0:
        v /= np.linalg.norm(v / v[1])

fem.trace.extend(range(N))
fem.trace.extend(range(N * (N-1), N*N))
fem.trace.extend(range(N, N * (N-1), N))
fem.trace.extend(range(2*N - 1, N*N - 1, N))
fem.trace = []

for y_i in range(N-1):
    for x_i in range(N-1):
        i = N * y_i + x_i
        fem.triangles.append((i, i+N + 1, i+N))
        fem.triangles.append((i, i + 1, i+N + 1))


u = np.zeros(N * N, dtype=complex)
f = np.zeros(N * N, dtype=complex)

s = 2.5
f = lambda z: z
#r2 = s * (np.square(vertices[elements_mask, 0]) + np.square(vertices[elements_mask, 1]))
#f[elements_mask] = 4 * s * np.exp(-r2) * (1 - r2)

eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(L) @ M)

l, v = eigenvalues[0], eigenvectors[0]
norm = np.sqrt(np.real(np.dot(M @ v, np.conj(v))))

u[elements_mask] = v / norm

#ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles, abs_z, alpha=0.75, cmap=cm.Greys)
#ax.tricontourf(vertices[:, 0], vertices[:, 1], arg_z, triangles=triangles, zdir='z', offset=0, cmap='coolwarm')

#ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 2),
#       xlabel='X', ylabel='Y', zlabel='Z')


plot.complex(vertices, triangles, u)

plt.matshow(M)
plt.matshow(L)
plt.show()