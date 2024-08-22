import numpy as np
import matplotlib.pyplot as plt

import VEM
import plot

N = 21
vem = VEM.Model()

vem.vertices = np.zeros([2, N*N])
X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

vem.vertices[0, :] = X.flatten()
vem.vertices[1, :] = Y.flatten()

vem.trace.extend(range(N))
vem.trace.extend(range(N * (N-1), N*N))
vem.trace.extend(range(N, N * (N-1), N))
vem.trace.extend(range(2*N - 1, N*N - 1, N))

for y_i in range(N-1):
    for x_i in range(N-1):
        i = N * y_i + x_i
        vem.polygons.append((i, i+1, i+N + 1, i+N))

u = vem.solve_poisson(lambda z: np.exp(-25 * z * np.conj(z)))

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, np.reshape(u, [N, N]))

plt.show()