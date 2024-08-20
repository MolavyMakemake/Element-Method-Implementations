import numpy as np
import matplotlib.pyplot as plt

import FEM
import plot

N = 13
fem = FEM.Model()

fem.vertices = np.zeros([N*N, 2])
X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

fem.vertices[:, 0] = X.flatten()
fem.vertices[:, 1] = Y.flatten()

fem.trace = [0]
fem.trace.extend(range(N * (N-1), N*N))
fem.trace.extend(range(N - 1, N*N, N))

for y_i in range(N-1):
    for x_i in range(N-1):
        i = N * y_i + x_i
        fem.triangles.append((i, i+N + 1, i+N))
        fem.triangles.append((i, i + 1, i+N + 1))

for i in range(1, N-1):
    fem.identify.append([i, N * (N-1) + i])
    fem.identify.append([N * i, N * (i+1) - 1])


f = lambda z: 2 * np.pi ** 2 * np.sin(np.pi * np.real(z)) * np.sin(np.pi * np.imag(z))

u = fem.solve_poisson(f)
u[range(N * (N-1), N*N)] = u[range(0, N)]
u[range(N - 1, N*N, N)] = u[range(0, N*N, N)]

plot.surface(fem.vertices, fem.triangles, u)
plt.show()