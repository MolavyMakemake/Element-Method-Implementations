import numpy as np
import matplotlib.pyplot as plt

N = 13

vertices = np.zeros([N*N, 2])
triangles = []
elements = []
elements_mask = []
trace = []

X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

vertices[:, 0] = X.flatten()
vertices[:, 1] = Y.flatten()

trace = [0]
trace.extend(range(N * (N-1), N*N))
trace.extend(range(N - 1, N*N, N))

for y_i in range(N-1):
    for x_i in range(N-1):
        i = N * y_i + x_i
        triangles.append((i, i+N + 1, i+N))
        triangles.append((i, i + 1, i+N + 1))

n = 0
for i in range(N*N):
    if i in trace:
        elements.append(-1)
    else:
        elements.append(n)
        elements_mask.append(i)
        n += 1

for i in range(1, N-1):
    elements[N * (N-1) + i] = elements[i]
    elements[N * (i+1) - 1] = elements[N * i]


M = np.zeros([n, n])
L = np.zeros([n, n])

for v_i, v_j, v_k in triangles:
    v1 = vertices[v_j] - vertices[v_i]
    v2 = vertices[v_k] - vertices[v_i]
    v3 = vertices[v_j] - vertices[v_k]
    Jac_A = np.abs(v1[0] * v2[1] - v2[0] * v1[1])

    m = Jac_A / 24
    l = .5 / Jac_A

    i, j, k = elements[v_i], elements[v_j], elements[v_k]

    if i >= 0:
        M[i, i] += m * 2
        L[i, i] += l * np.dot(v3, v3)
        if j >= 0:
            M[(i, j), (j, i)] += m
            L[(i, j), (j, i)] += l * np.dot(v3, v2)
        if k >= 0:
            M[(i, k), (k, i)] += m
            L[(i, k), (k, i)] -= l * np.dot(v3, v1)

    if j >= 0:
        M[j, j] += m * 2
        L[j, j] += l * np.dot(v2, v2)
        if k >= 0:
            M[(j, k), (k, j)] += m
            L[(j, k), (k, j)] -= l * np.dot(v1, v2)

    if k >= 0:
        M[k, k] += m * 2
        L[k, k] += l * np.dot(v1, v1)


u = np.zeros(N * N)
f = np.zeros(N * N)

f[elements_mask] = 2 * np.pi ** 2 * np.sin(np.pi * vertices[elements_mask, 0]) \
                   * np.sin(np.pi * vertices[elements_mask, 1])
u[elements_mask] = np.linalg.solve(L, M @ f[elements_mask])
u[range(N * (N-1), N*N)] = u[range(0, N)]
u[range(N - 1, N*N, N)] = u[range(0, N*N, N)]

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles, u)

plt.matshow(M)
plt.matshow(L)
plt.show()