import numpy as np
import matplotlib.pyplot as plt

N = 21

vertices = np.zeros([N*N, 2])
triangles = []
elements = []
elements_mask = []
trace = []

X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

vertices[:, 0] = X.flatten()
vertices[:, 1] = Y.flatten()

for v in vertices:
    if abs(v[0]) > abs(v[1]):
        v /= np.linalg.norm(v / v[0])
    elif v[1] != 0:
        v /= np.linalg.norm(v / v[1])

trace.extend(range(N))
trace.extend(range(N * (N-1), N*N))
trace.extend(range(N, N * (N-1), N))
trace.extend(range(2*N - 1, N*N - 1, N))

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

s = 2.5
r2 = s * (np.square(vertices[elements_mask, 0]) + np.square(vertices[elements_mask, 1]))
f[elements_mask] = 4 * s * np.exp(-r2) * (1 - r2)
u[elements_mask] = np.linalg.solve(L, M @ f[elements_mask])

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], triangles, u)

print(np.max(u))

plt.matshow(M)
plt.matshow(L)
plt.show()