import numpy as np
import matplotlib.pyplot as plt
import plot

def radius(p, q):
    a = np.tan(np.pi * (0.5 - 1.0 / q))
    b = np.tan(np.pi / p)
    return np.sqrt((a - b) / (a + b))

# must satisfy (p - 2) * (q - 2) > 4
p = 5
q = 5

r = radius(p, q)

angle = 2 * np.pi / p
X = [r * np.cos((k + 0.5) * angle) for k in range(p)]
Y = [r * np.sin((k + 0.5) * angle) for k in range(p)]
triangles = [[i for i in range(p)]]
flags = [[False for i in range(p)]]

l0 = 0.5 / np.cos(0.5 * angle) * (r + 1 / r)
r0 = l0 * l0 - 1

for _ in range(5):
    N = len(triangles)
    c = 0

    for k in range(p):
        m1 = l0 * np.cos(k * angle)
        m2 = l0 * np.sin(k * angle)

        for i in range(0, N):
            if flags[i][k]:
                continue

            poly = []
            for j in range(p):
                u1 = X[triangles[i][j]] - m1
                u2 = Y[triangles[i][j]] - m2
                s = r0 / (u1 * u1 + u2 * u2)

                X.append(m1 + s * u1)
                Y.append(m2 + s * u2)
                poly.append(3 * N + c + j)

            triangles.append(poly)
            flags.append([_i == k for _i in range(p)])
            flags[i][k] = True
            c += p

def pdisk_to_bkdisk(x, y):
    s = 0.5 * (1 + x * x + y * y)
    return x / s, y / s

X, Y = pdisk_to_bkdisk(np.array(X), np.array(Y))

print(len(triangles))

ax = plt.figure().add_subplot()
plot.add_wireframe(ax, X, Y, triangles)
plt.scatter(X, Y, s=5)
plt.axis("equal")
plt.show()
