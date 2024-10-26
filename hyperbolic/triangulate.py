import numpy as np
import matplotlib.pyplot as plt

def radius(p, q):
    a = np.tan(np.pi * (0.5 - 1.0 / q))
    b = np.tan(np.pi / p)
    return np.sqrt((a - b) / (a + b))

# must satisfy (p - 2) * (q - 2) > 4
p = 3
q = 7

r = radius(p, q)

angle = 2 * np.pi / p

X = [r * np.cos((k + 0.5) * angle) for k in range(p)]
Y = [r * np.sin((k + 0.5) * angle) for k in range(p)]

s0 = 0.5 * (r + 1 / r) / np.cos(0.5 * angle)
r_i = s0 * s0 - 1

for _ in range(3):
    N = np.size(X)
    for k in range(p):
        o_i = np.array([s0 * np.cos(k * angle), s0 * np.sin(k * angle)])
        for i in range(N):
            u = np.array([X[i] - o_i[0], Y[i] - o_i[1]])
            s = r_i / (u[0] * u[0] + u[1] * u[1])

            X.append(o_i[0] + s * u[0])
            Y.append(o_i[1] + s * u[1])


def pdisk_to_bkdisk(x, y):
    for i in range(len(x)):
        s = 0.5 * (1 + x[i] * x[i] + y[i] * y[i])
        x[i] /= s
        y[i] /= s

    return x, y

#X, Y = pdisk_to_bkdisk(X, Y)

plt.scatter(X, Y)
plt.axis("equal")
plt.show()
