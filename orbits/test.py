import numpy as np
import matplotlib.pyplot as plt
res = 100

vertices = np.zeros([2, res * (res + 1) // 2])

y = np.repeat(range(res), range(res, 0, -1))
x = np.arange(res * (res + 1) // 2) - res * y + ((y - 1) * y) // 2

vertices[0, :] = x / (res - 1)
vertices[1, :] = y / (res - 1)

integral_weight = np.zeros(dtype=float, shape=(np.size(vertices, axis=1)))

i = 0
while i < res * (res + 1) // 2 - 1:
    W = int(res - y[i])

    if x[i] >= W - 2:
        integral_weight[[i, i + 1, i + W]] += 1
        i += 2
    else:
        integral_weight[[i, i + 1, i + W]] += 1
        integral_weight[[i + 1, i + 1 + W, i + W]] += 1
        i += 1

h = 1 / (res - 1)
print("h=", h)

integral_weight *= h * h / 6.0
print(np.sum(vertices[0, :] * integral_weight, axis=0))

#plt.scatter(vertices[0, :], vertices[1, :])
plt.show()