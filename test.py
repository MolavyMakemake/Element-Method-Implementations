import numpy as np
import matplotlib.pyplot as plt
from hyperbolic.triangulate import save

X = 2 * np.random.rand(2, 10000) - 1

for i in range(np.size(X, axis=1)):
    X[:, i] *= X[:, i] @ X[:, i] < 1

Y = np.zeros_like(X)
Y[0, :] = X[0, :] / np.sqrt(1 - X[1, :] * X[1, :])
Y[1, :] = X[1, :] / np.sqrt(1 - X[0, :] * X[0, :])

plt.scatter(Y[0, :], Y[1, :])
plt.axis("equal")
plt.show()