import numpy as np

class Integrator:
    def __init__(self, res, open=False):
        self.vertices = np.zeros([2, res * (res + 1) // 2])

        y = np.repeat(range(res), range(res, 0, -1))
        x = np.arange(res * (res + 1) // 2) - res * y + ((y - 1) * y) // 2

        self.vertices[0, :] = x / (res - 1)
        self.vertices[1, :] = y / (res - 1)

        self.x = x
        self.y = y

        self.weight = np.zeros(dtype=float, shape=(np.size(self.vertices, axis=1)))

        i = 0
        while i < res * (res + 1) // 2 - 1:
            W = int(res - y[i])

            if x[i] >= W - 2:
                self.weight[[i, i + 1, i + W]] += 1
                i += 2
            else:
                self.weight[[i, i + 1, i + W]] += 1
                self.weight[[i + 1, i + 1 + W, i + W]] += 1
                i += 1

        h = 1.0 / (res - 1.0)
        self.weight *= h * h / 6.0

        if open:
            e = 1e-3 / (res - 1)
            self.vertices = e + self.vertices * (1 - 3 * e)

    def integrate(self, F, axis=0):
        return np.sum(F(self.vertices[0, :], self.vertices[1, :]) * self.weight, axis=axis)

    def integrate_vector(self, v):
        return np.sum(v * self.weight, axis=0)

    def mask(self, arg):
        if arg == "l":
            return self.x > 0