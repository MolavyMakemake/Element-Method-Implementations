import numpy as np

class Model:
    def __init__(self, domain="rectangle", bounds=np.array([[-1, 1], [-1, 1]]), resolution=[21, 21], isTraceFixed=True):
        self.bounds = bounds
        self.resolution = resolution
        self.domain = domain
        self.isTraceFixed = isTraceFixed

        self.vertices = []
        self.polygons = []
        self._exclude = []
        self._identify = [[], []]

        self.bake()

    def bake(self):
        W = self.resolution[0]
        H = self.resolution[1]

        self.vertices = np.zeros([2, W * H])
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))

        self.vertices[0, :] = X.flatten()
        self.vertices[1, :] = Y.flatten()

        self.polygons = []
        for y_i in range(H - 1):
            for x_i in range(W - 1):
                i = W * y_i + x_i
                self.polygons.append((i, i + 1, i + W + 1, i + W))

        if self.domain == "elliptic disk":
            for v in self.vertices.T:
                if abs(v[0]) > abs(v[1]):
                    v /= np.linalg.norm(v / v[0])
                elif v[1] != 0:
                    v /= np.linalg.norm(v / v[1])

        self._identify = [[], []]
        if self.domain == "torus":
            for i in range(0, W - 1):
                self.identify(W * (H - 1) + i, i)
            for i in range(0, H - 1):
                self.identify(W * (i + 1) - 1, W * i)

            self.identify(W * H - 1, 0)

        self.bake_trace()

    def bake_trace(self):
        W = self.resolution[0]
        H = self.resolution[1]

        self._exclude = self._identify[0].copy()

        if not self.isTraceFixed:
            return

        if self.domain in ["rectangle", "elliptic disk"]:
            self._exclude.extend(range(W))
            self._exclude.extend(range(W * (H - 1), W * H))
            self._exclude.extend(range(W, W * (H - 1), W))
            self._exclude.extend(range(2 * W - 1, W * H - 1, W))

        elif self.domain == "torus":
            self._exclude.append(0)

    def identify(self, i, j):
        self._identify[0].append(i)
        self._identify[1].append(j)

    def basis(self, vertices, f):
        n = 0
        elements = []
        elements_mask = []

        for i in range(np.size(vertices, axis=1)):
            if i in self.trace:
                elements.append(-1)
            else:
                elements.append(n)
                elements_mask.append(i)
                n += 1

        elements = np.array(elements)
        elements[self._identify[0]] = elements[self._identify[1]]

        L = np.zeros([n, n], dtype=complex)
        b = np.zeros(n, dtype=complex)

        rot = np.array([[0, 1], [-1, 0]])

        for p_i in self.polygons:
            p1 = self.vertices[:, p_i]
            p0 = np.roll(p1, 1, axis=1)
            p2 = np.roll(p1, -1, axis=1)

            N = np.size(p1, axis=1)

            d0 = np.linalg.norm(p1 - p0, axis=0)
            d1 = np.roll(d0, -1)

            area = 0.5 * np.abs(np.dot(p1[0, :], p0[1, :]) - np.dot(p0[0, :], p1[1, :]))  # shoelace
            circumference = np.sum(d0)

            print("area:", area, ", circumference:", circumference)
            print("points:\n", p1, "\n")

            I_f = area * np.average(f(p1[0, :] + 1j * p1[1, :]))

            proj_D = rot @ (p2 - p0) / (2 * area)
            proj_a = (d0 + d1 - ((p1 + p0) @ d0) @ proj_D) / (2 * circumference)

            for i in range(N):
                e_i = elements[p_i[i]]
                if e_i < 0:
                    continue

                d_ik = i == np.arange(0, N)
                for j in range(N):
                    e_j = elements[p_i[j]]
                    if e_j < 0:
                        continue

                    # compute local LHS
                    d_jk = j == np.arange(0, N)
                    a = np.dot(proj_D[:, i], proj_D[:, j]) * area
                    s = np.sum((d_ik - proj_a[i] - proj_D[:, i] @ p1) * (d_jk - proj_a[j] - proj_D[:, j] @ p1))

                    L[e_i, e_j] += a + s

                # compute local RHS
                b[e_i] += I_f / N

        return L, b, elements_mask

    def solve_poisson(self, f):
        bounds_offset = self.bounds[:, 0] - np.array([-1, -1])
        bounds_mat = 0.5 * np.array([
            [self.bounds[0, 1] - self.bounds[0, 0], 0],
            [0, self.bounds[1, 1] - self.bounds[1, 0]]
        ])
        vertices = np.reshape(bounds_offset, [2, 1]) + bounds_mat @ self.vertices

        L, b, mask = self.basis(vertices, f)

        print(L)
        print(b)

        u = np.zeros(np.size(self.vertices, axis=1), dtype=complex)
        u[mask] = np.linalg.solve(L, b)

        return vertices[0, :], vertices[0, :], u

    def solve_spectrum(self):
        L, M, mask = self.basis()
        u = np.zeros(len(self.vertices), dtype=complex)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(L) @ M)

        u[mask] = eigenvectors[0]
        return eigenvalues[0], u / np.sqrt(np.real(np.dot(M @ eigenvectors[0], np.conj(eigenvectors[0]))))