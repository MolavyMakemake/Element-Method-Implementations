import numpy as np

class Model:
    def __init__(self, domain="rectangle", bounds=np.array([[-1, 1], [-1, 1]]), resolution = [21, 21], isTraceFixed = True):
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
                self.polygons.append((i, i + W + 1, i + W))
                self.polygons.append((i, i + 1, i + W + 1))

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

    def basis(self, vertices):
        n = 0
        elements = []
        elements_mask = []

        for i in range(np.size(vertices, axis=1)):
            if i in self._exclude:
                elements.append(-1)
            else:
                elements.append(n)
                elements_mask.append(i)
                n += 1

        elements = np.array(elements)
        elements[self._identify[0]] = elements[self._identify[1]]

        M = np.zeros([n, n])
        L = np.zeros([n, n])

        for v_i, v_j, v_k in self.polygons:
            v1 = vertices[:, v_k] - vertices[:, v_j]
            v2 = vertices[:, v_i] - vertices[:, v_k]
            v3 = vertices[:, v_j] - vertices[:, v_i]
            Jac_A = np.abs(v1[0] * v2[1] - v2[0] * v1[1])

            m = Jac_A / 24
            l = .5 / Jac_A

            i, j, k = elements[v_i], elements[v_j], elements[v_k]

            if i >= 0:
                M[i, i] += m * 2
                L[i, i] += l * np.dot(v1, v1)
                if j >= 0:
                    M[(i, j), (j, i)] += m
                    L[(i, j), (j, i)] += l * np.dot(v1, v2)
                if k >= 0:
                    M[(i, k), (k, i)] += m
                    L[(i, k), (k, i)] += l * np.dot(v1, v3)

            if j >= 0:
                M[j, j] += m * 2
                L[j, j] += l * np.dot(v2, v2)
                if k >= 0:
                    M[(j, k), (k, j)] += m
                    L[(j, k), (k, j)] += l * np.dot(v2, v3)

            if k >= 0:
                M[k, k] += m * 2
                L[k, k] += l * np.dot(v3, v3)

        return M, L, elements_mask

    def solve_poisson(self, f):
        bounds_offset = self.bounds[:, 0] - np.array([-1, -1])
        bounds_mat = 0.5 * np.array([
            [self.bounds[0, 1] - self.bounds[0, 0], 0],
            [0, self.bounds[1, 1] - self.bounds[1, 0]]
        ])
        vertices = np.reshape(bounds_offset, [2, 1]) + bounds_mat @ self.vertices

        M, L, mask = self.basis(vertices)
        u = np.zeros(np.size(vertices, axis=1), dtype=complex)
        b = M @ f(vertices[0, mask] + 1j * vertices[1, mask])

        u[mask] = np.linalg.solve(L, b)
        u[self._identify[0]] = u[self._identify[1]]

        return vertices[0, :], vertices[1, :], u

    def solve_spectrum(self):
        L, M, mask = self.basis()
        u = np.zeros(len(self.vertices), dtype=complex)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(L) @ M)

        u[mask] = eigenvectors[0]
        return eigenvalues[0], u / np.sqrt(np.real(np.dot(M @ eigenvectors[0], np.conj(eigenvectors[0]))))