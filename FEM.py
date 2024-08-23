import numpy as np

class Model:
    def __init__(self, domain="rectangle", bounds=np.array([[-1, 1], [-1, 1]]), resolution=[21, 21], isTraceFixed=True, computeSpectrumOnBake=False):
        self.bounds = bounds
        self.resolution = resolution
        self.domain = domain
        self.isTraceFixed = isTraceFixed

        self._vertices_normalized = []
        self._exclude = []
        self._identify = [[], []]
        self._elements = []

        self.vertices = []
        self.polygons = []
        self.triangles = []
        self.L = np.array([[]])
        self.M = np.array([[]])
        self._mask = []

        self.eigenvectors = []
        self.eigenvalues = []
        self.computeSpectrumOnBake = computeSpectrumOnBake

        self.bake()

    def bake(self):
        self._bake_domain()
        self._bake_data()
        self._bake_matrices()

        if self.computeSpectrumOnBake:
            self.bake_spectrum()

    def _bake_domain(self):
        W = self.resolution[0]
        H = self.resolution[1]

        self._vertices_normalized = np.zeros([2, W * H])
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))

        self._vertices_normalized[0, :] = X.flatten()
        self._vertices_normalized[1, :] = Y.flatten()

        self.polygons = []
        for y_i in range(H - 1):
            for x_i in range(W - 1):
                i = W * y_i + x_i
                self.polygons.append([i, i + W + 1, i + W])
                self.polygons.append([i, i + 1, i + W + 1])

        if self.domain == "elliptic disk":
            for v in self._vertices_normalized.T:
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

    def _bake_data(self):
        bounds_offset = self.bounds[:, 0] - np.array([-1, -1])
        bounds_mat = 0.5 * np.array([
            [self.bounds[0, 1] - self.bounds[0, 0], 0],
            [0, self.bounds[1, 1] - self.bounds[1, 0]]
        ])
        self.vertices = np.reshape(bounds_offset, [2, 1]) + bounds_mat @ self._vertices_normalized

        self.triangles = self.polygons

    def _bake_matrices(self):
        n = 0
        elements = []
        self._mask = []
        self._area = []

        for i in range(np.size(self.vertices, axis=1)):
            if i in self._exclude:
                elements.append(-1)
            else:
                elements.append(n)
                self._mask.append(i)
                n += 1

        self._elements = np.array(elements)
        self._elements[self._identify[0]] = self._elements[self._identify[1]]

        self.L = np.zeros([n, n])
        self.M = np.zeros([n, n])

        for v_i, v_j, v_k in self.polygons:
            v1 = self.vertices[:, v_k] - self.vertices[:, v_j]
            v2 = self.vertices[:, v_i] - self.vertices[:, v_k]
            v3 = self.vertices[:, v_j] - self.vertices[:, v_i]
            Jac_A = np.abs(v1[0] * v2[1] - v2[0] * v1[1])

            m = Jac_A / 24
            l = .5 / Jac_A

            i, j, k = self._elements[v_i], self._elements[v_j], self._elements[v_k]

            if i >= 0:
                self.M[i, i] += m * 2
                self.L[i, i] += l * np.dot(v1, v1)
                if j >= 0:
                    self.M[(i, j), (j, i)] += m
                    self.L[(i, j), (j, i)] += l * np.dot(v1, v2)
                if k >= 0:
                    self.M[(i, k), (k, i)] += m
                    self.L[(i, k), (k, i)] += l * np.dot(v1, v3)

            if j >= 0:
                self.M[j, j] += m * 2
                self.L[j, j] += l * np.dot(v2, v2)
                if k >= 0:
                    self.M[(j, k), (k, j)] += m
                    self.L[(j, k), (k, j)] += l * np.dot(v2, v3)

            if k >= 0:
                self.M[k, k] += m * 2
                self.L[k, k] += l * np.dot(v3, v3)

    def solve_poisson(self, f):
        u = np.zeros(np.size(self.vertices, axis=1), dtype=complex)

        b = self.M @ f(self.vertices[0, self._mask] + 1j * self.vertices[1, self._mask])
        u[self._mask] = np.linalg.solve(self.L, b)

        u[self._identify[0]] = u[self._identify[1]]
        return u

    def bake_spectrum(self):
        u = np.zeros((len(self._mask), np.size(self.vertices, axis=1)), dtype=complex)

        self.eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(self.L) @ self.M)

        u[:, self._mask] = eigenvectors
        self.eigenvectors = u / np.reshape(np.sqrt(np.real(
            np.sum((self.M @ eigenvectors) * np.conj(eigenvectors), axis=1))), [len(self.eigenvalues), 1])