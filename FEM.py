import numpy as np

import mesh
import orbifolds


class Model:
    def __init__(self, domain="rectangle", bounds=np.array([[-1, 1], [-1, 1]]), resolution=[21, 21]
                 , isTraceFixed=True, computeSpectrumOnBake=False):
        self.bounds = bounds
        self.resolution = resolution
        self.domain = domain
        self.isTraceFixed = isTraceFixed

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
        self._bake_triangles()
        self._bake_matrices()

        if self.computeSpectrumOnBake:
            self.bake_spectrum()

    def _bake_domain(self):
        self._identify = [[], []]
        self._exclude = []

        W = self.resolution[0]
        H = self.resolution[1]
        trace = []

        if self.domain in orbifolds.orbit_sgn:
            self.vertices, self.polygons, trace = orbifolds.mesh(self.domain, W, H)
            self._identify = orbifolds.compute_idmap(self.domain, W, H)
            self._exclude.extend(self._identify[0])

            trace = [0]

        else:
            self.vertices, self.polygons, trace = mesh.generic(self.domain, W, H)

        if self.isTraceFixed:
            self._exclude.extend(trace)


    def _bake_triangles(self):
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
                    self.M[i, j] += m
                    self.M[j, i] += m
                    self.L[i, j] += l * np.dot(v1, v2)
                    self.L[j, i] += l * np.dot(v1, v2)
                if k >= 0:
                    self.M[i, k] += m
                    self.M[k, i] += m
                    self.L[i, k] += l * np.dot(v1, v3)
                    self.L[k, i] += l * np.dot(v1, v3)

            if j >= 0:
                self.M[j, j] += m * 2
                self.L[j, j] += l * np.dot(v2, v2)
                if k >= 0:
                    self.M[j, k] += m
                    self.M[k, j] += m
                    self.L[j, k] += l * np.dot(v2, v3)
                    self.L[k, j] += l * np.dot(v2, v3)

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

    def __str__(self):
        return f"FEM-{self.domain}-{self.resolution[0]}x{self.resolution[1]}"