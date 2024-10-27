import numpy as np
import scipy.sparse as sp

class Model:
    def __init__(self, vertices, triangles, trace
                 , isTraceFixed=True, computeSpectrumOnBake=False):

        self.isTraceFixed = isTraceFixed

        self._trace = trace
        self._exclude = []
        self._identify = [[], []]
        self._elements = []

        self.vertices = vertices
        self.polygons = triangles
        self.triangles = triangles
        self.L = np.array([[]])
        self.M = np.array([[]])
        self._mask = []

        self.k = 10;

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

        if self.isTraceFixed:
            self._exclude.extend(self._trace)


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

        L = np.zeros([n, n])
        M = np.zeros([n, n])
        I = np.zeros(n)

        for v_i, v_j, v_k in self.polygons:
            v1 = self.vertices[:, v_k] - self.vertices[:, v_j]
            v2 = self.vertices[:, v_i] - self.vertices[:, v_k]
            v3 = self.vertices[:, v_j] - self.vertices[:, v_i]
            Jac_A = np.abs(v1[0] * v2[1] - v2[0] * v1[1])

            m = Jac_A / 24
            l = .5 / Jac_A

            i, j, k = self._elements[v_i], self._elements[v_j], self._elements[v_k]

            if i >= 0:
                M[i, i] += m * 2
                L[i, i] += l * np.dot(v1, v1)
                I[i] += m * 4
                if j >= 0:
                    M[i, j] += m
                    M[j, i] += m
                    L[i, j] += l * np.dot(v1, v2)
                    L[j, i] += l * np.dot(v1, v2)
                if k >= 0:
                    M[i, k] += m
                    M[k, i] += m
                    L[i, k] += l * np.dot(v1, v3)
                    L[k, i] += l * np.dot(v1, v3)

            if j >= 0:
                M[j, j] += m * 2
                L[j, j] += l * np.dot(v2, v2)
                I[j] += m * 4
                if k >= 0:
                    M[j, k] += m
                    M[k, j] += m
                    L[j, k] += l * np.dot(v2, v3)
                    L[k, j] += l * np.dot(v2, v3)

            if k >= 0:
                M[k, k] += m * 2
                L[k, k] += l * np.dot(v3, v3)
                I[k] += m * 4

        self.L = sp.csc_matrix(L)
        self.M = sp.csc_matrix(M)
        self.I = I

    def solve_poisson(self, f):
        u = np.zeros(np.size(self.vertices, axis=1), dtype=complex)

        b = f(self.vertices[0, self._mask] + 1j * self.vertices[1, self._mask])
        b = self.M @ b

        u[self._mask] = sp.linalg.spsolve(self.L, b)
        #u -= u[0]

        u[self._identify[0]] = u[self._identify[1]]
        return u

    def bake_spectrum(self):
        k = min(len(self._mask) - 2, 40)
        self.eigenvectors = np.zeros((np.size(self.vertices, axis=1), k), dtype=complex)
        self.eigenvalues, self.eigenvectors[self._mask, :] = sp.linalg.eigs(self.M, k, M=self.L, sigma=0.01)
        self.eigenvectors[self._identify[0], :] = self.eigenvectors[self._identify[1], :]

    def __str__(self):
        return f"FEM-{self.domain}-{self.resolution[0]}x{self.resolution[1]}"

    def fd_center(self):
        return np.average(self.vertices[0, self._mask] + 1j * self.vertices[1, self._mask])
