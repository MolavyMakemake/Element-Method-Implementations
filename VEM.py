import numpy as np

import orbifolds
import mesh

class Model:
    def __init__(self, domain="rectangle", bounds=np.array([[-1, 1], [-1, 1]]),
                 resolution=[21, 21], isTraceFixed=True, computeSpectrumOnBake=False):
        self.bounds = bounds
        self.resolution = resolution
        self.domain = domain
        self.isTraceFixed = isTraceFixed

        self._exclude = []
        self._identify = [[], []]
        self._area = []
        self._elements = []

        self.vertices = []
        self.polygons = []
        self.triangles = []
        self.L = np.array([[]])
        self.M = np.array([[]])
        self.I = np.array([[]])
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
            self.vertices, self.polygons, trace = orbifolds.mesh(self.domain, W, H, 4)
            self._identify = orbifolds.compute_idmap(self.domain, W, H)
            self._exclude.extend(self._identify[0])

            trace = [self._identify[1][0]]

        else:
            self.vertices, self.polygons, trace = mesh.generic(self.domain, W, H, 4)

        if self.isTraceFixed:
            self._exclude.extend(trace)


    def _bake_triangles(self):
        self.triangles = []
        for p_i in self.polygons:
            for i in range(2, len(p_i)):
                self.triangles.append([p_i[0], p_i[i - 1], p_i[i]])

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
        self.I = np.zeros([n, len(self.polygons)])

        rot = np.array([[0, 1], [-1, 0]])

        for p_i, p_I in enumerate(self.polygons):
            p1 = self.vertices[:, p_I]
            p0 = np.roll(p1, 1, axis=1)
            p2 = np.roll(p1, -1, axis=1)

            N = np.size(p1, axis=1)

            d0 = np.linalg.norm(p1 - p0, axis=0)
            d1 = np.roll(d0, -1)

            area = 0.5 * np.abs(np.dot(p1[0, :], p0[1, :]) - np.dot(p0[0, :], p1[1, :]))  # shoelace
            self._area.append(area)
            circumference = np.sum(d0)

            proj_D = rot @ (p2 - p0) / (2 * area)
            proj_a = (d0 + d1 - ((p1 + p0) @ d0) @ proj_D) / (2 * circumference)

            for i in range(N):
                e_i = self._elements[p_I[i]]
                if e_i < 0:
                    continue

                d_ik = i == np.arange(0, N)
                for j in range(N):
                    e_j = self._elements[p_I[j]]
                    if e_j < 0:
                        continue

                    # compute local LHS
                    d_jk = j == np.arange(0, N)
                    a = np.dot(proj_D[:, i], proj_D[:, j]) * area
                    s = np.sum((d_ik - proj_a[i] - proj_D[:, i] @ p1) * (d_jk - proj_a[j] - proj_D[:, j] @ p1))

                    self.L[e_i, e_j] += a + s

                self.I[e_i, p_i] += 1 / N

        self.M = (self.I * self._area) @ np.transpose(self.I)

    def solve_poisson(self, f):
        u = np.zeros(np.size(self.vertices, axis=1), dtype=complex)

        I_f = np.zeros(len(self.polygons), dtype=complex)
        for p_i, p_I in enumerate(self.polygons):
            p = self.vertices[:, p_I]
            I_f[p_i] = np.average(f(p[0, :] + 1j * p[1, :])) * self._area[p_i]
        b = self.I @ I_f
        #p = self.vertices[:, self._mask]
        #b = self.M @ f(p[0, :] + 1j * p[1, :])

        u[self._mask] = np.linalg.solve(self.L, b)

        u[self._identify[0]] = u[self._identify[1]]
        return u

    def bake_spectrum(self):
        u = np.zeros((len(self._mask), np.size(self.vertices, axis=1)), dtype=complex)

        self.eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(self.L) @ self.M)

        u[:, self._mask] = eigenvectors
        self.eigenvectors = u / np.reshape(np.sqrt(np.real(
            np.sum((self.M @ eigenvectors) * np.conj(eigenvectors), axis=1))), [len(self.eigenvalues), 1])
