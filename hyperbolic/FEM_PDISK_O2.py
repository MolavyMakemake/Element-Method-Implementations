import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from Integrator import Integrator
from hyperbolic import triangulate, plot

def _dVol(x, y):
    return np.power(.5 * (1 - x * x - y * y), -2)

class Model:
    def __init__(self, vertices, triangles, trace
                 , isTraceFixed=True, computeSpectrumOnBake=False):

        self.isTraceFixed = isTraceFixed

        self._trace = trace
        self._exclude = []
        self._identify = [[], []]
        self._elements = []

        self._integrator = Integrator(100)

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

    def id(self):
        return "Poincaré k=2"

    def bake(self):
        self._bake_domain()
        self._bake_triangles()
        self._bake_matrices()

        if self.computeSpectrumOnBake:
            self.bake_spectrum()

    def _bake_domain(self):
        self._identify = [[], []]
        self._exclude = np.zeros(shape=np.size(self.vertices, axis=1), dtype=bool)

        if self.isTraceFixed:
            self._exclude[self._trace] = True

    def _bake_triangles(self):
        self.triangles = self.polygons

    def _bake_matrices(self):
        self._area = []

        def _are_neighbors(p_i, p_j):
            return len(set(p_i) & set(p_j)) > 1

        #polygon_neighbors = [[] for _ in range(len(polygons))]
        #for i in range(len(self.polygons)):
        #    for j in range(i+1, len(self.polygons)):
        #        if _are_neighbors(self.polygons[i], self.polygons[j]):
        #            polygon_neighbors[i].append(j)
        #            polygon_neighbors[j].append(i)

        print("Indexing elements...")
        self._mask = np.zeros(shape=(len(self.polygons), 6), dtype=bool)
        elements = np.zeros(shape=(len(self.polygons), 6), dtype=int)

        self._vertices_mask = np.logical_not(self._exclude)
        _vertex_map = np.cumsum(self._vertices_mask) - 1

        n = _vertex_map[-1] + 1
        _index_map = np.array([[None, 3, 5], [3, None, 4], [5, 4, None]])
        for i in range(len(self.polygons)):
            p_i = self.polygons[i]
            self._mask[i, [0, 1, 2]] = self._vertices_mask[p_i]
            self._mask[i, 3] = self._mask[i, 0] or self._mask[i, 1]
            self._mask[i, 4] = self._mask[i, 1] or self._mask[i, 2]
            self._mask[i, 5] = self._mask[i, 2] or self._mask[i, 0]

            elements[i, [0, 1, 2]] = _vertex_map[p_i]
            _e3 = -1
            _e4 = -1
            _e5 = -1

            for j in range(i):
                p_j = self.polygons[j]
                _b0 = p_i[0] in p_j
                _b1 = p_i[1] in p_j
                _b2 = p_i[2] in p_j

                if _b0 and _b1:
                    _e3 = elements[j, _index_map[
                        p_j.index(p_i[0]), p_j.index(p_i[1])]]

                if _b1 and _b2:
                    _e4 = elements[j, _index_map[
                        p_j.index(p_i[1]), p_j.index(p_i[2])]]

                if _b0 and _b2:
                    _e5 = elements[j, _index_map[
                        p_j.index(p_i[0]), p_j.index(p_i[2])]]

            if _e3 < 0 and self._mask[i, 3]:
                _e3 = n
                n += 1
            if _e4 < 0 and self._mask[i, 4]:
                _e4 = n
                n += 1
            if _e5 < 0 and self._mask[i, 5]:
                _e5 = n
                n += 1

            elements[i, [3, 4, 5]] = [_e3, _e4, _e5]

        self._elements = elements
        self._elements[self._identify[0]] = self._elements[self._identify[1]]
        self._n_elements = n

        print("Baking matrices...")
        L = np.zeros([n, n])
        for i in range(len(self.polygons)):
            v_i, v_j, v_k = self.polygons[i]
            v0 = self.vertices[:, v_k] - self.vertices[:, v_j]
            v1 = self.vertices[:, v_i] - self.vertices[:, v_k]
            v2 = self.vertices[:, v_j] - self.vertices[:, v_i]

            Jac_A = np.abs(-v2[0] * v1[1] + v2[1] * v1[0])
            e = self._elements[i, :]

            if self._mask[i, 0]:
                L[e[0], e[0]] += 1.0 / (2 * Jac_A) * np.dot(v0, v0)

                if self._mask[i, 1]:
                    L01 = -1.0 / (6 * Jac_A) * np.dot(v0, v1)
                    L[e[0], e[1]] += L01
                    L[e[1], e[0]] += L01

                if self._mask[i, 2]:
                    L02 = -1.0 / (6 * Jac_A) * np.dot(v0, v2)
                    L[e[0], e[2]] += L02
                    L[e[2], e[0]] += L02

                if self._mask[i, 3]:
                    L03 = 2.0 / (3 * Jac_A) * np.dot(v0, v1)
                    L[e[0], e[3]] += L03
                    L[e[3], e[0]] += L03

                if self._mask[i, 4]:
                    pass

                if self._mask[i, 5]:
                    L05 = 2.0 / (3 * Jac_A) * np.dot(v0, v2)
                    L[e[0], e[5]] += L05
                    L[e[5], e[0]] += L05

            if self._mask[i, 1]:
                L[e[1], e[1]] += 1.0 / (2 * Jac_A) * np.dot(v1, v1)

                if self._mask[i, 2]:
                    L12 = -1.0 / (6 * Jac_A) * np.dot(v1, v2)
                    L[e[1], e[2]] += L12
                    L[e[2], e[1]] += L12

                if self._mask[i, 3]:
                    L13 = 2.0 / (3 * Jac_A) * np.dot(v1, v0)
                    L[e[1], e[3]] += L13
                    L[e[3], e[1]] += L13

                if self._mask[i, 4]:
                    L14 = 2.0 / (3 * Jac_A) * np.dot(v1, v2)
                    L[e[1], e[4]] += L14
                    L[e[4], e[1]] += L14

                if self._mask[i, 5]:
                    pass

            if self._mask[i, 2]:
                L[e[2], e[2]] += 1.0 / (2 * Jac_A) * np.dot(v2, v2)

                if self._mask[i, 3]:
                    pass

                if self._mask[i, 4]:
                    L24 = 2.0 / (3 * Jac_A) * np.dot(v2, v1)
                    L[e[2], e[4]] += L24
                    L[e[4], e[2]] += L24

                if self._mask[i, 5]:
                    L25 = 2.0 / (3 * Jac_A) * np.dot(v2, v0)
                    L[e[2], e[5]] += L25
                    L[e[5], e[2]] += L25

            if self._mask[i, 3]:
                L[e[3], e[3]] += 4.0 / (3 * Jac_A) * (np.dot(v0, v0) + np.dot(v0, v1) + np.dot(v1, v1))

                if self._mask[i, 4]:
                    L34 = 2.0 / (3 * Jac_A) * (np.dot(v1, v2) + np.dot(v1, v1) + 2 * np.dot(v0, v2) + np.dot(v0, v1))
                    L[e[3], e[4]] += L34
                    L[e[4], e[3]] += L34

                if self._mask[i, 5]:
                    L35 = 2.0 / (3 * Jac_A) * (np.dot(v1, v0) + 2 * np.dot(v1, v2) + np.dot(v0, v0) + np.dot(v0, v2))
                    L[e[3], e[5]] += L35
                    L[e[5], e[3]] += L35

            if self._mask[i, 4]:
                L[e[4], e[4]] += 4.0 / (3 * Jac_A) * (np.dot(v1, v1) + np.dot(v1, v2) + np.dot(v2, v2))

                if self._mask[i, 5]:
                    L45 = 2.0 / (3 * Jac_A) * (np.dot(v2, v0) + np.dot(v2, v2) + 2 * np.dot(v1, v0) + np.dot(v1, v2))
                    L[e[4], e[5]] += L45
                    L[e[5], e[4]] += L45

            if self._mask[i, 5]:
                L[e[5], e[5]] += 4.0 / (3 * Jac_A) * (np.dot(v0, v0) + np.dot(v0, v2) + np.dot(v2, v2))

        self.L = sp.csc_matrix(L)

    def solve_poisson(self, f):
        self._solution = np.zeros(shape=self._n_elements, dtype=complex)
        b = np.zeros(dtype=complex, shape=(self._n_elements))

        print("Integrating...")
        for i in range(len(self.polygons)):
            v_i, v_j, v_k = self.polygons[i]

            v1 = self.vertices[:, v_k] - self.vertices[:, v_j]
            v2 = self.vertices[:, v_i] - self.vertices[:, v_k]
            v3 = self.vertices[:, v_j] - self.vertices[:, v_i]

            F1 = lambda x, y: self.vertices[0, v_i] + x * v3[0] - y * v2[0]
            F2 = lambda x, y: self.vertices[1, v_i] + x * v3[1] - y * v2[1]

            Jac_A = np.abs(-v3[0] * v2[1] + v3[1] * v2[0])
            _f_dv = lambda x, y: f(F1(x, y) + 1j * F2(x, y)) * _dVol(F1(x, y), F2(x, y))

            e = self._elements[i, :]

            if self._mask[i, 0]:
                b[e[0]] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * (1 - 2*x - 2*y) * (1 - x - y))
            if self._mask[i, 1]:
                b[e[1]] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * x * (2*x - 1))
            if self._mask[i, 2]:
                b[e[2]] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * y * (2*y - 1))
            if self._mask[i, 3]:
                b[e[3]] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * 4*x * (1 - x - y))
            if self._mask[i, 4]:
                b[e[4]] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * 4 * x * y)
            if self._mask[i, 5]:
                b[e[5]] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * 4*y * (1 - x - y))


        print("Solving the linear system...")
        self._solution = sp.linalg.spsolve(self.L, b)
        u = np.zeros(shape=np.size(self.vertices, axis=1), dtype=complex)
        u[self._vertices_mask] = self._solution[:np.sum(self._vertices_mask)]
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

    def area(self):
        A = 0
        for v_i, v_j, v_k in self.polygons:
            v2 = self.vertices[:, v_i] - self.vertices[:, v_k]
            v3 = self.vertices[:, v_j] - self.vertices[:, v_i]
            A += np.abs(-v3[0] * v2[1] + v3[1] * v2[0]) * \
                self._integrator.integrate(
                     lambda x, y: _dVol(
                         self.vertices[0, v_i] + x * v3[0] - y * v2[0],
                         self.vertices[1, v_i] + x * v3[1] - y * v2[1])
             )

        return A

    def compare(self, u, norm):
        _dV = None
        if norm == "L2":
            _dV = lambda x, y: 1
        elif norm == "L2_g":
            _dV = _dVol
        else:
            print("Does not support norm", norm)
            return 0

        A = 0
        B = 0
        for I in range(len(self.polygons)):
            p0 = self.vertices[:, self.polygons[I][0]]
            v1 = self.vertices[:, self.polygons[I][1]] - p0
            v2 = self.vertices[:, self.polygons[I][2]] - p0

            Jac_A = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            F1 = lambda x, y: p0[0] + x * v1[0] + y * v2[0]
            F2 = lambda x, y: p0[1] + x * v1[1] + y * v2[1]

            e = self._solution[self._elements[I, :]]
            e[np.logical_not(self._mask[I, :])] = 0

            _u = lambda x, y: u(F1(x, y) + 1j * F2(x, y))
            w = lambda x, y: _u(x, y) \
                        - e[0] * (1 - 2*x - 2*y) * (1 - x - y) \
                        - e[1] * x * (2*x - 1) \
                        - e[2] * y * (2*y - 1) \
                        - e[3] * 4*x * (1 - x - y) \
                        - e[4] * 4 * x * y \
                        - e[5] * 4*y * (1 - x - y)

            A += Jac_A * self._integrator.integrate(
                lambda x, y: w(x, y) * np.conj(w(x, y)) * _dV(F1(x, y), F2(x, y)))
            B += Jac_A * self._integrator.integrate(
                lambda x, y: _u(x, y) * np.conj(_u(x, y)) * _dV(F1(x, y), F2(x, y)))


        return np.sqrt(np.real(A) / np.real(B))

if __name__ == "__main__":
    vertices, polygons, trace = triangulate.generate(p=3, q=7, iterations=3, subdivisions=2, model="Poincare", minimal=True)
    model = Model(vertices, polygons, trace)

    f = lambda z: 1
    u = np.real(model.solve_poisson(f))

    ax = plt.figure().add_subplot(projection="3d")
    plot.surface(ax, model.vertices, model.triangles, u)
    plot.add_wireframe(ax, model.vertices, model.triangles, u)
    plt.show()

