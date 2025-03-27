import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from Integrator import Integrator
from hyperbolic import triangulate, plot

def _dVol(x, y):
    return np.power(1 - x * x - y * y, -1.5)

class Model:
    def __init__(self, vertices, triangles, trace, int_res=100
                 , isTraceFixed=True, computeSpectrumOnBake=False):

        self.isTraceFixed = isTraceFixed

        self._trace = trace
        self._exclude = []
        self._identify = [[], []]
        self._elements = []

        self._integrator = Integrator(int_res)

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
        self._exclude = np.zeros(shape=np.size(self.vertices, axis=1), dtype=bool)

        if self.isTraceFixed:
            self._exclude[self._trace] = True

    def _bake_triangles(self):
        self.triangles = self.polygons

    def _bake_matrices(self):
        self._mask = np.logical_not(self._exclude)
        self._area = []

        elements = np.cumsum(self._mask) - 1
        n = elements[-1] + 1

        self._elements = elements
        self._elements[self._identify[0]] = self._elements[self._identify[1]]
        self._n_elements = n

        L = np.zeros([n, n])

        _dDVol = lambda x, y: np.power(1 - x * x - y * y, -.5)

        for v_i, v_j, v_k in self.polygons:
            v1 = self.vertices[:, v_k] - self.vertices[:, v_j]
            v2 = self.vertices[:, v_i] - self.vertices[:, v_k]
            v3 = self.vertices[:, v_j] - self.vertices[:, v_i]

            F1 = lambda x, y: self.vertices[0, v_i] + x * v3[0] - y * v2[0]
            F2 = lambda x, y: self.vertices[1, v_i] + x * v3[1] - y * v2[1]

            D11 = v2[1] + v3[1]; D21 = -v2[0] - v3[0]
            D12 = -v2[1]; D22 = v2[0]
            D13 = -v3[1]; D23 = v3[0]

            Jac_A = np.abs(-v3[0] * v2[1] + v3[1] * v2[0])
            I_0 = self._integrator.integrate(lambda x, y: _dDVol(F1(x, y), F2(x, y))) / Jac_A
            I11 = self._integrator.integrate(lambda x, y: F1(x, y) * F1(x, y) * _dDVol(F1(x, y), F2(x, y))) / Jac_A
            I12 = self._integrator.integrate(lambda x, y: F1(x, y) * F2(x, y) * _dDVol(F1(x, y), F2(x, y))) / Jac_A
            I22 = self._integrator.integrate(lambda x, y: F2(x, y) * F2(x, y) * _dDVol(F1(x, y), F2(x, y))) / Jac_A

            i, j, k = self._elements[v_i], self._elements[v_j], self._elements[v_k]

            if self._mask[v_i]:
                L[i, i] += np.dot(v1, v1) * I_0 - D11 * D11 * I11 - 2 * D11 * D21 * I12 - D21 * D21 * I22

                if self._mask[v_j]:
                    L12 = np.dot(v1, v2) * I_0 - D11 * D12 * I11 - (D21 * D12 + D11 * D22) * I12 - D21 * D22 * I22
                    L[i, j] += L12
                    L[j, i] += L12

                if self._mask[v_k]:
                    L13 = np.dot(v1, v3) * I_0 - D11 * D13 * I11 - (D21 * D13 + D11 * D23) * I12 - D21 * D23 * I22
                    L[i, k] += L13
                    L[k, i] += L13

            if self._mask[v_j]:
                L[j, j] += np.dot(v2, v2) * I_0 - D12 * D12 * I11 - 2 * D12 * D22 * I12 - D22 * D22 * I22

                if self._mask[v_k]:
                    L23 = np.dot(v2, v3) * I_0 - D12 * D13 * I11 - (D22 * D13 + D12 * D23) * I12 - D22 * D23 * I22
                    L[j, k] += L23
                    L[k, j] += L23

            if self._mask[v_k]:
                L[k, k] += np.dot(v3, v3) * I_0 - D13 * D13 * I11 - 2 * D13 * D23 * I12 - D23 * D23 * I22

            #print(L)

        self.L = sp.csc_matrix(L)

    def id(self):
        return "Klein k=1"

    def solve_poisson(self, f):
        b = np.zeros(dtype=complex, shape=self._n_elements)

        for v_i, v_j, v_k in self.polygons:
            v1 = self.vertices[:, v_k] - self.vertices[:, v_j]
            v2 = self.vertices[:, v_i] - self.vertices[:, v_k]
            v3 = self.vertices[:, v_j] - self.vertices[:, v_i]

            F1 = lambda x, y: self.vertices[0, v_i] + x * v3[0] - y * v2[0]
            F2 = lambda x, y: self.vertices[1, v_i] + x * v3[1] - y * v2[1]

            Jac_A = np.abs(-v3[0] * v2[1] + v3[1] * v2[0])
            _f_dv = lambda x, y: f(F1(x, y) + 1j * F2(x, y)) * _dVol(F1(x, y), F2(x, y))

            i, j, k = self._elements[v_i], self._elements[v_j], self._elements[v_k]

            if self._mask[v_i]:
                b[i] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * (1 - x - y))
            if self._mask[v_j]:
                b[j] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * x)
            if self._mask[v_k]:
                b[k] += Jac_A * self._integrator.integrate(lambda x, y: _f_dv(x, y) * y)

        self._solution = sp.linalg.spsolve(self.L, b)
        u = np.zeros(np.size(self.vertices, axis=1), dtype=complex)
        u[self._mask] = self._solution
        u[self._identify[0]] = u[self._identify[1]]
        return u

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

    def bake_spectrum(self):
        k = min(len(self._mask) - 2, 40)
        self.eigenvectors = np.zeros((np.size(self.vertices, axis=1), k), dtype=complex)
        self.eigenvalues, self.eigenvectors[self._mask, :] = sp.linalg.eigs(self.M, k, M=self.L, sigma=0.01)
        self.eigenvectors[self._identify[0], :] = self.eigenvectors[self._identify[1], :]

    def __str__(self):
        return f"FEM-{self.domain}-{self.resolution[0]}x{self.resolution[1]}"

    def fd_center(self):
        return np.average(self.vertices[0, self._mask] + 1j * self.vertices[1, self._mask])

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
        for p_i in self.polygons:
            p0 = self.vertices[:, p_i[0]]
            v1 = self.vertices[:, p_i[1]] - p0
            v2 = self.vertices[:, p_i[2]] - p0

            Jac_A = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            F1 = lambda x, y: p0[0] + x * v1[0] + y * v2[0]
            F2 = lambda x, y: p0[1] + x * v1[1] + y * v2[1]

            e = self._solution[self._elements[p_i]]
            e[np.logical_not(self._mask[p_i])] = 0

            _u = lambda x, y: u(F1(x, y) + 1j * F2(x, y))
            w = lambda x, y: _u(x, y) - e[0] * (1 - x - y) - e[1] * x - e[2] * y

            A += Jac_A * self._integrator.integrate(
                lambda x, y: w(x, y) * np.conj(w(x, y)) * _dV(F1(x, y), F2(x, y)))
            B += Jac_A * self._integrator.integrate(
                lambda x, y: _u(x, y) * np.conj(_u(x, y)) * _dV(F1(x, y), F2(x, y)))

        #print("B =", np.sqrt(np.real(B)))

        return np.sqrt(np.real(A) / np.real(B))



if __name__ == "__main__":
    vertices, polygons, trace = triangulate.load("./triangulations/uniform_disk_euc_1024.npz")
    vertices *= np.tanh(3.0)
    model = Model(vertices, polygons, trace)

    f = lambda z: 1
    u = np.real(model.solve_poisson(f))

    ax = plt.figure().add_subplot(projection="3d")
    plot.surface(ax, model.vertices, model.triangles, u)
    plot.add_wireframe(ax, model.vertices, model.triangles, u)
    plt.show()

