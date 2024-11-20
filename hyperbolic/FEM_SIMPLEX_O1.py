import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from Integrator import Integrator
from hyperbolic import triangulate, plot

def _dVol(x, y):
    return np.power(.5 * (1 - x * x - y * y), -2)

def _dVol_K(x):
    return np.power(1 - np.sum(x * x, axis=0), -1.5)

def _vec_delta(x, u):
    w = x - u[:, np.newaxis]
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * w2 / ((1 - x2) * (1 - u @ u))

def _V(u, v, x):
    d0 = 1 + 2 * ((u - v) @ (u - v)) / ((1 - u @ u) * (1 - v @ v))
    d1 = _vec_delta(x, u)
    d2 = _vec_delta(x, v)

    xu = x - u[:, np.newaxis]
    xv = x - v[:, np.newaxis]

    x2 = np.sum(x * x, axis=0)

    Dd1 = (2 * (d1 - 1) * x[0, :] / (1 - x2) + 4 * xu[0, :] / ((1 - x2) * (1 - u @ u)),
           2 * (d1 - 1) * x[1, :] / (1 - x2) + 4 * xu[1, :] / ((1 - x2) * (1 - u @ u)))

    Dd2 = (2 * (d2 - 1) * x[0, :] / (1 - x2) + 4 * xv[0, :] / ((1 - x2) * (1 - v @ v)),
           2 * (d2 - 1) * x[1, :] / (1 - x2) + 4 * xv[1, :] / ((1 - x2) * (1 - v @ v)))

    A = 1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2
    B = 1 + d0 + d1 + d2

    a1 = -d1 * Dd1[0] - d2 * Dd2[0] + d0 * d1 * Dd2[0] + d0 * Dd1[0] * d2
    a2 = -d1 * Dd1[1] - d2 * Dd2[1] + d0 * d1 * Dd2[1] + d0 * Dd1[1] * d2
    b1 = Dd1[0] + Dd2[0]
    b2 = Dd1[1] + Dd2[1]

    V = np.sqrt(A) / B
    DV = (2 * V / (1 + V * V) * (a1 / A - b1 / B),
       2 * V / (1 + V * V) * (a2 / A - b2 / B))

    vol = 2 * np.atan(V)
    M = np.max(vol)

    return vol / M, DV[0] / M, DV[1] / M

def _Phi_KtD(x):
    return x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

def _Phi_DtK(x):
    return 2 * x / (1 + np.sum(x * x, axis=0))

def _Jac_Phi_KtD(x):
    A = np.sqrt(1 - np.sum(x * x, axis=0))
    return 1 / (A * (1 + A) * (1 + A))

class Model:
    def __init__(self, vertices, triangles, trace
                 , isTraceFixed=True, computeSpectrumOnBake=False):

        self.isTraceFixed = isTraceFixed

        self._trace = trace
        self._exclude = []
        self._identify = [[], []]
        self._elements = []

        self._integrator = Integrator(100, open=True)

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
        print("Baking matrices...")

        self._mask = np.logical_not(self._exclude)
        self._area = []

        elements = np.cumsum(self._mask) - 1
        n = elements[-1] + 1

        self._elements = elements
        self._elements[self._identify[0]] = self._elements[self._identify[1]]
        self._n_elements = n

        L = np.zeros([n, n])
        for i0, i1, i2 in self.polygons:
            v_D = self.vertices[:, [i0, i1, i2]]
            v_K = _Phi_DtK(v_D)

            A = np.array([
                v_K[:, 1] - v_K[:, 0],
                v_K[:, 2] - v_K[:, 0]
            ]).T
            F = lambda x: v_K[:, 0, np.newaxis] + A @ x
            Jac_F = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])

            X = _Phi_KtD(F(self._integrator.vertices))
            V0, D1V0, D2V0 = _V(v_D[:, 0], v_D[:, 1], X)
            V1, D1V1, D2V1 = _V(v_D[:, 1], v_D[:, 2], X)
            V2, D1V2, D2V2 = _V(v_D[:, 2], v_D[:, 0], X)
            Jac_Phi = _Jac_Phi_KtD(F(self._integrator.vertices))

            e0, e1, e2 = self._elements[i0], self._elements[i1], self._elements[i2]

            if self._mask[i0]:
                L[e0, e0] += Jac_F * self._integrator.integrate_vector(
                    (D1V0 * D1V0 + D2V0 * D2V0) * Jac_Phi
                )

                if self._mask[i1]:
                    L01 = Jac_F * self._integrator.integrate_vector(
                        (D1V0 * D1V1 + D2V0 * D2V1) * Jac_Phi
                    )
                    L[e0, e1] += L01
                    L[e1, e0] += L01

                if self._mask[i2]:
                    L02 = Jac_F * self._integrator.integrate_vector(
                        (D1V0 * D1V2 + D2V0 * D2V2) * Jac_Phi
                    )
                    L[e0, e2] += L02
                    L[e2, e0] += L02

            if self._mask[i1]:
                L[e1, e1] += Jac_F * self._integrator.integrate_vector(
                    (D1V1 * D1V1 + D2V1 * D2V1) * Jac_Phi
                )

                if self._mask[i2]:
                    L12 = Jac_F * self._integrator.integrate_vector(
                        (D1V1 * D1V2 + D2V1 * D2V2) * Jac_Phi
                    )
                    L[e1, e2] += L12
                    L[e2, e1] += L12

            if self._mask[i2]:
                L[e2, e2] += Jac_F * self._integrator.integrate_vector(
                    (D1V2 * D1V2 + D2V2 * D2V2) * Jac_Phi
                )

            #print(L)

        self.L = sp.csc_matrix(L)

    def id(self):
        return "Klein k=1"

    def solve_poisson(self, f):
        b = np.zeros(dtype=complex, shape=self._n_elements)

        print("Integrating...")
        for i0, i1, i2 in self.polygons:
            v_D = self.vertices[:, [i0, i1, i2]]
            v_K = _Phi_DtK(v_D)

            A = np.array([
                v_K[:, 1] - v_K[:, 0],
                v_K[:, 2] - v_K[:, 0]
            ]).T
            F = lambda x: v_K[:, 0, np.newaxis] + A @ x
            Jac_F = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])

            X_K = F(self._integrator.vertices)
            X_D = _Phi_KtD(X_K)
            V0, D1V0, D2V0 = _V(v_D[:, 0], v_D[:, 1], X_D)
            V1, D1V1, D2V1 = _V(v_D[:, 1], v_D[:, 2], X_D)
            V2, D1V2, D2V2 = _V(v_D[:, 2], v_D[:, 0], X_D)

            _f_dv = f(X_D) * _dVol_K(X_K)

            e0, e1, e2 = self._elements[[i0, i1, i2]]

            if self._mask[i0]:
                b[e0] += Jac_F * self._integrator.integrate_vector(
                    V0 * _f_dv
                )
            if self._mask[i1]:
                b[e1] += Jac_F * self._integrator.integrate_vector(
                    V1 * _f_dv
                )
            if self._mask[i2]:
                b[e2] += Jac_F * self._integrator.integrate_vector(
                    V2 * _f_dv
                )

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

