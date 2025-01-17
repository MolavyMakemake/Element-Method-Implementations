import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from Integrator import Integrator
from hyperbolic import triangulate, plot

def _dVol(x, y):
    return np.power(1 - x * x - y * y, -1.5)

def _dVol_K(x):
    return np.power(1 - np.sum(x * x, axis=0), -1.5)

_BC_INTEGRATOR = Integrator(20)
def barycenter(v):
    X = v[:, 0, np.newaxis] + np.array([
        v[:, 1] - v[:, 0],
        v[:, 2] - v[:, 0]
    ]).T @ _BC_INTEGRATOR.vertices

    dv = _dVol_K(X)
    bc = np.array([
        _BC_INTEGRATOR.integrate_vector(X[0, :] * dv),
        _BC_INTEGRATOR.integrate_vector(X[1, :] * dv)
    ]) / _BC_INTEGRATOR.integrate_vector(dv)

    t = 1 / np.sqrt(1 - bc @ bc)
    return np.array([t, t * bc[0], t * bc[1]])

def translate(a):
    return np.array([
        [a[0], -a[1], -a[2]],
        [-a[1], a[1] * a[1] / (a[0] + 1) + 1, a[1] * a[2] / (a[0] + 1)],
        [-a[2], a[1] * a[2] / (a[0] + 1), a[2] * a[2] / (a[0] + 1) + 1]
    ])

def klein_to_hyperboloid(x):
    t = 1 / np.sqrt(1 - np.sum(x*x, axis=0))
    return np.concatenate((t[np.newaxis, :], t * x))

def hyperboloid_to_klein(x):
    return x[1:, :] / x[0, :]


def _distance(x, y, A):
    u0 = x - A[0]
    u1 = y - A[1]

    a = 1 - x * x - y * y
    b = u0 * u0 + u1 * u1
    d = 1.0 + 2.0 * b / (a * (1 - A @ A))

    dst = np.arccosh(d)

    s = 4.0 / (a * (1 - A @ A) * np.sqrt(d * d - 1))
    D1dst = s * (u0 + x * b / a)
    D2dst = s * (u1 + y * b / a)

    return dst, D1dst, D2dst

def _V(u, v, x):
    a, D1a, D2a = _distance(x[0, :], x[1, :], u)
    b, D1b, D2b = _distance(x[0, :], x[1, :], v)
    c, _, _ = _distance(u[0], u[1], v)
    s = (a + b + c) / 2.0

    S = np.sqrt(np.sinh(s) * np.sinh(s - a) * np.sinh(s - b) * np.sinh(s - c))
    D1S = 0.5 / S * (
            np.cosh(s) * np.sinh(s - a) * np.sinh(s - b) * np.sinh(s - c) * 0.5 * ( D1a + D1b) +
            np.sinh(s) * np.cosh(s - a) * np.sinh(s - b) * np.sinh(s - c) * 0.5 * (-D1a + D1b) +
            np.sinh(s) * np.sinh(s - a) * np.cosh(s - b) * np.sinh(s - c) * 0.5 * ( D1a - D1b) +
            np.sinh(s) * np.sinh(s - a) * np.sinh(s - b) * np.cosh(s - c) * 0.5 * ( D1a + D1b)
    )
    D2S = 0.5 / S * (
            np.cosh(s) * np.sinh(s - a) * np.sinh(s - b) * np.sinh(s - c) * 0.5 * ( D2a + D2b) +
            np.sinh(s) * np.cosh(s - a) * np.sinh(s - b) * np.sinh(s - c) * 0.5 * (-D2a + D2b) +
            np.sinh(s) * np.sinh(s - a) * np.cosh(s - b) * np.sinh(s - c) * 0.5 * ( D2a - D2b) +
            np.sinh(s) * np.sinh(s - a) * np.sinh(s - b) * np.cosh(s - c) * 0.5 * ( D2a + D2b)
    )

    M = np.max(S)

    return S / M, np.array([D1S, D2S]) / M

def _Phi_KtD(x):
    return x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

def _Phi_DtK(x):
    return 2 * x / (1 + np.sum(x * x, axis=0))

def _Jac_Phi_KtD(x):
    A = np.sqrt(1 - np.sum(x * x, axis=0))
    return 1 / (A * (1 + A) * (1 + A))

class Model:
    def __init__(self, vertices, triangles, trace, int_res=100
                 , isTraceFixed=True, computeSpectrumOnBake=False):

        self.isTraceFixed = isTraceFixed

        self._trace = trace
        self._exclude = []
        self._identify = [[], []]
        self._elements = []

        self._integrator = Integrator(int_res, open=True)

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
            v_K = self.vertices[:, [i0, i1, i2]]
            w = klein_to_hyperboloid(v_K)
            w = translate(barycenter(v_K)) @ w
            v_K = hyperboloid_to_klein(w)

            v_D = _Phi_KtD(v_K)

            A = np.array([
                v_K[:, 1] - v_K[:, 0],
                v_K[:, 2] - v_K[:, 0]
            ]).T
            F = lambda x: v_K[:, 0, np.newaxis] + A @ x
            Jac_F = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])

            X_K = F(self._integrator.vertices)
            X_D = _Phi_KtD(X_K)

            V0, DV0 = _V(v_D[:, 0], v_D[:, 1], X_D)
            V1, DV1 = _V(v_D[:, 1], v_D[:, 2], X_D)
            V2, DV2 = _V(v_D[:, 2], v_D[:, 0], X_D)
            dv = np.square(1 - np.sum(X_D * X_D, axis=0)) * _dVol_K(X_K) / 4.0
            e0, e1, e2 = self._elements[i0], self._elements[i1], self._elements[i2]

            if self._mask[i0]:
                L[e0, e0] += Jac_F * self._integrator.integrate_vector(
                    np.sum(DV0 * DV0, axis=0) * dv
                )

                if self._mask[i1]:
                    L01 = Jac_F * self._integrator.integrate_vector(
                        np.sum(DV0 * DV1, axis=0) * dv
                    )
                    L[e0, e1] += L01
                    L[e1, e0] += L01

                if self._mask[i2]:
                    L02 = Jac_F * self._integrator.integrate_vector(
                        np.sum(DV0 * DV2, axis=0) * dv
                    )
                    L[e0, e2] += L02
                    L[e2, e0] += L02

            if self._mask[i1]:
                L[e1, e1] += Jac_F * self._integrator.integrate_vector(
                    np.sum(DV1 * DV1, axis=0) * dv
                )

                if self._mask[i2]:
                    L12 = Jac_F * self._integrator.integrate_vector(
                        np.sum(DV1 * DV2, axis=0) * dv
                    )
                    L[e1, e2] += L12
                    L[e2, e1] += L12

            if self._mask[i2]:
                L[e2, e2] += Jac_F * self._integrator.integrate_vector(
                    np.sum(DV2 * DV2, axis=0) * dv
                )

            # print(L)

        self.L = sp.csc_matrix(L)

    def id(self):
        return "Klein k=1"

    def solve_poisson(self, f):
        b = np.zeros(dtype=complex, shape=self._n_elements)

        print("Integrating...")
        for i0, i1, i2 in self.polygons:
            v_K = self.vertices[:, [i0, i1, i2]]
            w = klein_to_hyperboloid(v_K)
            T = translate(barycenter(v_K))
            v_K = hyperboloid_to_klein(T @ w)

            v_D = _Phi_KtD(v_K)

            A = np.array([
                v_K[:, 1] - v_K[:, 0],
                v_K[:, 2] - v_K[:, 0]
            ]).T
            F = lambda x: v_K[:, 0, np.newaxis] + A @ x
            Jac_F = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])

            X_K = F(self._integrator.vertices)
            X_D = _Phi_KtD(X_K)

            Y = klein_to_hyperboloid(X_K)
            Y = hyperboloid_to_klein(np.linalg.inv(T) @ Y)

            V0, DV0 = _V(v_D[:, 0], v_D[:, 1], X_D)
            V1, DV1 = _V(v_D[:, 1], v_D[:, 2], X_D)
            V2, DV2 = _V(v_D[:, 2], v_D[:, 0], X_D)

            _f_dv = f(Y[0, :] + 1j * Y[1, :]) * _dVol_K(X_K)

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
            _dV = lambda x: 1
        elif norm == "L2_g":
            _dV = _dVol_K
        else:
            print("Does not support norm", norm)
            return 0

        norm_error = 0.0
        norm_soln = 0.0
        for i0, i1, i2 in self.polygons:
            v = self.vertices[:, [i0, i1, i2]]
            w = klein_to_hyperboloid(v)
            T = translate(barycenter(v))
            v = hyperboloid_to_klein(T @ w)

            v_D = _Phi_KtD(v)

            A = np.array([
                v[:, 1] - v[:, 0],
                v[:, 2] - v[:, 0]
            ]).T
            F = lambda x: v[:, 0, np.newaxis] + A @ x
            Jac_F = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])

            X = F(self._integrator.vertices)
            Y = klein_to_hyperboloid(X)
            Y = hyperboloid_to_klein(np.linalg.inv(T) @ Y)

            X_D = _Phi_KtD(X)

            V0, DV0 = _V(v_D[:, 0], v_D[:, 1], X_D)
            V1, DV1 = _V(v_D[:, 1], v_D[:, 2], X_D)
            V2, DV2 = _V(v_D[:, 2], v_D[:, 0], X_D)

            e = self._solution[self._elements[[i0, i1, i2]]]
            e[np.logical_not(self._mask[[i0, i1, i2]])] = 0

            U = u(Y[0, :] + 1j * Y[1, :])
            W = U - e[0] * V0 - e[1] * V1 - e[2] * V2
            dv = _dV(X)

            norm_error += Jac_F * self._integrator.integrate_vector(
                W * np.conj(W) * dv)
            norm_soln += Jac_F * self._integrator.integrate_vector(
                U * np.conj(U) * dv)

        return np.sqrt(np.real(norm_error) / np.real(norm_soln))



if __name__ == "__main__":
    vertices, polygons, trace = triangulate.generate(p=3, q=7, iterations=3, subdivisions=2, model="Klein", minimal=True)
    model = Model(vertices, polygons, trace)

    f = lambda z: 1
    u = np.real(model.solve_poisson(f))

    ax = plt.figure().add_subplot(projection="3d")
    plot.surface(ax, model.vertices, model.triangles, u)
    plot.add_wireframe(ax, model.vertices, model.triangles, u)
    plt.show()

