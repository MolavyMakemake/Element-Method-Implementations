import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from Integrator import Integrator
from hyperbolic import triangulate, plot

def _dVol(x, y):
    return np.power(1 - x * x - y * y, -1.5)

def _dVol_K(x):
    return np.power(1 - np.sum(x * x, axis=0), -1.5)

def klein_to_poincare(x):
    return x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

def klein_to_hyperboloid(x):
    return np.concatenate((np.ones_like(x[0, :])[np.newaxis, :], x)) \
        / np.sqrt(1 - np.sum(x*x, axis=0))

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

def hyperboloid_to_klein(x):
    return x[1:, :] / x[0, :]

def translate(a):
    return np.array([
        [a[0], -a[1], -a[2]],
        [-a[1], a[1] * a[1] / (a[0] + 1) + 1, a[1] * a[2] / (a[0] + 1)],
        [-a[2], a[1] * a[2] / (a[0] + 1), a[2] * a[2] / (a[0] + 1) + 1]
    ])

def rotate(a):
    s = 1.0 / np.sqrt(a[1] * a[1] + a[2] * a[2])
    return np.array([
        [1, 0, 0],
        [0, a[2] * s, -a[1] * s],
        [0, a[1] * s, a[2] * s]
    ])

def shift(a):
    s = 1.0 / np.sqrt(1 + a[1] * a[1])
    return np.array([
        [a[0] * s, 0, -a[2] * s],
        [0, 1, 0],
        [-a[2] * s, 0, a[0] * s]
    ])

def orthomap(a):
    A = translate(a[:, 1])
    A = rotate(A @ a[:, 2]) @ A
    return shift(A @ a[:, 0]) @ A

def Jac_klein_to_hyperboloid(x, u):
    r = x * x
    return np.power(1 - r[0, :] - r[1, :], -1.5) * np.array([
        x[0, :] * u[0, :] + (1 - r[1, :]) * u[1, :] + x[0, :] * x[1, :] * u[2, :],
        x[1, :] * u[0, :] + (1 - r[0, :]) * u[2, :] + x[0, :] * x[1, :] * u[1, :],
    ])

def Jac_hyperboloid_to_klein(x, u):
    return np.array([
        -x[1, :] / x[0, :] * u[0, :] - x[2, :] / x[0, :] * u[1, :],
        u[0, :], u[1, :]]) / x[0, :]

def _V(x, a):
    z = klein_to_hyperboloid(x)
    b = klein_to_hyperboloid(a)

    A = orthomap(b)
    z = A @ z
    y = hyperboloid_to_klein(z)

    r = y * y
    S = np.power((1 - r[1, :]) / ((1 - r[0, :]) * (1 - r[0, :] - r[1, :])), .25)

    f = y[0, :] * S
    m = np.min(f)
    M = np.max(f)
    M = M if M > -m else m

    #D1f = -.5 / M * r[0, :] * r[1, :] / (1 - r[0, :]) + 1 - r[1, :]
    #D2f = .5 / M * y[0, :] * y[1, :] * r[0, :] / (1 - r[1, :])
    #Dgf = S * np.array([
    #    (1 - r[0, :]) * D1f - y[0, :] * y[1, :] * D2f,
    #    (1 - r[1, :]) * D2f - y[0, :] * y[1, :] * D1f
    #])

    Dgf = S * np.array([
        1 + .5 * r[0, :] * (1.0 / (1 - r[0, :] - r[1, :]) + 1.0 / (1 - r[0, :])),
        .5 * y[0, :] * y[1, :] * (1 / (1 - r[0, :] - r[1, :]) - 1 / (1 - r[1, :]))
    ])

    Dgf = Jac_hyperboloid_to_klein(z, Dgf)
    Dgf = Jac_klein_to_hyperboloid(x, A.T @ Dgf)

    return f / M, Dgf / M

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

            v = self.vertices[:, [i0, i1, i2]]
            w = klein_to_hyperboloid(v)
            w = translate(barycenter(v)) @ w
            v = hyperboloid_to_klein(w)

            A = np.array([
                v[:, 1] - v[:, 0],
                v[:, 2] - v[:, 0]
            ]).T
            Jac_F = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])

            X = v[:, 0, np.newaxis] + A @ self._integrator.vertices

            V0, DV0 = _V(X, v)
            V1, DV1 = _V(X, np.roll(v, -1, axis=1))
            V2, DV2 = _V(X, np.roll(v, -2, axis=1))

            dv = np.power(1 - np.sum(X * X, axis=0), -.5)

            G11 = (1 - X[0, :] * X[0, :])
            G22 = (1 - X[1, :] * X[1, :])
            G12 = -X[0, :] * X[1, :]
            g = lambda u: np.array([G11 * u[0, :] + G12 * u[1, :], G12 * u[0, :] + G22 * u[1, :]])

            e0, e1, e2 = self._elements[i0], self._elements[i1], self._elements[i2]

            if self._mask[i0]:
                L[e0, e0] += Jac_F * self._integrator.integrate_vector(
                    np.sum(g(DV0) * DV0, axis=0) * dv
                )

                if self._mask[i1]:
                    L01 = Jac_F * self._integrator.integrate_vector(
                        np.sum(g(DV0) * DV1, axis=0) * dv
                    )
                    L[e0, e1] += L01
                    L[e1, e0] += L01

                if self._mask[i2]:
                    L02 = Jac_F * self._integrator.integrate_vector(
                        np.sum(g(DV0) * DV2, axis=0) * dv
                    )
                    L[e0, e2] += L02
                    L[e2, e0] += L02

            if self._mask[i1]:
                L[e1, e1] += Jac_F * self._integrator.integrate_vector(
                    np.sum(g(DV1) * DV1, axis=0) * dv
                )

                if self._mask[i2]:
                    L12 = Jac_F * self._integrator.integrate_vector(
                        np.sum(g(DV1) * DV2, axis=0) * dv
                    )
                    L[e1, e2] += L12
                    L[e2, e1] += L12

            if self._mask[i2]:
                L[e2, e2] += Jac_F * self._integrator.integrate_vector(
                    np.sum(g(DV2) * DV2, axis=0) * dv
                )

            #print(L)

        self.L = sp.csc_matrix(L)

    def id(self):
        return "Klein k=1"

    def solve_poisson(self, f):
        b = np.zeros(dtype=complex, shape=self._n_elements)

        print("Integrating...")
        for i0, i1, i2 in self.polygons:
            v = self.vertices[:, [i0, i1, i2]]

            v = self.vertices[:, [i0, i1, i2]]
            w = klein_to_hyperboloid(v)
            T = translate(barycenter(v))
            v = hyperboloid_to_klein(T @ w)

            A = np.array([
                v[:, 1] - v[:, 0],
                v[:, 2] - v[:, 0]
            ]).T
            F = lambda x: v[:, 0, np.newaxis] + A @ x
            Jac_F = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])

            X = F(self._integrator.vertices)
            Y = klein_to_hyperboloid(X)
            Y = hyperboloid_to_klein(np.linalg.inv(T) @ Y)

            V0, DV0 = _V(X, v)
            V1, DV1 = _V(X, np.roll(v, -1, axis=1))
            V2, DV2 = _V(X, np.roll(v, -2, axis=1))

            _f_dv = f(Y[0, :] + 1j * Y[1, :]) * _dVol_K(X)

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

        norm_error = 0
        norm_soln = 0
        for i0, i1, i2 in self.polygons:
            v = self.vertices[:, [i0, i1, i2]]
            w = klein_to_hyperboloid(v)
            T = translate(barycenter(v))
            v = hyperboloid_to_klein(T @ w)

            A = np.array([
                v[:, 1] - v[:, 0],
                v[:, 2] - v[:, 0]
            ]).T
            F = lambda x: v[:, 0, np.newaxis] + A @ x
            Jac_F = np.abs(A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1])

            X = F(self._integrator.vertices)
            Y = klein_to_hyperboloid(X)
            Y = hyperboloid_to_klein(np.linalg.inv(T) @ Y)

            V0, DV0 = _V(X, v)
            V1, DV1 = _V(X, np.roll(v, -1, axis=1))
            V2, DV2 = _V(X, np.roll(v, -2, axis=1))

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

