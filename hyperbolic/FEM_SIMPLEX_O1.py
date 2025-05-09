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


def _vec_delta(x, u):
    w = x - u[:, np.newaxis]
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * w2 / ((1 - x2) * (1 - u @ u))

def _V(u, v, x):
    d0 = 1 + 2 * ((u - v) @ (u - v)) / ((1 - u @ u) * (1 - v @ v))
    d1 = _vec_delta(x, u)
    d2 = _vec_delta(x, v)

    A = np.abs(1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2)
    B = 1 + d0 + d1 + d2

    vol = 2 * np.arctan(np.sqrt(A) / B)
    return vol / np.max(vol)

def _RHS_V(Y, d0, v1, v2, b1, b2):
    Y2 = np.sum(Y * Y, axis=0)
    d1 = 1.0 + 2.0 * np.sum(v1 * v1, axis=0) / ((1.0 - Y2) * b1)
    d2 = 1.0 + 2.0 * np.sum(v2 * v2, axis=0) / ((1.0 - Y2) * b2)

    Dd1 = (2 * (d1 - 1) * Y[0, :] / (1 - Y2) + 4 * v1[0] / ((1 - Y2) * b1),
           2 * (d1 - 1) * Y[1, :] / (1 - Y2) + 4 * v1[1] / ((1 - Y2) * b1))

    Dd2 = (2 * (d2 - 1) * Y[0, :] / (1 - Y2) + 4 * v2[0] / ((1 - Y2) * b2),
           2 * (d2 - 1) * Y[1, :] / (1 - Y2) + 4 * v2[1] / ((1 - Y2) * b2))

    A = np.abs(1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2)
    B = 1 + d0 + d1 + d2

    a1 = -d1 * Dd1[0] - d2 * Dd2[0] + d0 * d1 * Dd2[0] + d0 * Dd1[0] * d2
    a2 = -d1 * Dd1[1] - d2 * Dd2[1] + d0 * d1 * Dd2[1] + d0 * Dd1[1] * d2
    b1 = Dd1[0] + Dd2[0]
    b2 = Dd1[1] + Dd2[1]

    V = np.sqrt(A) / B
    DV = (2 * V / (1 + V * V) * (a1 / A - b1 / B),
          2 * V / (1 + V * V) * (a2 / A - b2 / B))

    vol = 2 * np.arctan(V)
    M = vol[0]

    return vol / M, np.array(DV) / M

def _RHS_bdry_int(a, b, t, dt):
    u1 = a[:, 1] - a[:, 0]
    u2 = a[:, 2] - a[:, 0]

    b1 = 1.0 - b[:, 1] @ b[:, 1]
    b2 = 1.0 - b[:, 2] @ b[:, 2]

    d0 = 1.0 + 2.0 * (b[:, 2] - b[:, 1]) @ (b[:, 2] - b[:, 1]) / (b1 * b2)

    X = a[:, 0, np.newaxis] + np.outer(u1, t)
    X2 = X * X

    c = np.sqrt(1.0 - np.sum(X2, axis=0))
    Y = X / (1.0 + c)

    v1 = Y - b[:, 1, np.newaxis]
    v2 = Y - b[:, 2, np.newaxis]

    V, DV = _RHS_V(Y, d0, v1, v2, b1, b2)

    s = (1 + c) * (1 + c) * c
    g11 = 1.0 - X2[0, :]
    g12 = -X[0, :] * X[1, :]
    g22 = 1.0 - X2[1, :]

    dV = np.array([
        DV[0, :] * (1 / (1 + c) + X2[0, :] / s) - DV[1, :] * g12 / s,
        DV[1, :] * (1 / (1 + c) + X2[1, :] / s) - DV[0, :] * g12 / s,
    ])
    s_dV = np.array([
        -g12 * dV[0, :] - g22 * dV[1, :],
        g11 * dV[0, :] + g12 * dV[1, :]
    ]) / c

    I_v = V * (u1 @ s_dV)
    I = np.sum(I_v * dt) - .5 * I_v[0] * dt

    ###

    X = a[:, 0, np.newaxis] + np.outer(u2, t)
    X2 = X * X

    c = np.sqrt(1.0 - np.sum(X2, axis=0))
    Y = X / (1.0 + c)

    v1 = Y - b[:, 1, np.newaxis]
    v2 = Y - b[:, 2, np.newaxis]

    V, DV = _RHS_V(Y, d0, v1, v2, b1, b2)

    s = (1 + c) * (1 + c) * c
    g11 = 1.0 - X2[0, :]
    g12 = -X[0, :] * X[1, :]
    g22 = 1.0 - X2[1, :]

    dV = np.array([
        DV[0, :] * (1 / (1 + c) + X2[0, :] / s) - DV[1, :] * g12 / s,
        DV[1, :] * (1 / (1 + c) + X2[1, :] / s) - DV[0, :] * g12 / s,
    ])
    s_dV = np.array([
        -g12 * dV[0, :] - g22 * dV[1, :],
        g11 * dV[0, :] + g12 * dV[1, :]
    ]) / c

    I_v = V * (u2 @ s_dV)
    I -= np.sum(I_v * dt) - .5 * I_v[0] * dt

    return np.abs(I)


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
        self._int_res = int_res

        self.vertices = vertices
        self.polygons = triangles
        self.triangles = triangles
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

            N = self._int_res * (self._int_res + 1) // 4

            dt = 1.0 / N
            t = np.linspace(0, 1.0 - dt, N)

            I0 = _RHS_bdry_int(np.roll(v_K, -0, axis=1), np.roll(v_D, -0, axis=1), t, dt)
            I1 = _RHS_bdry_int(np.roll(v_K, -1, axis=1), np.roll(v_D, -1, axis=1), t, dt)
            I2 = _RHS_bdry_int(np.roll(v_K, -2, axis=1), np.roll(v_D, -2, axis=1), t, dt)

            e0, e1, e2 = self._elements[i0], self._elements[i1], self._elements[i2]

            if self._mask[i0]:
                L[e0, e0] += I0

                if self._mask[i1]:
                    L01 = .5 * (I2 - I1 - I0)
                    L[e0, e1] += L01
                    L[e1, e0] += L01

                if self._mask[i2]:
                    L02 = .5 * (I1 - I0 - I2)
                    L[e0, e2] += L02
                    L[e2, e0] += L02

            if self._mask[i1]:
                L[e1, e1] += I1

                if self._mask[i2]:
                    L12 = .5 * (I0 - I2 - I1)
                    L[e1, e2] += L12
                    L[e2, e1] += L12

            if self._mask[i2]:
                L[e2, e2] += I2

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

            V0 = _V(v_D[:, 1], v_D[:, 2], X_D)
            V1 = _V(v_D[:, 2], v_D[:, 0], X_D)
            V2 = _V(v_D[:, 0], v_D[:, 1], X_D)

            _f_dv = f(Y[0, :] + 1j * Y[1, :]) * _dVol_K(X_K)

            e0, e1, e2 = self._elements[i0], self._elements[i1], self._elements[i2]

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

            V0 = _V(v_D[:, 1], v_D[:, 2], X_D)
            V1 = _V(v_D[:, 2], v_D[:, 0], X_D)
            V2 = _V(v_D[:, 0], v_D[:, 1], X_D)

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

