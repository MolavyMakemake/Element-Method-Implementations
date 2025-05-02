import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from Integrator import Integrator
from hyperbolic import triangulate, plot
import vem_triangulate

def _dVol(x, y):
    return np.power(1 - x * x - y * y, -1.5)

def _dVol_K(x):
    return np.power(1 - np.sum(x * x, axis=0), -1.5)

def klein_to_poincare(x):
    return x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

def klein_to_hyperboloid(x):
    return np.concatenate((np.ones_like(x[0, :])[np.newaxis, :], x)) \
        / np.sqrt(1 - np.sum(x*x, axis=0))

def _vec_delta(x, u):
    w = x - u[:, np.newaxis]
    x2 = np.sum(x * x, axis=0)
    w2 = np.sum(w * w, axis=0)
    return 1 + 2 * w2 / ((1 - x2) * (1 - u @ u))

def nrm_dst(a, x):
    b = a / (1 + np.sqrt(1 - np.sum(a * a, axis=0)))
    y = x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

    det = (x[0, :] - a[0, 0]) * (a[1, 2] - a[1, 1]) \
            - (x[1, :] - a[1, 0]) * (a[0, 2] - a[0, 1])
    t = ((a[1, 2] - a[1, 1]) * (a[0, 1] - a[0, 0])
         - (a[0, 2] - a[0, 1]) * (a[1, 1] - a[1, 0])) / det

    x_A = a[:, 0, np.newaxis] + t * (x - a[:, 0, np.newaxis])

    d0 = _vec_delta(y, b[:, 0])
    d1 = _vec_delta(x_A / (1 + np.sqrt(1 - np.sum(x_A * x_A, axis=0))), b[:, 0])

    f = 1 - np.arccosh(d0) / np.arccosh(d1)
    f[np.isnan(f)] = 1
    return f

def compute_hyperbolic_area(v):
    N_v = np.size(v, axis=1)

    I = (N_v - 2) * np.pi
    for i1 in range(N_v):
        i0 = i1 - 1
        i2 = (i1 + 1) % N_v

        u1 = v[:, i2] - v[:, i1]
        u2 = v[:, i0] - v[:, i1]

        r = v[:, i1] * v[:, i1]
        g = np.array([
            [1 - r[1], v[0, i1] * v[1, i1]],
            [v[0, i1] * v[1, i1], 1 - r[0]]
        ]) / np.square(1 - r[0] - r[1])

        I -= np.arccos(np.dot(u1, g @ u2) / np.sqrt(
            np.dot(u1, g @ u1) * np.dot(u2, g @ u2)
        ))

    return I

def compute_area(v, t, dt):
    dv = np.roll(v, -1, axis=1) - v

    I = 0
    for i in range(np.size(v, axis=1)):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)
        r = X * X
        c = np.sqrt(1.0 - np.sum(r, axis=0))

        Y = X / (1 + c)
        dy = np.array([
            Y[0, :] * Y[1, :] / c,
            1 / (1 + c) + Y[1, :] * Y[1, :] / c
        ])

        I_v = dv[:, i] @ dy * Y[0, :]
        I += np.sum(I_v * dt) - .5 * (I_v[0] + I_v[-1]) * dt

    return I

def V1(x):
    r = x * x
    t = np.sqrt(1 - r[0, :] - r[1, :])
    s = np.sqrt(t * (1 + t))

    return x[0, :] / s, np.array([
        t * (1 + t) + r[0, :] * (.5 / t + 1),
        x[0, :] * x[1, :] * (.5 / t + 1)]) / (s * s * s)

def V2(x):
    r = x * x
    t = np.sqrt(1 - r[0, :] - r[1, :])
    s = np.sqrt(t * (1 + t))

    return x[1, :] / s, np.array([
        x[0, :] * x[1, :] * (.5 / t + 1),
        t * (1 + t) + r[1, :] * (.5 / t + 1)]) / (s * s * s)

def dst(x0, x1):
    y0 = x0 / (1 + np.sqrt(1 - np.sum(x0 * x0, axis=0)))
    y1 = x1 / (1 + np.sqrt(1 - np.sum(x1 * x1, axis=0)))

    delta = 1 + 2 * np.sum((y0 - y1) * (y0 - y1), axis=0) \
            / ((1 - np.sum(y0 * y0, axis=0)) * (1 - np.sum(y1 * y1, axis=0)))

    return np.arccosh(delta)

def W(x):
    r = x * x
    t = np.sqrt(1 - r[0, :] - r[1, :])

    return 2 / (t * t * (1 + t)) * x

def star(v, x):
    r = x * x
    g11 = 1 - r[0, :]
    g12 = -x[0, :] * x[1, :]
    g22 = 1 - r[1, :]
    return np.array([
        -g12 * v[0, :] - g22 * v[1, :],
        g11 * v[0, :] + g12 * v[1, :]
    ]) / np.sqrt(1 - r[0, :] - r[1, :])

def int_v(f, ds):
    return np.sum(ds * (f[:-1] + f[1:])) / 2

def normalize(v, x):
    r = x * x
    g11 = 1 - r[0, :]
    g12 = x[0, :] * x[1, :]
    g22 = 1 - r[1, :]

    v2 = (g11 * v[0] + g12 * v[1]) * v[0] + (g12 * v[0] + g22 * v[1]) * v[1]
    return (1 - r[0, :] - r[1, :]) / np.sqrt(v2) * v[:, np.newaxis]

def Int_PP(v, t, dt):
    dv = np.roll(v, -1, axis=1) - v

    area = compute_hyperbolic_area(v)

    I11 = area / 2
    I12 = 0
    I22 = area / 2
    for i in range(np.size(v, axis=1)):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)

        p1, dp1 = V1(X)
        p2, dp2 = V2(X)

        sW = star(W(X), X)

        c = np.sqrt(1.0 - np.sum(X * X, axis=0))
        Y = X / (1 + c)
        r = Y * Y

        dx = np.array([1 / (1 + c) + r[0, :] / c, Y[0, :] * Y[1, :] / c])
        dy = np.array([Y[0, :] * Y[1, :] / c, 1 / (1 + c) + r[1, :] / c])

        I11_v = (1 / 16) * dv[:, i] @ sW * p1 * p1
        I12_v = (1 / 16) * dv[:, i] @ sW * p1 * p2
        I22_v = (1 / 16) * dv[:, i] @ sW * p2 * p2

        s = 1.0 / np.sqrt(1 - r)
        I11_v += .5 * dv[:, i] @ dx * (X[1, :] / c - 2 * s[0, :] * np.arctanh(s[0, :] * Y[1, :]))
        I12_v += .5 * dv[:, i] @ dy * X[1, :] / c
        I22_v -= .5 * dv[:, i] @ dy * (X[0, :] / c - 2 * s[1, :] * np.arctanh(s[1, :] * Y[0, :]))

        I11 += np.sum(I11_v * dt) - .5 * (I11_v[0] + I11_v[-1]) * dt
        I12 += np.sum(I12_v * dt) - .5 * (I12_v[0] + I12_v[-1]) * dt
        I22 += np.sum(I22_v * dt) - .5 * (I22_v[0] + I22_v[-1]) * dt

    return np.array([[I11, I12], [I12, I22]])

def Int_WP(v, t):
    dv = np.roll(v, -1, axis=1) - v
    h = dst(v, np.roll(v, -1, axis=1))

    N_v = np.size(v, axis=1)
    I = np.zeros(shape=(2, N_v), dtype=float)
    for i in range(N_v):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)

        ds = dst(X[:, :-1], X[:, 1:])

        p1, dp1 = V1(X)
        p2, dp2 = V2(X)

        I0 = np.sum(ds * (p1[:-1] + p1[1:])) / 2
        I1 = np.sum(ds * (p2[:-1] + p2[1:])) / 2

        j = (i + 1) % N_v

        I[0, i] += I0 * h[i-1]
        I[1, i] += I1 * h[i-1]

        I[0, j] -= I0 * h[j]
        I[1, j] -= I1 * h[j]

    return I[:, :-1]

def Int_VW(v):
    N_v = np.size(v, axis=1)
    h = dst(v, np.roll(v, -1, axis=1))

    I = np.zeros(shape=(N_v - 1, N_v), dtype=float)
    i = np.arange(0, N_v - 1)

    I[i[1:], i[:-1]] = -h[i[:-1]] * h[i[1:]] / 2
    I[i, i+1] = h[i] * h[i-1] / 2
    I[0, -1] = -h[0] * h[-1] / 2
    return I

def Stabilizer(v, t, proj_P):
    dv = np.roll(v, -1, axis=1) - v
    h = dst(v, np.roll(v, -1, axis=1))

    I00 = 0
    I01 = 0
    I11 = 0

    N_v = np.size(v, axis=1)
    I = np.zeros(shape=(N_v, N_v), dtype=float)
    for i in range(N_v):
        X = v[:, i, np.newaxis] + np.outer(dv[:, i], t)

        ds = dst(X[:, :-1], X[:, 1:])

        p1, dp1 = V1(X)
        p2, dp2 = V2(X)

        j = (i + 1) % N_v

        I[i, i] += h[i] * h[i - 1] * (h[i] + h[i - 1])
        I[i, j] += -h[i - 1] * h[i] * h[j]
        I[j, i] += -h[i - 1] * h[i] * h[j]

        dT = normalize(dv[:, i], X)

        s0 = np.sum(star(dp1, X) * dT, axis=0)
        s1 = np.sum(star(dp2, X) * dT, axis=0)

        I0 = int_v(s0, ds)
        I1 = int_v(s1, ds)

        I00 += int_v(s0 * s0, ds)
        I01 += int_v(s0 * s1, ds)
        I11 += int_v(s1 * s1, ds)

        for k in range(N_v - 1):
            _I_ik = -h[i - 1] * (proj_P[0, k] * I0 + proj_P[1, k] * I1)
            _I_jk = h[j] * (proj_P[0, k] * I0 + proj_P[1, k] * I1)

            I[i, k] += _I_ik
            I[k, i] += _I_ik
            I[j, k] += _I_jk
            I[k, j] += _I_jk

    for i in range(N_v - 1):
        I[i, i] += proj_P[0, i] * proj_P[0, i] * I00 \
                   + 2 * proj_P[0, i] * proj_P[1, i] * I01 \
                   + proj_P[1, i] * proj_P[1, i] * I11
        for j in range(i + 1, N_v - 1):
            _I_ij = proj_P[0, i] * proj_P[0, j] * I00 \
                    + proj_P[0, i] * proj_P[1, j] * I01 \
                    + proj_P[1, i] * proj_P[0, j] * I01 \
                    + proj_P[1, i] * proj_P[1, j] * I11

            I[i, j] += _I_ij
            I[j, i] += _I_ij

    return I[:-1, :-1] * np.max(h)

def proj_RHS(v, t, dt):
    I_wp = Int_WP(v, t)
    I_pp = Int_PP(v, t, dt)
    I_vw = Int_VW(v)

    proj_P = np.linalg.inv(I_pp) @ I_wp

    return proj_P @ np.linalg.solve(proj_P.T @ I_wp + Stabilizer(v, t, proj_P), I_vw), I_pp
    #return np.linalg.lstsq(proj_P.T @ I_wp, I_vw)[0], I_vw

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

def _V(u, v, x):
    d0 = 1 + 2 * ((u - v) @ (u - v)) / ((1 - u @ u) * (1 - v @ v))
    d1 = _vec_delta(x, u)
    d2 = _vec_delta(x, v)

    A = np.abs(1 - d0 * d0 - d1 * d1 - d2 * d2 + 2 * d0 * d1 * d2)
    B = 1 + d0 + d1 + d2

    vol = 2 * np.arctan(np.sqrt(A) / B)
    return vol / np.max(vol)

def _Phi_KtD(x):
    return x / (1 + np.sqrt(1 - np.sum(x * x, axis=0)))

def _Phi_DtK(x):
    return 2 * x / (1 + np.sum(x * x, axis=0))

def _Jac_Phi_KtD(x):
    A = np.sqrt(1 - np.sum(x * x, axis=0))
    return 1 / (A * (1 + A) * (1 + A))

class Model:
    def __init__(self, vertices, polygons, trace, int_res=100
                 , isTraceFixed=True, computeSpectrumOnBake=False):

        self.isTraceFixed = isTraceFixed

        self._trace = trace
        self._exclude = []
        self._identify = [[], []]
        self._elements = []

        self._integrator = Integrator(int_res, open=True)
        self.int_res = int_res

        self.vertices = vertices
        self.polygons = polygons
        self.L = np.array([[]])
        self.M = np.array([[]])
        self._mask = []

        self.eigenvectors = []
        self.eigenvalues = []
        self.computeSpectrumOnBake = computeSpectrumOnBake

        rot = np.array([[0, -1], [1, 0]])
        for i in range(len(polygons)):
            p_i = polygons[i]
            v = vertices[:, p_i]

            if np.dot(rot @ (v[:, 1] - v[:, 0]), v[:, 2] - v[:, 0]) < 0:
                polygons[i] = [p_i[0], p_i[2], p_i[1]]

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
        triangles = []
        for I in self.polygons:
            for i in range(len(I) - 2):
                triangles.append([I[0], I[i + 1], I[i + 2]])

        self.triangles = triangles

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
        for I in self.polygons:
            N_v = len(I)

            v = self.vertices[:, I]
            w = klein_to_hyperboloid(v)
            w = translate(barycenter(v)) @ w
            v = hyperboloid_to_klein(w)

            N = self.int_res ** 2
            t = np.linspace(0, 1, N+1)
            dt = 1.0 / N

            proj_P, I_pp = proj_RHS(v, t, dt)

            for i in range(N_v):
                if not self._mask[I[i]]:
                    continue

                e_i = self._elements[I[i]]
                L[e_i, e_i] += np.dot(proj_P[:, i], I_pp @ proj_P[:, i])

                for j in range(i + 1, N_v):
                    if not self._mask[I[j]]:
                        continue

                    e_j = self._elements[I[j]]
                    Lij = np.dot(proj_P[:, i], I_pp @ proj_P[:, j])
                    L[e_i, e_j] += Lij
                    L[e_j, e_i] += Lij

        self.L = sp.csc_matrix(L)

    def id(self):
        return "Klein k=1"

    def solve_poisson(self, f):
        b = np.zeros(dtype=complex, shape=self._n_elements)

        print("Integrating...")
        for I in self.polygons:
            N_v = len(I)

            v = self.vertices[:, I]
            area = compute_hyperbolic_area(v)

            for i in I:
                if not self._mask[i]:
                    continue

                b[self._elements[i]] += area / N_v

        self._solution = sp.linalg.spsolve(self.L, b)
        u = np.zeros(np.size(self.vertices, axis=1), dtype=complex)
        u[self._mask] = self._solution
        u[self._identify[0]] = u[self._identify[1]]
        return u

    def area(self):
        A = 0
        for I in self.polygons:
            A += compute_hyperbolic_area(self.vertices[:, I])

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
        for i0, i1, i2 in self.triangles:
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

            V0 = nrm_dst(np.roll(v, -0, axis=1), X)
            V1 = nrm_dst(np.roll(v, -1, axis=1), X)
            V2 = nrm_dst(np.roll(v, -2, axis=1), X)

            e = self._solution[self._elements[[i0, i1, i2]]]
            e[np.logical_not(self._mask[[i0, i1, i2]])] = 0

            U = u(Y[0, :] + 1j * Y[1, :])
            W = U - (e[0] * V0 + e[1] * V1 + e[2] * V2) / (V0 + V1 + V2)
            dv = _dV(X)

            norm_error += Jac_F * self._integrator.integrate_vector(
                W * np.conj(W) * dv)
            norm_soln += Jac_F * self._integrator.integrate_vector(
                U * np.conj(U) * dv)

        return np.sqrt(np.real(norm_error) / np.real(norm_soln))



if __name__ == "__main__":
    vertices, polygons, trace = vem_triangulate.vem_mesh(512)
    vertices = _Phi_DtK(vertices)
    model = Model(vertices, polygons, trace)

    f = lambda z: 1
    u = np.real(model.solve_poisson(f))

    ax = plt.figure().add_subplot(projection="3d")
    plot.surface(ax, model.vertices, model.triangles, u)
    plot.add_wireframe(ax, model.vertices, model.triangles, u)
    plt.show()

