import numpy as np
import matplotlib.pyplot as plt
import hyperbolic.plot as plot
from tempfile import TemporaryFile
import hyperbolic.Integrator as Integrator

def save(vertices, polygons, boundary, filename):
    np.savez("./triangulations/" + filename, vertices=vertices, polygons=polygons, boundary=boundary)

def load(filename):
    v = None; p = None; b = None
    with np.load(filename) as data:
        v = data["vertices"]
        p = data["polygons"]
        b = data["boundary"]

    return v, [list(p_i) for p_i in p], b

def _pdisk_to_bkdisk(x):
    s = 0.5 * (1 + x[0, :] * x[0, :] + x[1, :] * x[1, :])
    return x / s

def _bkdisk_to_pdisk(x):
    s = 1 + np.sqrt(1 - x[0, :] * x[0, :] - x[1, :] * x[1, :])
    return x / s

def _midpoint_bkdisk(x, y):
    _a = (x - y) @ (x - y)
    _b = x @ x - x @ y
    _c = 1 - x @ x

    _d = np.sqrt(_b * _b + _a * _c)
    t0 = (_b + _d) / _a
    t1 = (_b - _d) / _a

    A = x + t0 * (y - x)
    B = x + t1 * (y - x)

    Ax = np.sqrt((x - A) @ (x - A))
    Ay = np.sqrt((y - A) @ (y - A))
    AB = np.sqrt((B - A) @ (B - A))

    ab = np.sqrt(Ax * Ay)
    t = ab / (np.sqrt((AB - Ax) * (AB - Ay)) + ab)

    return A + t * (B - A)

def _midpoint(vertices, triangles, _int, _dVol):
    x_m = 0
    y_m = 0
    A = 0
    for i0, i1, i2 in triangles:
        p0 = vertices[:, i0]
        v1 = vertices[:, i1] - p0
        v2 = vertices[:, i2] - p0

        Jac_A = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
        F1 = lambda x, y: p0[0] + v1[0] * x + v2[0] * y
        F2 = lambda x, y: p0[1] + v1[1] * x + v2[1] * y

        x_m += Jac_A * _int.integrate(lambda x, y: F1(x, y) * _dVol(F1(x, y), F2(x, y)))
        y_m += Jac_A * _int.integrate(lambda x, y: F2(x, y) * _dVol(F1(x, y), F2(x, y)))
        A += Jac_A * _int.integrate(lambda x, y: _dVol(F1(x, y), F2(x, y)))

    return np.array([x_m, y_m]) / A

def _midpoint_pdisk(x, y):
    x_m = _midpoint_bkdisk(
            2 / (1 + x[0] * x[0] + x[1] * x[1]) * x,
            2 / (1 + y[0] * y[0] + y[1] * y[1]) * y
    )

    return x_m / (1 + np.sqrt(1 - x_m[0] * x_m[0] - x_m[1] * x_m[1]))

def _radius(p, q):
    a = np.tan(np.pi * (0.5 - 1.0 / q))
    b = np.tan(np.pi / p)
    return np.sqrt((a - b) / (a + b))

def subdivide_center(polygons, vertices):
    _polygons = []
    X = np.array(vertices)
    n = np.size(vertices, axis=0)

    for p_i in polygons:
        x_m = np.average(X[p_i], axis=0)
        vertices.append(x_m)

        for i in range(len(p_i) - 1):
            _polygons.append([p_i[i], p_i[i+1], n])
        _polygons.append([p_i[-1], p_i[0], n])

        n += 1

    return _polygons

def subdivide_triangles(polygons, vertices):
    _polygons = []
    n = len(vertices)

    for p_i in polygons:
        m1 = _midpoint_pdisk(vertices[p_i[0]], vertices[p_i[1]])
        m2 = _midpoint_pdisk(vertices[p_i[1]], vertices[p_i[2]])
        m3 = _midpoint_pdisk(vertices[p_i[0]], vertices[p_i[2]])

        i1, i2, i3 = -1, -1, -1
        for l in range(len(vertices)):
            if ((m1 - vertices[l]) @ (m1 - vertices[l]) < 1e-10):
                i1 = l
            if ((m2 - vertices[l]) @ (m2 - vertices[l]) < 1e-10):
                i2 = l
            if ((m3 - vertices[l]) @ (m3 - vertices[l]) < 1e-10):
                i3 = l

        if i1 < 0:
            vertices.append(m1)
            i1 = n
            n += 1
        if i2 < 0:
            vertices.append(m2)
            i2 = n
            n += 1
        if i3 < 0:
            vertices.append(m3)
            i3 = n
            n += 1

        _polygons.extend([
            [p_i[0], i1, i3],
            [p_i[1], i2, i1],
            [p_i[2], i3, i2],
            [i1, i2, i3]
        ])

    return _polygons

def subdivide_square(polygons, vertices):
    _polygons = []
    n = len(vertices)

    _int = Integrator.Integrator(100)
    _dVol = lambda x, y: np.power(.5 * (1 - x*x - y*y), -2)

    for p_i in polygons:
        M = [
            _midpoint_pdisk(vertices[p_i[0]], vertices[p_i[1]]),
            _midpoint_pdisk(vertices[p_i[1]], vertices[p_i[2]]),
            _midpoint_pdisk(vertices[p_i[2]], vertices[p_i[3]]),
            _midpoint_pdisk(vertices[p_i[3]], vertices[p_i[0]])
        ]

        I = [-1, -1, -1, -1]
        for l in range(len(vertices)):
            for i in range(4):
                if (M[i] - vertices[l]) @ (M[i] - vertices[l]) < 1e-10:
                    I[i] = l

        for i in range(4):
            if I[i] < 0:
                vertices.append(M[i])
                I[i] = n
                n += 1

        c = _midpoint(
            vertices=np.array([vertices[i] for i in p_i]).T,
            triangles=[[0, 1, 2], [0, 2, 3]],
            _int=_int, _dVol=_dVol
        )
        _polygons.extend([
            [p_i[0], I[0], n, I[3]],
            [p_i[1], I[1], n, I[0]],
            [p_i[2], I[2], n, I[1]],
            [p_i[3], I[3], n, I[2]]
        ])

        vertices.append(c)
        n += 1

    return _polygons

def trim_vertices(vertices, polygons, boundary, mask):
    boundary = np.array(boundary)
    vertices_map = np.cumsum(mask) - 1
    _boundary = set(vertices_map[boundary[mask[boundary]]])

    for i in range(len(polygons)):
        p_i = polygons[i]
        if not np.all(mask[p_i]):
            _boundary.update(vertices_map[[j for j in p_i if mask[j]]])

    vertices = vertices[:, mask]
    polygons = [p_i for p_i in polygons if np.all(mask[p_i])]
    polygons = vertices_map[polygons].tolist()

    return vertices, polygons, list(_boundary)

# must satisfy (p - 2) * (q - 2) > 4
def generate(p, q, iterations, subdivisions, model="Poincare", minimal=False, sieve=lambda x: False):
    r = _radius(p, q)
    angle = 2 * np.pi / p

    _sieve = None
    if model == "Poincare":
        _sieve = sieve

    if model == "Klein":
        _sieve = lambda x: sieve(2 * x / (1 + x @ x))

    X = [np.array([r * np.cos((k + 0.5) * angle), r * np.sin((k + 0.5) * angle)]) for k in range(p)]
    Q = [q for _ in range(p)]

    polygons = [[i for i in range(p)]]

    print("Subdividing...")
    if p == 3:
        for _ in range(subdivisions):
            polygons = subdivide_triangles(polygons, X)
        Q.extend([6 for _ in range(len(X) - p)])
    elif p == 4:
        for _ in range(subdivisions):
            polygons = subdivide_square(polygons, X)
        Q.extend([4 for _ in range(len(X) - p)])

    flags = [[False for i in range(p)] for _ in range(len(polygons))]

    l0 = 0.5 / np.cos(0.5 * angle) * (r + 1 / r)
    r0 = l0 * l0 - 1

    def _invert(x, m, r):
        u = x - m
        return m + (r / (u @ u)) * u

    print("Building triangulation...")
    #polygons = [set(p_i) for p_i in polygons]
    for _ in range(iterations):
        N = len(polygons)

        print("Iteration:", _ + 1, end='')
        for k in range(p): # iterate over every angle
            m = np.array([l0 * np.cos(k * angle), l0 * np.sin(k * angle)])

            if N > 1000:
                print(" .", end='')

            for i in range(0, N):
                if flags[i][k]:
                    continue

                p_i = list(polygons[i])
                _inv = [_invert(X[j], m, r0) for j in p_i]
                if all([_sieve(x) for x in _inv]):
                    flags[i][k] = True
                    continue

                poly = []
                for j in range(len(p_i)):
                    for l in range(len(X)):
                        if ((_inv[j] - X[l]) @ (_inv[j] - X[l]) < 1e-10):
                            poly.append(l)
                            break
                    else:
                        poly.append(len(X))
                        X.append(_inv[j] )
                        Q.append(Q[p_i[j]])

                for _poly in polygons:
                    if set(_poly) == set(poly):
                        break
                else:
                    polygons.append(poly)
                    flags.append([_i == k for _i in range(p)])

                flags[i][k] = True

        print()

    pre_trace = np.zeros(shape=(len(X)), dtype=int)
    for i in range(len(polygons)):
        pre_trace[polygons[i]] += 1

    trace_mask = pre_trace < Q
    if np.any(pre_trace > q):
        print("Overlapping triangulation!")

    X = np.array(X).T
    trace = np.arange(np.size(X, axis=1))[trace_mask]

    if minimal:
        print("Removing unnecessary triangles...")
        # remove triangles where all vertices touch the boundary
        polygons_mask = np.logical_not(np.all(trace_mask[polygons], axis=1))
        polygons = [polygons[i] for i in range(len(polygons)) if polygons_mask[i]]

        vertices_mask = np.zeros(shape=(np.size(X, axis=1)), dtype=bool)
        for p_i in polygons:
            vertices_mask[p_i] = True

        X, polygons, trace = trim_vertices(X, polygons, trace, vertices_mask)

    else:
        trace = trace.tolist()

    if model == "Klein":
        return _pdisk_to_bkdisk(X), polygons, trace

    else:
        return X, polygons, trace

def retract(vertices, polygons, trace, dVol):
    trace_mask = np.zeros(shape=np.size(vertices, axis=1), dtype=bool)
    trace_mask[trace] = True
    vert_nbh = [[] for _ in range(np.size(vertices, axis=1))]

    for p_i in polygons:
        for i in p_i:
            if trace_mask[i]:
                vert_nbh[i].append(p_i)

    _int = Integrator.Integrator(100)
    for i in trace:
        vertices[:, i] = _midpoint(vertices, vert_nbh[i], _int, dVol)

    return vertices, polygons, trace

def _fix_area(vertices, polygons, trace, threshold = 1e-2):
    print("Checking area...")
    integrator = Integrator.Integrator(100)

    vertices_area = np.zeros(shape=np.size(vertices, axis=1), dtype=float)

    _min = 1e10
    _max = 0
    _avg = 0
    _dVol = lambda x, y: np.power(.5 * (1 - x*x - y*y), -2)
    for I in range(len(polygons)):
        p_i = polygons[I]

        p0 = vertices[:, p_i[0]]
        u = vertices[:, p_i[1]] - p0
        v = vertices[:, p_i[2]] - p0

        F1 = lambda x, y: p0[0] + x * u[0] + y * v[0]
        F2 = lambda x, y: p0[1] + x * u[1] + y * v[1]

        area = np.abs(u[0] * v[1] - u[1] * v[0]) * \
               integrator.integrate(lambda x, y: _dVol(F1(x, y), F2(x, y)))

        vertices_area[p_i] = np.maximum(vertices_area[p_i], area)

        _min = min(area, _min)
        _max = max(area, _max)
        _avg += area

    print("min:", _min, "; max:", _max, "; avg:", _avg / len(polygons))

    vertices_mask = vertices_area > threshold * _max
    if np.all(vertices_mask):
        return vertices, polygons, trace
    else:
        return trim_vertices(vertices, polygons, trace, vertices_mask)

def plot_triangulation(filename):
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(xlim=(-1.01, 1.01), ylim=(-1.05, 1.05))
    plt.axis("off")

    t = np.linspace(0, 2 * np.pi, 60)
    plt.plot(np.cos(t), np.sin(t), "--", color="gray", linewidth=.8)

    vertices, polygons, trace = load("./triangulations/" + filename + ".npz")
    plot.add_wireframe(ax, vertices, polygons)

    d = np.max(np.sum(vertices * vertices, axis=0))
    d = np.sqrt(d)
    print(d / (1 + np.sqrt(1 - d*d)))

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig("./figures/" + filename + ".png", dpi=300, bbox_inches=extent, transparent=True)

if __name__ == "__main__":
    plot_triangulation("uniform_disk_hyp_256")
    plot_triangulation("uniform_disk_hyp_512")
    plot_triangulation("uniform_disk_hyp_1024")
    plot_triangulation("uniform_disk_hyp_2048")
    plot_triangulation("uniform_disk_hyp_4096")
    plot_triangulation("uniform_disk_hyp_8192")
    #plt.show()
    '''

    ax = plt.figure().add_subplot()

    vertices, polygons, trace = generate(p=3, q=7, iterations=7, subdivisions=2, model="Poincare", minimal=True)
    t = np.linspace(0, 2 * np.pi, 60)
    plt.plot(np.cos(t), np.sin(t), "--", color="gray", linewidth=.8)
    plot.add_wireframe(ax, vertices, polygons)
    #plt.scatter(vertices[0, trace], vertices[1, trace], s=1)
    plt.axis("equal")
    plt.show()
    '''
    '''
    ax = plt.figure().add_subplot()
    r = .95

    r_k = 2 * r / (1 + r*r)
    vertices, polygons, trace = rect(0.04)
    #vertices *= r_k / np.sqrt(2)

    #vertices = _pdisk_to_bkdisk(vertices)
    vertices, polygons, trace = force_disk(vertices, polygons, trace, r_k)
    #vertices = _bkdisk_to_pdisk(vertices)
#
    #vertices, polygons, trace = _fix_area(vertices, polygons, trace, 1e-2)

    plot.add_wireframe(ax, vertices, polygons)
    plt.scatter(vertices[0, trace], vertices[1, trace], s=3)

    save(vertices, polygons, trace, "rect04__klein__d95")

    plt.axis("equal")
    plt.show()
    '''
'''

def force_disk(vertices, polygons, trace, r, s, debug=False):
    R = np.tanh(.5 * r)
    R_l = np.tanh(.5 * (r - s))
    R_u = np.tanh(.5 * (r + s))

    if debug:
        t = np.linspace(0, 2 * np.pi)
        plt.plot(R * np.cos(t), R * np.sin(t), linewidth=0.5, color="black")
        plt.plot(R_u * np.cos(t), R_u * np.sin(t), linewidth=0.5, color="gray")
        plt.plot(R_l * np.cos(t), R_l * np.sin(t), linewidth=0.5, color="gray")

    else:
        print("Making disk...")
        vertices, polygons, trace = trim_vertices(vertices, polygons, trace,
                              np.sum(vertices * vertices, axis=0) < R_u * R_u)

        vert_norm = np.sqrt(np.sum(vertices * vertices, axis=0))
        shift_mask = vert_norm > R_l
        vertices[:, shift_mask] *= R / vert_norm[shift_mask]

        trace_mask = np.zeros(shape=np.size(vertices, axis=1), dtype=bool)
        trace_mask[trace] = True

        polygons_boundary = [p_i for p_i in polygons if np.any(trace_mask)]
        trace_nbh = [set() for _ in range(np.size(vertices, axis=1))]
        for I in range(len(polygons_boundary)):
            p_i = polygons_boundary[I]
            shifted = [i for i in p_i if shift_mask[i]]
            if len(shifted) == 0:
                continue

            for i in p_i:
                if shift_mask[i]:
                    continue

                d = 1e10
                edge_i = None
                for j in shifted:
                    trace_nbh[i].discard(j)
                    _d = (vertices[:, i] - vertices[:, j]) @ (vertices[:, i] - vertices[:, j])
                    if _d < d:
                        edge_i = j
                        d = _d

                if not -edge_i - 1 in trace_nbh[i]:
                    trace_nbh[i].add(edge_i)

                trace_nbh[i].update([-j - 1 for j in shifted])

        for i in range(len(trace_nbh)):
            if not trace_mask[i]:
                continue

            p_i = [j for j in trace_nbh[i] if j > 0]
            if len(p_i) >= 2:
                plt.scatter(vertices[0, i], vertices[1, i])
                trace.remove(i)
                polygons.append([i, p_i[0], p_i[1]])

        for i in trace:
            vertices[:, i] *= R / np.sqrt(vertices[:, i] @ vertices[:, i])

    return vertices, polygons, trace

def connect_inactives(vertices, polygons, trace):
    trace_mask = np.zeros(shape=np.size(vertices, axis=1), dtype=bool)
    trace_mask[trace] = True

    polygons_boundary = [p_i for p_i in polygons if np.any(trace_mask[p_i])]
    polygons_inactive = [p_i for p_i in polygons_boundary if np.all(trace_mask[p_i])]
    p_nbh = [[None, None, None] for _ in polygons_inactive]

    for I in range(len(polygons_inactive)):
        p_i = polygons_inactive[I]
        for J in range(len(polygons_boundary)):
            p_j = polygons_boundary[J]
            if p_i == p_j:
                continue

            _J = -1
            if p_j in polygons_inactive:
                _J = polygons_inactive.index(p_j)

            if p_i[0] in p_j:
                p_nbh[I][0] = _J
            if p_i[1] in p_j:
                p_nbh[I][1] = _J
            if p_i[2] in p_j:
                p_nbh[I][2] = _J

    for I in range(len(polygons_inactive)):
        p_i = polygons_inactive[I]

        print(p_nbh[I])
        edge_i = p_nbh[I].index(None)
        i1 = (edge_i + 1) % 3
        i2 = (edge_i + 2) % 3

        J = p_nbh[I][i1]
        K = p_nbh[I][i2]

        if I < J:
            p_j = polygons_inactive[J]
            edge_j = p_nbh[J].index(None)
            polygons.append([p_i[edge_i], p_j[edge_j], p_i[i1]])
            trace.remove(p_i[i1])

        if I < K:
            p_k = polygons_inactive[K]
            edge_k = p_nbh[K].index(None)
            polygons.append([p_i[edge_i], p_k[edge_k], p_i[i2]])
            trace.remove(p_i[i2])

    return vertices, polygons, trace
'''
