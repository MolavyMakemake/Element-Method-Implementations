import numpy as np
import matplotlib.pyplot as plt
import plot

def _pdisk_to_bkdisk(x):
    s = 0.5 * (1 + x[0, :] * x[0, :] + x[1, :] * x[1, :])
    return x / s

def _bkdisk_to_pdisk(x):
    s = 1 + np.sqrt(1 - x[0, :] * x[0, :] - x[1, :] * x[1, :])
    return x / s

def _midpoint_bkdisk(x, y):
    return .5 * (x + y)

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

def trim_vertices(vertices, polygons, boundary, mask):
    vertices_map = np.cumsum(mask) - 1

    vertices = vertices[:, mask]
    polygons = [p_i for p_i in polygons if np.all(mask[p_i])]
    boundary = boundary[mask[boundary]]

    polygons = vertices_map[polygons].tolist()
    boundary = vertices_map[boundary]

    return vertices, polygons, boundary

# must satisfy (p - 2) * (q - 2) > 4
def generate(p, q, iterations, subdivisions, model="Poincare", minimal=False):
    r = _radius(p, q)

    angle = 2 * np.pi / p

    X = [np.array([r * np.cos((k + 0.5) * angle), r * np.sin((k + 0.5) * angle)]) for k in range(p)]
    Q = [q for _ in range(p)]

    polygons = [[i for i in range(p)]]

    print("Subdividing...")
    for _ in range(subdivisions):
        polygons = subdivide_triangles(polygons, X)

    Q.extend([6 for _ in range(len(X) - p)])

    flags = [[False for i in range(p)] for _ in range(len(polygons))]

    l0 = 0.5 / np.cos(0.5 * angle) * (r + 1 / r)
    r0 = l0 * l0 - 1

    def _invert(x, m, r):
        u = x - m
        return m + (r / (u @ u)) * u

    print("Building triangulation...")
    polygons = [set(p_i) for p_i in polygons]
    for _ in range(iterations):
        N = len(polygons)

        print("Iteration:", _ + 1)
        for k in range(p): # iterate over every angle
            m = np.array([l0 * np.cos(k * angle), l0 * np.sin(k * angle)])

            for i in range(0, N):
                if flags[i][k]:
                    continue

                poly = set()
                for v_i in polygons[i]:
                    x_inv = _invert(X[v_i], m, r0)

                    for l in range(len(X)):
                        if ((x_inv - X[l]) @ (x_inv - X[l]) < 1e-10):
                            poly.add(l)
                            break

                    else:
                        poly.add(len(X))
                        X.append(x_inv)
                        Q.append(Q[v_i])

                for _poly in polygons:
                    if _poly == poly:
                        break
                else:
                    polygons.append(poly)
                    flags.append([_i == k for _i in range(p)])

                flags[i][k] = True

    pre_trace = np.zeros(shape=(len(X)), dtype=int)
    for i in range(len(polygons)):
        polygons[i] = list(polygons[i])
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

    if model == "Klein":
        return _pdisk_to_bkdisk(X), polygons, trace

    else:
        return X, polygons, trace

if __name__ == "__main__":
    vertices, polygons, trace = generate(3, 7, iterations=1, subdivisions=1, model="Poincare", minimal=True)

    ax = plt.figure().add_subplot()
    plot.add_wireframe(ax, vertices, polygons)
    plt.scatter(vertices[0, trace], vertices[1, trace], s=3)
    plt.axis("equal")
    plt.show()
