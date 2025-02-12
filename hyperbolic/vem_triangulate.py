import numpy as np
import matplotlib.pyplot as plt
from hyperbolic.triangulate import load
import hyperbolic.plot as plot

R = np.tanh(1.5)

def vem_mesh(N_v):
    vertices, triangles, boundary = load(f"./triangulations/uniform_disk_hyp_{N_v}.npz")
    _triangles = []

    rot = np.array([[0, -1], [1, 0]])
    for i0, i1, i2 in triangles:
        x0 = vertices[:, i0]
        x1 = vertices[:, i1]
        x2 = vertices[:, i2]

        p = []
        if np.dot(rot @ (x1 - x0), x2 - x0) > 0:
            p = [i0, i1, i2]
        else:
            p = [i0, i2, i1]

        while p[0] in boundary:
            p = [p[1], p[2], p[0]]

        _triangles.append(p)

    vertices_hng = []
    n_hng = 20
    N_hng = 0
    polygons = []
    for I in _triangles:
        p = [I[0], I[1]]
        if I[1] in boundary and I[2] in boundary:
            x0 = vertices[:, I[1]]
            x1 = vertices[:, I[2]]
            t0 = np.arctan2(x0[1], x0[0])
            t1 = np.arctan2(x1[1], x1[0])

            d = (t1 - t0) % (2 * np.pi)

            for i in range(1, n_hng + 1):
                t = t0 + d * i / (n_hng + 1)
                vertices_hng.append([R * np.cos(t), R * np.sin(t)])

            p.extend(range(N_v + N_hng, N_v + N_hng + n_hng))
            N_hng += n_hng

        p.append(I[2])
        polygons.append(p)

    vertices = np.concatenate((vertices, np.array(vertices_hng).T), axis=1)
    boundary = np.concatenate((boundary, np.arange(N_v, N_v + N_hng, 1)))

    return vertices, polygons, boundary

if __name__ == "__main__":
    v, p, b = vem_mesh(256)

    print(p)

    ax = plt.figure().add_subplot(projection="3d")
    plot.add_wireframe(ax, v, p, np.zeros_like(v))
    plt.scatter(v[0, :], v[1, :], s=.4)
    plt.show()