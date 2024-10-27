import numpy as np
import matplotlib.pyplot as plt
import plot

def _pdisk_to_bkdisk(x):
    s = 0.5 * (1 + x[0, :] * x[0, :] + x[1, :] * x[1, :])
    return x / s

def _radius(p, q):
    a = np.tan(np.pi * (0.5 - 1.0 / q))
    b = np.tan(np.pi / p)
    return np.sqrt((a - b) / (a + b))

# must satisfy (p - 2) * (q - 2) > 4
def generate(p, q, iterations, model="Poincare"):
    r = _radius(p, q)

    angle = 2 * np.pi / p

    X = [np.array([r * np.cos((k + 0.5) * angle), r * np.sin((k + 0.5) * angle)]) for k in range(p)]
    polygons = [set([i for i in range(p)])]
    flags = [[False for i in range(p)]]

    l0 = 0.5 / np.cos(0.5 * angle) * (r + 1 / r)
    r0 = l0 * l0 - 1

    def _invert(x, m, r):
        u = x - m
        return m + (r / (u @ u)) * u

    for _ in range(iterations):
        N = len(polygons)

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

                for _poly in polygons:
                    if _poly == poly:
                        break
                else:
                    polygons.append(poly)
                    flags.append([_i == k for _i in range(p)])

                flags[i][k] = True

    X = np.array(X).T

    pre_trace = np.zeros(shape=(np.size(X, axis=1)),dtype=int)
    for i in range(len(polygons)):
        polygons[i] = list(polygons[i])
        pre_trace[polygons[i]] += 1

    if np.any(pre_trace > q):
        print("Overlapping triangulation!")

    print(np.arange(len(pre_trace))[pre_trace > q])

    trace = np.arange(np.size(X, axis=1))[pre_trace < q]

    if model == "Klein":
        return _pdisk_to_bkdisk(X), polygons, trace

    else:
        return X, polygons, trace

if __name__ == "__main__":
    vertices, polygons, trace = generate(3, 7, iterations=4, model="Klein")

    ax = plt.figure().add_subplot()
    plot.add_wireframe(ax, vertices, polygons)
    plt.scatter(vertices[0, trace], vertices[1, trace], s=3)
    plt.axis("equal")
    plt.show()
