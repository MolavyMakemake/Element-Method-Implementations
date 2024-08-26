import numpy as np


def square(res_x, res_y, nV=3):
    vertices = np.zeros([2, res_x * res_y])

    X, Y = np.meshgrid(np.linspace(-1, 1, res_x), np.linspace(-1, 1, res_y))
    vertices[0, :] = X.flatten()
    vertices[1, :] = Y.flatten()

    polygons = []
    if nV==4:
        for y_i in range(res_y - 1):
            for x_i in range(res_x - 1):
                i = res_x * y_i + x_i
                polygons.append([i, i + 1, i + res_x + 1, i + res_x])
    else:
        for y_i in range(res_y - 1):
            for x_i in range(res_x - 1):
                i = res_x * y_i + x_i
                polygons.append([i, i + res_x + 1, i + res_x])
                polygons.append([i, i + 1, i + res_x + 1])

    trace = []
    trace.extend(range(res_x))
    trace.extend(range(res_x * (res_y - 1), res_x * res_y))
    trace.extend(range(res_x, res_x * (res_y - 1), res_x))
    trace.extend(range(2 * res_x - 1, res_x * res_y - 1, res_x))

    return vertices, polygons, trace


def disk(res_x, res_y, nV=3):
    vertices, polygons, trace = square(res_x, res_y, nV)

    for v in vertices.T:
        if abs(v[0]) > abs(v[1]):
            v /= np.linalg.norm(v / v[0])
        elif v[1] != 0:
            v /= np.linalg.norm(v / v[1])

    return vertices, polygons, trace

def rhombus(res_x, res_y, nV=3):
    vertices, polygons, trace = square(res_x, res_y, nV)
    return np.array([[np.sqrt(1.25), 0.5], [0, 1]]) @ vertices, polygons, trace

def triangle(res, nV=3):
    vertices = np.zeros([2, res * (res + 1) // 2])
    polygons = []

    y = np.repeat(range(res), range(res, 0, -1))
    x = np.arange(res * (res + 1) // 2) - res * y + ((y - 1) * y) // 2

    vertices[1, :] = 2 * y / (res - 1) - 1
    vertices[0, :] = 2 * x / (res - 1) + 0.5 * vertices[1, :]

    i = 0
    while i < res * (res + 1) // 2 - 1:
        W = int(res - y[i])

        if x[i] >= W - 2:
            polygons.append([i, i + 1, i + W])
            i += 2
        else:
            polygons.append([i, i + 1, i + W])
            polygons.append([i + 1, i + 1 + W, i + W])
            i += 1

    trace = []
    k = np.arange(res - 1)
    l = (k + 1) * res - k * (k + 1) // 2
    trace.extend(k)
    trace.extend(l)
    trace.extend(l - 1)

    return vertices, polygons, trace

def generic(domain, res_x, res_y):
    if domain == "rectangle":
        return square(res_x, res_y)

    elif domain == "elliptic disk":
        return disk(res_x, res_y)

    elif domain == "rhombus":
        return rhombus(res_x, res_y)

    elif domain == "triangle":
        return triangle(res_x)

