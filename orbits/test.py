import numpy as np
import matplotlib.pyplot as plt


def d(x, y):
    u = y - x

    s = 1 / (u @ u)
    b = s * (x @ u)
    c = s * (1 - x @ x)

    d = np.sqrt(b*b + c)
    t1 = -b + d
    t2 = -b - d

    # magic
    return .5 * np.log(t1 * (t2 - 1) / (t2 * (t1 - 1)))

n = 5
res = 3
edge = np.sqrt(1 / n) * .9

t = np.linspace(-edge, edge, res)


c1 = .5
c2 = np.sqrt(1 - c1 * c1)
def T(x):
    off = np.zeros_like(x)
    off[0] = (x[0] + c1) / c2 - x[0]
    return c2 / (1 + c1 * x[0]) * (x + off)

M = 0
RES = res ** np.arange(0, n)
for _i in range(res ** n):
    i = (_i // RES) % res
    for _j in range(_i + 1, res **n):
        j = (_j // RES) % res

        x1 = t[i]
        x2 = t[j]
        d1 = d(x1, x2)
        d2 = d(T(x1), T(x2))

        M = np.max(np.abs(d1 - d2), 0)

print(M)