import numpy as np
from mesh import *

orbit_sgn_r1 = ["p1 (o)", "pm (**)", "pg (xx)", "cm (*x)"]
orbit_sgn_r2 = ["p2 (2222)", "pmm (*2222)", "pmg (22*)", "pgg (22x)", "cmm (2*22)"]
orbit_sgn_r3 = ["p3 (333)", "p3m1 (*333)", "p31m (3*3)"]
orbit_sgn_r4 = ["p4 (442)", "p4m (*442)", "p4g (4*2)"]
orbit_sgn_r6 = ["p6 (632)", "p6m (*632)"]

orbit_sgn = orbit_sgn_r1 + orbit_sgn_r2 \
            + orbit_sgn_r3 + orbit_sgn_r4 + orbit_sgn_r6

class IdMap:
    def __init__(self, N):
        self.idmap = [{i} for i in range(N)]

    def identify(self, I, J):
        print(I, J)
        for i, j in zip(I, J):
            self.idmap[i].add(j)

    def resolve(self):
        for j in range(len(self.idmap)):
            I = self.idmap[j].copy()
            while len(I) > 0:
                i = I.pop()
                if self.idmap[i] is self.idmap[j]:
                    continue

                self.idmap[j].update(self.idmap[i])
                I.update(self.idmap[i])

                self.idmap[i] = self.idmap[j]

        idmap_sparse = [[], []]
        for j in range(len(self.idmap)):
            self.idmap[j].discard(j)
            idmap_sparse[0].extend(self.idmap[j])
            idmap_sparse[1].extend([j] * len(self.idmap[j]))
            self.idmap[j].clear()

        return idmap_sparse

def mesh(signature, res_x, res_y, nV=3):
    if signature in orbit_sgn_r1 + orbit_sgn_r2:
        return square(res_x, res_y, nV)

    elif signature in orbit_sgn_r4:
        return square(res_x, res_x, nV)

    elif signature in orbit_sgn_r6 + ["p31m (3*3)"]:
        return triangle(res_x, nV)

    else:
        return rhombus(res_x, res_y, nV)


def _idmap_r1(signature, W, H):
    idmap = IdMap(W*H)
    W2 = (W + 1) // 2
    H2 = (H + 1) // 2
    W4 = (W2 + 1) // 2

    r12 = np.arange(0, W2 * H)
    x12 = r12 % W2
    y12 = (r12 // W2) * W

    r14 = np.arange(0, W4 * H)
    x14 = r14 % W4
    y14 = (r14 // W4) * W

    r22 = np.arange(0, W2 * H2)
    x22 = r22 % W2
    y22 = (r22 // W2) * W

    if signature == "pm (**)":
        idmap.identify(W - 1 - x12 + y12, x12 + y12)

    elif signature == "pg (xx)":
        idmap.identify(W * H - W2 - y12 + x12, x12 + y12)

    elif signature == "cm (*x)":
        idmap.identify(W2 - 1 - x14 + y14, x14 + y14)
        idmap.identify(W - W2 + x22 + y22, W * H - W * H2 + x22 + y22)
        idmap.identify(W * H - W * H2 + W - W2 + x22 + y22, x22 + y22)

    idmap.identify(range(W * (H - 1), W * H - 1), range(0, W - 1))
    idmap.identify(range(W - 1, W * H - 1, W), range(0, W * (H - 1), W))
    idmap.identify([W * H - 1], [0])

    return idmap.resolve()


def _idmap_r2(signature, W, H):
    idmap = IdMap(W*H)

    W2 = (W + 1) // 2
    H2 = (H + 1) // 2
    W4 = (W2 + 1) // 2

    r12 = np.arange(0, W2 * H)
    x12 = r12 % W2
    y12 = (r12 // W2) * W

    r14 = np.arange(0, W4 * H)
    x14 = r14 % W4
    y14 = (r14 // W4) * W

    r22 = np.arange(0, W2 * H2)
    x22 = r22 % W2
    y22 = (r22 // W2) * W

    r24 = np.arange(0, W4 * H2)
    x24 = r24 % W4
    y24 = (r24 // W4) * W

    if signature == "p2 (2222)":
        idmap.identify(W * H - 1 - x12 - y12, x12 + y12)

    elif signature == "pmm (*2222)":
        idmap.identify(W - 1 - x22 + y22, x22 + y22)
        idmap.identify(W * (H-1) + x22 - y22, x22 + y22)
        idmap.identify(W*H - 1 - x22 - y22, x22 + y22)

    elif signature == "pmg (22*)":
        idmap.identify(W2 - 1 - x14 + y14, x14 + y14)
        idmap.identify(W*H - W2 + x14 - y14, x14 + y14)
        idmap.identify(W*H - 1 - x14 - y14, x14 + y14)

    elif signature == "pgg (22x)":
        idmap.identify(W * (H2-1) + W2 - 1 - x24 - y24, x24 + y24)
        idmap.identify(W * (H-1) + W2 - 1 - x24 - y24, W * (H-H2) + x24 + y24)
        idmap.identify(W - 1 - x22 + y22, W*(H - H2) + x22 + y22)
        idmap.identify(W*H - W2 + x22 - y22, x22 + y22)

    elif signature == "cmm (2*22)":
        idmap.identify(W * (H2-1) + W2 - 1 - x24 - y24, x24 + y24)
        idmap.identify(W * (H-1) + x22 - y22, x22 + y22)
        idmap.identify(W*H - 1 - x12 - y12, x12 + y12)

    idmap.identify(range(W * (H - 1), W * H - 1), range(0, W - 1))
    idmap.identify(range(W - 1, W * H - 1, W), range(0, W * (H - 1), W))
    idmap.identify([W * H - 1], [0])

    return idmap.resolve()


def _idmap_r3(signature, W):
    if signature == "p3 (333)":
        idmap = IdMap(W*W)
        idmap.identify(range(W*W - 1, W-1, -W), range(0, W - 1))
        idmap.identify(range(W*W - 2, W * (W-1), -1), range(W, W * (W-1), W))
        return idmap.resolve()

    if signature == "p3m1 (*333)":
        j = np.repeat(range(W), range(W, 0, -1))
        i = np.arange(W * (W + 1) // 2) - W * j + ((j - 1) * j) // 2

        idmap = IdMap(W*W)
        idmap.identify(W*W - 1 - i * W - j, i + j * W)
        idmap.identify(range(W*W - 1, W-1, -W), range(0, W-1))

    elif signature == "p31m (3*3)":
        H = (W + 2) // 3
        k = np.arange(H)
        j = np.repeat(range(H), W - 3 * k)
        x = np.arange(len(j)) + j ** 2
        y = j * W - (j - 1) * j // 2
        i = x - y

        tri_1 = W * (i+1) - (i+1) * i // 2 - 1 - j
        tri_2 = W * (W+1) // 2 - (i + j) * (i + j + 3) // 2 + j - 1

        print(W)

        idmap = IdMap((W+1) * W // 2)
        idmap.identify(tri_1, x)
        idmap.identify(tri_2, x)

        r = np.arange(W)
        idmap.identify(tri_1[r], r)
        idmap.identify(tri_2[r], r)

    return idmap.resolve()


def _idmap_r4(signature, W):
    idmap = IdMap(W * W)

    W2 = (W + 1) // 2
    W4 = (W2 + 1) // 2

    x12 = np.arange(0, W2 * W) % W2
    y12 = (np.arange(0, W2 * W) // W2) * W
    i22 = np.arange(0, W2 * W2) % W2
    j22 = np.arange(0, W2 * W2) // W2

    if signature == "p4 (442)":
        idmap.identify(W * (W - 1) - i22 * W + j22, i22 + j22 * W)
        idmap.identify(W * W - 1 - x12 - y12, x12 + y12)

    elif signature == "p4m (*442)":
        idmap.identify(i22 * W + j22, i22 + j22 * W)
        idmap.identify(W * (W - 1) + i22 - j22 * W, i22 + j22 * W)
        idmap.identify(W * W - 1 - x12 - y12, x12 + y12)

    elif signature == "p4g (4*2)":
        idmap.identify(W2 * (W - 1) + W2 - 1 - j22 - i22 * W, i22 + j22 * W)
        idmap.identify(W * (W - 1) - i22 * W + j22, i22 + j22 * W)
        idmap.identify(W * W - 1 - x12 - y12, x12 + y12)

    idmap.identify(range(W * (W - 1), W * W - 1), range(0, W - 1))
    idmap.identify(range(W - 1, W * W - 1, W), range(0, W * (W - 1), W))
    idmap.identify([W * W - 1], [0])

    return idmap.resolve()


def _idmap_r6(signature, W):
    if signature == "p6 (632)":
        pass

    elif signature == "p6m (*632)":
        pass


def compute_idmap(signature, W, H):
    if signature in orbit_sgn_r1:
        return _idmap_r1(signature, W, H)

    elif signature in orbit_sgn_r2:
        return _idmap_r2(signature, W, H)

    elif signature in orbit_sgn_r3:
        return _idmap_r3(signature, W)

    elif signature in orbit_sgn_r4:
        return _idmap_r4(signature, W)

    elif signature in orbit_sgn_r6:
        return _idmap_r6(signature, W)