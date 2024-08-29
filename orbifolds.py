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
    if signature in ["cm (*x)", "p3 (333)", "p3m1 (*333)", "p6 (632)"]:
        return rhombus(res_x, res_x, nV)

    elif signature in ["pgg (22x)", "cmm (2*22)", "p4 (442)", "p4m (*442)", "p4g (4*2)"]:
        return square(res_x, res_x, nV)

    elif signature in orbit_sgn_r1 + orbit_sgn_r2:
        return square(res_x, res_y, nV)

    elif signature in ["p31m (3*3)", "p6m (*632)"]:
        return triangle(res_x, nV)


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
        idmap = IdMap(W * W)

        j2 = np.repeat(range(W), range(W, 0, -1))
        i2 = np.arange(W * (W + 1) // 2) - W * j2 + ((j2 - 1) * j2) // 2

        idmap.identify(W * W - 1 - i2 * W - j2, i2 + j2 * W)
        return idmap.resolve()

    idmap.identify(range(W * (H - 1), W * H), range(0, W))
    idmap.identify(range(W - 1, W * H, W), range(0, W * H, W))

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
        idmap.identify(W * (H-1) + W2 - 1 - x22 - y22, x22 + y22)
        idmap.identify(W*H - W2 + x12 - y12, x12 + y12)

    elif signature == "pgg (22x)":
        j2 = np.repeat(range(W), range(W, 0, -1))
        i2 = np.arange(W * (W + 1) // 2) - W * j2 + ((j2 - 1) * j2) // 2

        idmap = IdMap(W * W)
        idmap.identify(W*W - 1 - i2 - j2 * W, i2 + j2 * W)
        idmap.identify(range(W-1, W*W, W), range(W))
        return idmap.resolve()

    elif signature == "cmm (2*22)":
        j2 = np.repeat(range(W), range(W, 0, -1))
        i2 = np.arange(W * (W + 1) // 2) - W * j2 + ((j2 - 1) * j2) // 2

        idmap = IdMap(W * W)
        idmap.identify(W * W - 1 - i2 - j2 * W, i2 + j2 * W)
        return idmap.resolve()

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
        idmap.identify(range(W * W - 2, W * (W - 1), -1), range(W, W * (W - 1), W))

    elif signature == "p31m (3*3)":
        H = (W + 2) // 3
        k = np.arange(H)
        j = np.repeat(range(H), W - 3 * k)
        x = np.arange(len(j)) + j ** 2
        y = j * W - (j - 1) * j // 2
        i = x - y

        tri_1 = W * (i+1) - (i+1) * i // 2 - 1 - j
        tri_2 = W * (W+1) // 2 - (i + j) * (i + j + 3) // 2 + j - 1

        idmap = IdMap((W+1) * W // 2)
        idmap.identify(tri_1, x)
        idmap.identify(tri_2, x)

        r = np.arange(W)
        idmap.identify(tri_1[r], r)
        idmap.identify(tri_2[r], r)

    return idmap.resolve()


def _idmap_r4(signature, W):
    if signature == "p4 (442)":
        idmap = IdMap(W * W)
        idmap.identify(range(W*W, W), range(W))
        idmap.identify(range(W * (W-1), W*W), range(W-1, W*W, W))
        return idmap.resolve()

    elif signature == "p4m (*442)":
        j2 = np.repeat(range(W), range(W, 0, -1))
        i2 = np.arange(W * (W + 1) // 2) - W * j2 + ((j2 - 1) * j2) // 2

        idmap = IdMap(W * W)
        idmap.identify(W * W - 1 - i2 * W - j2, i2 + j2 * W)
        return idmap.resolve()

    elif signature == "p4g (4*2)":
        j2 = np.repeat(range(W), range(W, 0, -1))
        i2 = np.arange(W * (W + 1) // 2) - W * j2 + ((j2 - 1) * j2) // 2

        idmap = IdMap(W * W)
        idmap.identify(W * W - 1 - i2 * W - j2, i2 + j2 * W)
        idmap.identify(range(0, W*W, W), range(W))
        return idmap.resolve()


def _idmap_r6(signature, W):
    if signature == "p6 (632)":
        H = (W + 2) // 3
        k = np.arange(H)
        j = np.repeat(range(H), W - 3 * k)
        x = np.arange(len(j)) + j ** 2
        y = j * W - (j - 1) * j // 2
        i = x - y

        j2 = np.repeat(range(W), range(W, 0, -1))
        i2 = np.arange(W * (W + 1) // 2) - W * j2 + ((j2 - 1) * j2) // 2

        tri_1 = W * (i + 1) - (i + 1) * i // 2 - 1 - j
        tri_2 = W * (W + 1) // 2 - (i + j) * (i + j + 3) // 2 + j - 1

        tri_1 += i * (i-1) // 2
        tri_2 += (W-2 - i - j) * (W-1 - i - j) // 2
        x += j * (j-1) // 2

        idmap = IdMap(W * W)
        idmap.identify(tri_1, x)
        idmap.identify(tri_2, x)

        idmap.identify(W*W - 1 - i2 - j2 * W, i2 + j2 * W)

        idmap.identify(range(W * (W - 1), W * W - 1), range(0, W - 1))
        idmap.identify(range(W - 1, W * W - 1, W), range(0, W * (W - 1), W))
        idmap.identify([W * W - 1], [0])

        return idmap.resolve()

    elif signature == "p6m (*632)":
        H = (W + 2) // 3
        k = np.arange(H)
        j = np.repeat(range(H), W - 3 * k)
        x = np.arange(len(j)) + j ** 2
        y = j * W - (j - 1) * j // 2
        i = x - y

        r2 = (W - 3 * k + 1) // 2
        j2 = np.repeat(range(H), r2)
        i2 = np.roll(r2, 1);
        i2[0] = 0
        i2 = np.arange(len(j2)) - np.repeat(np.cumsum(i2), r2)

        tri_1 = W * (i + 1) - (i + 1) * i // 2 - 1 - j
        tri_2 = W * (W + 1) // 2 - (i + j) * (i + j + 3) // 2 + j - 1

        idmap = IdMap(W * W)
        idmap.identify((j2+1) * W - j2 * (j2+1) // 2 - 1 - j2 - i2, j2 * W - j2 * (j2-1) // 2 + j2 + i2)
        idmap.identify(tri_1, x)
        idmap.identify(tri_2, x)

        r = np.arange(W)
        idmap.identify(tri_1[r], r)
        idmap.identify(tri_2[r], r)

        return idmap.resolve()


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