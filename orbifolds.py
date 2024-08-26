import numpy as np

orbit_sgn_11 = ["p1 (o)"]
orbit_sgn_12 = ["p2 (2222)", "pm (**)", "pg (xx)"]
orbit_sgn_24 = ["cm (*x)", "pmm (*2222)", "pmg (22*)", "pgg (22x)", "cmm (2*22)"]

orbit_sgn = orbit_sgn_11 + orbit_sgn_12 + orbit_sgn_24


def compute_idmap(signature, W, H):
    idmap = [{i} for i in range(W*H)]

    def identify(I, J):
        print(I, J)
        for i, j in zip(I, J):
            idmap[i].add(j)

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
        identify(W * H - 1 - x12 - y12, x12 + y12)

    elif signature == "pm (**)":
        identify(W - 1 - x12 + y12, x12 + y12)

    elif signature == "pg (xx)":
        identify(W * H - W2 - y12 + x12, x12 + y12)

    elif signature == "cm (*x)":
        identify(W2 - 1 - x14 + y14, x14 + y14)
        identify(W - W2 + x22 + y22, W*H - W*H2 + x22 + y22)
        identify(W*H - W*H2 + W - W2 + x22 + y22, x22 + y22)

    elif signature == "pmm (*2222)":
        identify(W - 1 - x22 + y22, x22 + y22)
        identify(W * (H-1) + x22 - y22, x22 + y22)
        identify(W*H - 1 - x22 - y22, x22 + y22)

    elif signature == "pmg (22*)":
        identify(W2 - 1 - x14 + y14, x14 + y14)
        identify(W*H - W2 + x14 - y14, x14 + y14)
        identify(W*H - 1 - x14 - y14, x14 + y14)

    elif signature == "pgg (22x)":
        identify(W * (H2-1) + W2 - 1 - x24 - y24, x24 + y24)
        identify(W * (H-1) + W2 - 1 - x24 - y24, W * (H-H2) + x24 + y24)
        identify(W - 1 - x22 + y22, W*(H - H2) + x22 + y22)
        identify(W*H - W2 + x22 - y22, x22 + y22)

    elif signature == "cmm (2*22)":
        identify(W * (H2-1) + W2 - 1 - x24 - y24, x24 + y24)
        identify(W * (H-1) + x22 - y22, x22 + y22)
        identify(W*H - 1 - x12 - y12, x12 + y12)

    elif signature == "p4 (*442)":
        pass

    identify(range(W * (H - 1), W * H - 1), range(0, W - 1))
    identify(range(W - 1, W * H - 1, W), range(0, W * (H - 1), W))
    identify([W * H - 1], [0])

    print(idmap)

    for j in range(W*H):
        I = idmap[j].copy()
        while len(I) > 0:
            i = I.pop()
            if idmap[i] is idmap[j]:
                continue

            idmap[j].update(idmap[i])
            I.update(idmap[i])

            idmap[i] = idmap[j]

    idmap_sparse = [[], []]
    for j in range(W*H):
        idmap[j].discard(j)
        idmap_sparse[0].extend(idmap[j])
        idmap_sparse[1].extend([j] * len(idmap[j]))
        idmap[j].clear()

    return idmap_sparse
