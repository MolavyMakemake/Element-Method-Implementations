import numpy as np

orbit_sgn = [
    "p1 (o)", "p2 (2222)", "pm (**)", "pg (xx)",
    "cm (*x)"
]


def compute_idmap(signature, W, H):
    W2 = (W + 1) // 2
    H2 = (H + 1) // 2
    idmap = [[], []]

    def identify(i, j):
        idmap[0].extend(i)
        idmap[1].extend(j)

    if signature == "p1 (o)":
        identify(range(W * (H - 1), W*H - 1), range(0, W - 1))
        identify(range(W - 1, W*H - 1, W), range(0, W * (H - 1), W))
        identify([W * H - 1], [0])

    elif signature == "p2 (2222)":
        r = np.arange(0, W2 * H2)
        i = (r // W2) * W2 + r % W2
        identify(W*H - 1 - i, i)
        pass

    return idmap
