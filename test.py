import numpy as np
W = 10
H = (W + 2) // 3
W2 = (W + 1) // 2

k = np.arange(H)
j = np.repeat(range(H), W - 3 * k)
x = np.arange(len(j)) + j ** 2
y = j * W - (j - 1) * j // 2
i = x - y

r2 = (W - 3 * k + 1) // 2
j2 = np.repeat(range(H), r2)
i2 = np.roll(r2, 1); i2[0] = 0
i2 = np.arange(len(j2)) - np.repeat(np.cumsum(i2), r2)

print(j2 * W - j2 * (j2-1) // 2 + j2 + i2)
print((j2+1) * W - j2 * (j2+1) // 2 - 1 - j2 - i2)

tri_1 = W * (i + 1) - (i + 1) * i // 2 - 1 - j
tri_2 = W * (W + 1) // 2 - (i + j) * (i + j + 3) // 2 + j - 1

