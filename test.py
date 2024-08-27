import numpy as np
W = 10

H = (W + 2) // 3
k = np.arange(H)
j = np.repeat(range(H), W - 3 * k)
x = np.arange(len(j)) + j ** 2
y = j * W - (j - 1) * j // 2
i = x - y

tri_1 = W * (i+1) - (i+1) * i // 2 - 1 - j
tri_2 = W * (W+1) // 2 - (i + j) * (i + j + 3) // 2 + j - 1

print(i + j)
print(tri_1)
print(tri_2)
print(x + y)