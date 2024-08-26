import numpy as np
N = 4

def triangle_i(n):
    j = np.repeat(range(N), range(N, 0, -1))
    i = np.arange(N * (N+1) // 2) - N * j + ((j - 1) * j) // 2
    return i, j


x_i, y_i = triangle_i(N)

print(y_i)
k = np.arange(N-1)
l = (k+1) * N - (k) * (k+1) // 2
print(k)
print(l)
print(l-1)