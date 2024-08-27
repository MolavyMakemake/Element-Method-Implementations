import numpy as np
N = 10

def triangle_i(n):
    j = np.repeat(range(N), range(N, 0, -1))
    i = np.arange(N * (N+1) // 2) - N * j + ((j - 1) * j) // 2
    return i, j

x_i, y_i = triangle_i(N)

H = (N+2) // 3
k = np.arange(H)
j = np.repeat(range(H), N - 3 * k)

x = np.arange(len(j)) + j**2
y = j * N - (j-1) * j // 2

i = x - y
print(H)
print(N - 3 * k)
print(x, y)
print(i, j)
print(N * (i+1) - i * (i+1) // 2 - 1 - j)
print()
print()