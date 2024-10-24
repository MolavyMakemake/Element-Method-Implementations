import numpy as np
import matplotlib.pyplot as plt

N = 1000
X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

r = 1
A = np.array([1.2, 2.5])
B = np.array([0.5, 0.5])

m = np.array([A[0], A[1] * np.cosh(r)])
s = A[1] * np.sinh(r)

X *= s; Y *= s
X += m[0]; Y += m[1]

I = 1 / (Y * Y)
dS = (X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0])

#l_con = (X - A[0]) * (B[1] - A[1]) + (Y - A[1]) * (A[0] - B[0]) > 0
l_con = (X - A[0] - .1) ** 2 + (Y) ** 2 < A[1] ** 2 + .01
s_con = (X - m[0]) ** 2 + (Y - m[1]) ** 2 < s * s

print(np.sum(I, where=s_con * l_con) * dS)
print(np.sum(I, where=s_con * np.logical_not(l_con)) * dS)