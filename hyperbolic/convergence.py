import numpy as np
import matplotlib.pyplot as plt
import plot, FEM_BKDISK_O1, FEM_PDISK_O1, triangulate

N = 3
models = [FEM_PDISK_O1.Model(
    *triangulate.generate(p=3, q=7, iterations=i,
        subdivisions=1, model="Poincare", minimal=True))
    for i in range(N)]

f = lambda z: 1
U = [models[i].solve_poisson(f) for i in range(N)]

for i in range(N - 1):
    pass