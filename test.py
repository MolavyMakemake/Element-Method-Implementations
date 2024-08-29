import numpy as np


arr = np.array([[0, 0], [0, 0]])

arr[(0, 0), (0, 0)] += 1

print(arr)

