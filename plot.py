import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import tri

alpha_gradiant = {
    'alpha': [(0, 0, 0),
            (1, 1, 0)],
    'red': [(0, 0, 0),
            (1, 0, 0)],
    'green': [(0, 0, 0),
            (1, 0, 0)],
    'blue': [(0, 0, 0),
            (1, 0, 0)]
}

alpha_cm = colors.LinearSegmentedColormap('my_colormap2',alpha_gradiant,256)

def complex(vertices, triangles, u):
    ax = plt.figure().add_subplot()

    tr = tri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax.tricontourf(tr, np.angle(u), levels=100, cmap='hsv')
    ax.tricontourf(tr, np.abs(u), cmap=alpha_cm)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='X', ylabel='Y')
    ax.set_aspect("equal")

def surface(x, y, polygons, u, ax):
    ax.set(xlabel='X', ylabel='Y')
    ax.triplot(x, y, np.real(u))