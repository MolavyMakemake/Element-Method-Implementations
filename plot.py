import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import tri
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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

def complex(ax, vertices, triangles, u):
    tr = tri.Triangulation(vertices[0, :], vertices[1, :], triangles)
    ax.tricontourf(tr, np.angle(u), levels=100, vmin=-np.pi, vmax=np.pi, cmap='hsv')
    ax.tricontourf(tr, np.abs(u), cmap=alpha_cm)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='X', ylabel='Y')
    ax.set_aspect("equal")

def surface(ax, vertices, triangles, u):
    ax.set(xlabel='X', ylabel='Y')
    ax.plot_trisurf(vertices[0, :], vertices[1, :], triangles, u, cmap="Blues")


def add_wireframe(ax, vertices, polygons, u=None):
    lines = []
    if u is None:
        for p_i in polygons:
            p = np.array([np.take(vertices[0, :], p_i + [p_i[0]]), np.take(vertices[1, :], p_i + [p_i[0]])]).T
            lines.append(p)

        ax.add_collection(LineCollection(lines, linewidths=0.2, edgecolors="black"))

    else:
        for p_i in polygons:
            p = np.array([np.take(vertices[0, :], p_i + [p_i[0]]), np.take(vertices[1, :], p_i + [p_i[0]]), np.take(u, p_i + [p_i[0]])]).T
            lines.append(p)

        ax.add_collection(Line3DCollection(lines, linewidths=0.5, edgecolors="black"))