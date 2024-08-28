import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import tri
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

alpha_gradiant = {
    'alpha': [(0, 0, 0.8),
            (0.7, 0.3, 0.3),
            (1, 0.5, 0)],
    'red': [(0, 0, 0),
            (1, 1, 0)],
    'green': [(0, 0, 0),
            (1, 1, 0)],
    'blue': [(0, 0, 0),
            (1, 1, 0)]
}

alpha_cm = colors.LinearSegmentedColormap('my_colormap2',alpha_gradiant,256)

def complex(ax, vertices, triangles, u):
    tr = tri.Triangulation(vertices[0, :], vertices[1, :], triangles)
    arg = np.angle(u) / np.pi + 1
    arg = np.minimum(arg, 2 - arg)

    shade = 0.5 * np.log(np.maximum(np.abs(u), 1e-10)) % 1

    ax.tricontourf(tr, arg, levels=50, vmin=0, vmax=1, cmap='hsv')
    ax.tricontourf(tr, shade, levels=30, vmin=0, vmax=1, cmap=alpha_cm)
    #ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='X', ylabel='Y')
    ax.set_aspect("equal")

def surface(ax, vertices, triangles, u):
    ax.plot_trisurf(vertices[0, :], vertices[1, :], triangles, u, cmap="Blues")
    ax.set(xlabel='X', ylabel='Y')
    #ax.set_aspect("equal")


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
