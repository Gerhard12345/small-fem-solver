import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from space import H1Space
from solving import solve_bvp
from element import (
    print_matrix,
    duffy,
    barycentric_coordinates,
    barycentric_coordinates_line,
)
from visual import (
    show_grid_function,
    show_shape,
    show_edge_shape,
    show_boundary_function,
)

points = np.array(
    [[0, 0], [0.5, 0], [1, 0], [1, 1], [0.5, 1], [0, 1], [0, 0.5], [1, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75], [0.5, 0.5]]
)
tri1 = Delaunay(points)
tri1.edges = list(set([(int(a), int(b)) if a < b else (int(b), int(a)) for edge in tri1.simplices for a, b in zip(edge, np.roll(edge, -1))]))
tri1.is_boundary = np.array([True, True, True, True, True, True, True, True, False, False, False, False, False])


def g(x, y):
    # return np.sin(np.pi * x) + np.sin(np.pi * y)
    # return np.exp(-((x - 0.25) ** 2) * 32) + np.exp(-((y - 0.25) ** 2) * 32)
    return (x - 0.5) ** 1 + (y - 0.5) ** 1
    # return np.ones(x.shape)


def g3(x, y):
    # return np.sin(np.pi * x) + np.sin(np.pi * y)
    # return np.exp(-((x - 0.25) ** 2) * 32) + np.exp(-((y - 0.25) ** 2) * 32)
    return (x - 0.5) ** 3 + (y - 0.5) ** 3
    # return np.ones(x.shape)


def f(x, y):
    # return np.ones(x.shape)
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


p = 4
space1 = H1Space(tri1, p)
u2, mass, f_vector = solve_bvp(1, 0, space1, g3, f)
u3, mass, f_vector = solve_bvp(0, 1, space1, f, f)


ax = show_grid_function(u2, tri1, space1, vrange=[-1.025, 1.025], dx=0.1, dy=0.1)
show_boundary_function(g3, tri1, space1, ax)
ax = show_grid_function(u3, tri1, space1, vrange=[-0.0125, 0.0125], dx=0.1, dy=0.1)
show_boundary_function(f, tri1, space1, ax)

f_vec = space1.elements[0].calc_element_vector(np.array([[-1, -1], [1, -1], [0, 1]]), lambda x, y: np.ones(x.shape))
m = space1.elements[0].calc_mass_matrix(np.array([[-1, -1], [1, -1], [0, 1]]))
du_dv = space1.elements[0].calc_gradu_gradv_matrix(np.array([[-1, -1], [1, -1], [0, 1]]))

plt.show()


nodes = []
delta = 0.0
is_boundary = []
x_vals = np.arange(-1, 1.25, 0.25)
n = len(x_vals)
y_vals = np.arange(-1, 1.25, 0.25)
y_vals2 = np.arange(-1 + 0.125, 1.125 - delta, 0.25)
n = len(y_vals)
n1 = len(y_vals2)
for i, x in enumerate(x_vals):
    if i % 2 == 0:
        t = y_vals
        n_compare = n
    else:
        t = y_vals2
        n_compare = n1
    for j, y in enumerate(t):
        nodes.append([x, y])
        if i == 0 or i == n - 1 or ((j == 0 or j == n_compare - 1) and n_compare == n):
            is_boundary.append(True)
        else:
            is_boundary.append(False)

for x in np.arange(-0.15, 0.16, 0.3):
    for y in np.arange(-0.15, 0.16, 0.3):
        nodes.append([x, y])
        is_boundary.append(True)


nodes_full = []
is_boundary_full = []
for node, boundary in zip(nodes, is_boundary):
    if not (np.abs(node[0]) < 0.14 and np.abs(node[1]) < 0.14):
        nodes_full.append(node)
        is_boundary_full.append(boundary)

import matplotlib.pyplot as plt

tri2 = Delaunay(nodes_full)
tri2.is_boundary = is_boundary
to_remove = []
for i, trig in enumerate(tri2.simplices):
    for edge in [[0, 1], [1, 2], [2, 0]]:
        p0 = tri2.points[trig[edge[0]], :]
        p1 = tri2.points[trig[edge[1]], :]
        pcenter = 0.5 * (p0 + p1)
        if np.abs(pcenter[0]) < 0.14 and np.abs(pcenter[1]) < 0.14:
            to_remove.append(i)

temp = []
for i, trig in enumerate(tri2.simplices):
    if i not in to_remove:
        temp.append(trig)

tri2.simplices = np.array(temp)
tri2.is_boundary = is_boundary_full
tri2.edges = list(set([(int(a), int(b)) if a < b else (int(b), int(a)) for edge in tri2.simplices for a, b in zip(edge, np.roll(edge, -1))]))


plt.triplot(tri2.points[:, 0], tri2.points[:, 1], tri2.simplices)
plt.plot(tri2.points[:, 0], tri2.points[:, 1], "o")

space2 = H1Space(tri2, 3)
u32, mass2, f_vector2 = solve_bvp(0, 1, space2, g3, f)
u42, mass2, f_vector2 = solve_bvp(0, 3, space2, lambda x, y: x * x + y * y, lambda x, y: -np.zeros(x.shape))
ax = show_grid_function(u32, tri2, space2, vrange=[-6.01, 0.01], dx=0.125, dy=0.125)
show_boundary_function(g3, tri2, space2, ax)
show_grid_function(u42, tri2, space2, vrange=[0, 2], dx=0.125, dy=0.125)
plt.show()
