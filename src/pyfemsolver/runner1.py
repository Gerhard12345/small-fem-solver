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


nodes = []
delta = 0.0
is_boundary = []
x_vals = np.arange(-2, 2.25,0.25)
n = len(x_vals)
y_vals = np.arange(-2, 2.25,0.25)
y_vals2 = np.arange(-2+0.125, 2.125-delta,0.25)
n = len(y_vals)
n1 = len(y_vals2)

height = 0.3
width = 1.2
mesh_size_x = 0.15
mesh_size_y = 0.15
center_x = [0,0]
center_y = [-0.375,0.375]
safety=0.01
for i, x in enumerate(x_vals):
    if i%2 == 0:
        t = y_vals
        n_compare = n
    else:
        t = y_vals2
        n_compare = n1
    for j, y in enumerate(t):
        nodes.append([x,y])
        if i == 0 or i==n-1 or ((j == 0 or j == n_compare-1) and n_compare == n):
            is_boundary.append(True)
        else:
            is_boundary.append(False)


for j in [0,1]:
    for x in np.arange(-width/2, width/2+mesh_size_x/2,mesh_size_x):
        for y in np.arange(-height/2, height/2+mesh_size_y/2,mesh_size_y):
            nodes.append([x-center_x[j],y-center_y[j]])
            is_boundary.append(True)


nodes_full = []
is_boundary_full = []
for pcenter, boundary in zip(nodes,is_boundary):
    if not ((np.abs(pcenter[0]-center_x[0])<width/2-safety and np.abs(pcenter[1]-center_y[0])<height/2-safety) or (np.abs(pcenter[0]-center_x[1])<width/2-safety and np.abs(pcenter[1]-center_y[1])<height/2-safety)):
        nodes_full.append(pcenter)
        is_boundary_full.append(boundary)

import matplotlib.pyplot as plt
tri2 = Delaunay(nodes_full)
tri2.is_boundary = is_boundary_full
to_remove = []
for i,trig in enumerate(tri2.simplices):
    for edge in [[0,1], [1,2],[2,0]]:
        p0 = tri2.points[trig[edge[0]],:]
        p1 = tri2.points[trig[edge[1]],:]
        pcenter = 0.5*(p0+p1)
        if ((np.abs(pcenter[0]-center_x[0])<width/2-safety and np.abs(pcenter[1]-center_y[0])<height/2-safety) or (np.abs(pcenter[0]-center_x[1])<width/2-safety and np.abs(pcenter[1]-center_y[1])<height/2-safety)):
            to_remove.append(i)

temp = []
for i, trig in enumerate(tri2.simplices):
    if i not in to_remove:
        temp.append(trig)

tri2.simplices = np.array(temp)
tri2.edges = list(
    set(
        [
            (int(a), int(b)) if a < b else (int(b), int(a))
            for edge in tri2.simplices
            for a, b in zip(edge, np.roll(edge, -1))
        ]
    )
)

space2 = H1Space(tri2,3)

plt.triplot(tri2.points[:, 0], tri2.points[:, 1], tri2.simplices)
plt.plot(tri2.points[:, 0], tri2.points[:, 1], "o")
plt.plot(tri2.points[space2.tri.is_boundary, 0], tri2.points[space2.tri.is_boundary, 1], "o", color="red")
plt.show()

def g_bound(x,y):
    if type(x) == np.float64:
        x = np.array([x])
        y = np.array([y])
    vals = np.zeros(x.shape)
    for i,point in enumerate(zip(x,y)):
        if (np.abs(point[0]-center_x[0])<width/2+safety and np.abs(point[1]-center_y[0])<height/2+safety):
            vals[i] = -0.1
        elif (np.abs(point[0]-center_x[1])<width/2+safety and np.abs(point[1]-center_y[1])<height/2+safety):
            vals[i] = 0.1
        else:
            vals[i] = 0
    return vals

# u0, mass1, f_vector1 = solve_bvp(10,1,space2,g_bound,lambda x,y:np.zeros(x.shape))
u1, mass1, f_vector1 = solve_bvp(0,1,space2,g_bound,lambda x,y:np.zeros(x.shape))
u2, mass2, f_vector2 = solve_bvp(-1,0.06125*0.5,space2,g_bound,lambda x,y:np.zeros(x.shape))
# ax = show_grid_function(u0,tri2,space2,vrange=[-1,1],dx=0.25,dy=0.25)
# show_boundary_function(g_bound, tri2, space2, ax)
ax = show_grid_function(u1,tri2,space2,vrange=[-0.1,0.1],dx=0.5,dy=0.5)
ax.set_zlim([-5.0,5.0])
#show_boundary_function(g_bound, tri2, space2, ax)
ax = show_grid_function(u2,tri2,space2,vrange=[-0.18,0.18],dx=0.5,dy=0.5)
#show_boundary_function(g_bound, tri2, space2, ax)
ax.set_zlim([-5.0,5.0])
plt.show()