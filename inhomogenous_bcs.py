import matplotlib.pyplot as plt
import numpy as np
from space import H1Space
from solving import solve_bvp
from visual import (
    show_grid_function,
    show_boundary_function,
)

from meshing import generate_mesh

def g3(x, y):
    return (x - 0.5) ** 3 + (y - 0.5) ** 3

def f(x,y):
    return np.zeros(x.shape)


orders = [1,4]
edge_mesh_sizes = [0.4,0.5]
domain_mesh_sizes = [0.4,0.5]
for order, edge_mesh_size, domain_mesh_size in zip(orders, edge_mesh_sizes, domain_mesh_sizes):
    data = {
        "lines": [
            # Region 1's outer boundary
            {"start": [-1, -1], "end": [1, -1], "left_region": 1, "right_region": 0, "h": edge_mesh_size, "boundary_index": 1},
            {"start": [1, -1], "end": [1, 1], "left_region": 1, "right_region": 0, "h": edge_mesh_size, "boundary_index": 1},
            {"start": [1, 1], "end": [-1, 1], "left_region": 1, "right_region": 0, "h": edge_mesh_size, "boundary_index": 1},
            {"start": [-1, 1], "end": [-1, -1], "left_region": 1, "right_region": 0, "h": edge_mesh_size, "boundary_index": 2},
        ],
        "regions": [
            {"region_id": 1, "mesh_inner": domain_mesh_size},
        ]
    }


    max_gradient = 0.07
    tri = generate_mesh(data, max_gradient)



    space = H1Space(tri, order)

    u, mass2, f_vector2 = solve_bvp(0,1,space,g3,f)
    ax,mini, maxi = show_grid_function(u,space,vrange=[-6.01,0.01],dx=0.125,dy=0.125)
    show_boundary_function(g3, tri, ax)
    plt.show()