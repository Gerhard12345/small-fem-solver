import matplotlib.pyplot as plt
import numpy as np
from space import H1Space
from solving import solve_bvp
from visual import (
    show_grid_function,
    show_boundary_function,
)

from meshing import generate_mesh

orders = [1, 4]
edge_mesh_sizes = [0.1, 0.25]
domain_mesh_sizes = [[0.1, 0.2],[0.25, 0.5]]
for order, edge_mesh_size, domain_mesh_size in zip(orders, edge_mesh_sizes, domain_mesh_sizes):
    data = {
        "lines": [
            # Region 1's outer boundary
            {"start": [0, 0], "end": [2, 0], "left_region": 1, "right_region": 0, "h": edge_mesh_size, "boundary_index": 1},
            {"start": [2, 0], "end": [2, 2], "left_region": 1, "right_region": 2, "h": edge_mesh_size, "boundary_index": 1},
            {"start": [2, 2], "end": [0, 2], "left_region": 1, "right_region": 0, "h": edge_mesh_size, "boundary_index": 1},
            {"start": [0, 2], "end": [0, 0], "left_region": 1, "right_region": 0, "h": edge_mesh_size, "boundary_index": 2},
            # Region 2's outer boundary
            {"start": [2, 0], "end": [4, 0], "left_region": 2, "right_region": 0, "h": edge_mesh_size, "boundary_index": 1},
            {"start": [4, 0], "end": [4, 2], "left_region": 2, "right_region": 0, "h": edge_mesh_size, "boundary_index": 1},
            {"start": [4, 2], "end": [2, 2], "left_region": 2, "right_region": 0, "h": edge_mesh_size, "boundary_index": 1},
        ],
        "regions": [
            {"region_id": 1, "mesh_inner": domain_mesh_size[0]},
            {"region_id": 2, "mesh_inner": domain_mesh_size[1]}
        ]
    }

    max_gradient = 0.07
    t = generate_mesh(data, max_gradient)

    def f(x,y):
        #return np.ones(x.shape)
        return np.sin(0.25*np.pi*x)*np.sin(0.5*np.pi*y)

    def g(x,y):
        return np.zeros(x.shape)

    space = H1Space(t,order)
    u, mass, f_vector = solve_bvp(0, 1, space, g, f)
    ax, mini, maxi = show_grid_function(u, space, vrange=[0.0, 0.32], dx=0.2, dy=0.2)
    show_boundary_function(g, t, ax)
    print(mini, maxi)
    ax.set_zlim([-5.0, 5.0])
    plt.show()