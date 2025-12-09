import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyfemsolver.solverlib.space import H1Space
from pyfemsolver.solverlib.solving import solve_bvp
from pyfemsolver.visual.visual import show_grid_function
from pyfemsolver.visual.visual import show_boundary_function
from pyfemsolver.solverlib.meshing import generate_mesh


def f(x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
    # return np.ones(x.shape)
    return np.sin(0.75 * np.pi * x) * np.sin(1.5 * np.pi * y)


def u_bnd(x: NDArray[np.floating], _: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.zeros(x.shape)


orders = [1, 4]
edge_mesh_sizes = [0.25, 1.25]
domain_mesh_sizes = [[0.25, 0.25], [0.5, 1.0]]
plot_spacings = [0.25, 0.075]
for order, edge_mesh_size, domain_mesh_size, plot_spacing in zip(orders, edge_mesh_sizes, domain_mesh_sizes, plot_spacings):
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
        "regions": [{"region_id": 1, "mesh_inner": domain_mesh_size[0]}, {"region_id": 2, "mesh_inner": domain_mesh_size[1]}],
    }

    max_gradient = 0.07
    mesh = generate_mesh(data, max_gradient)

    space = H1Space(mesh, order)
    u, mass, f_vector = solve_bvp(0, 1, space, u_bnd, f)
    ax, mini, maxi = show_grid_function(u, space, vrange=(-0.035, 0.035), dx=plot_spacing, dy=plot_spacing)
    show_boundary_function(u_bnd, mesh, ax)
    print(mini, maxi)
    ax.set_zlim([-5.0, 5.0])  # type:ignore
    plt.show()  # type:ignore
