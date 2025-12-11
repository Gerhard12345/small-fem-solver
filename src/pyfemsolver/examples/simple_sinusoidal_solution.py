"""
Example solving Laplacian BVP with sinusoidal source term and homogeneous boundary conditions.
Can be used to compute an approximation error.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from ..solverlib.space import H1Space
from ..solverlib.solving import solve_bvp
from ..visual.visual import show_grid_function
from ..visual.visual import show_boundary_function
from ..solverlib.meshing import generate_mesh
from ..solverlib.geometry import Line, Region, Geometry
from ..solverlib.coefficientfunction import VariableCoefficientFunction, ConstantCoefficientFunction


def f1(x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:  # pylint:disable=C0116
    # return np.ones(x.shape)
    return np.sin(0.75 * np.pi * x) * np.sin(1.5 * np.pi * y)


def f2(x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:  # pylint:disable=C0116
    # return np.ones(x.shape)
    return -np.sin(0.25 * np.pi * x) * np.sin(1.5 * np.pi * y)


f_domain = VariableCoefficientFunction({1: f1, 2: f2})
u_bnd = ConstantCoefficientFunction(0)

orders = [1, 6]
edge_mesh_sizes = [0.25, 1.25]
domain_mesh_sizes = [[0.25, 0.25], [0.5, 1.0]]
plot_spacings = [0.25, 0.025]
for order, edge_mesh_size, domain_mesh_size, plot_spacing in zip(orders, edge_mesh_sizes, domain_mesh_sizes, plot_spacings):
    lines: List[Line] = []
    lines.append(Line(start=(0, 0), end=(2, 0), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(2, 0), end=(2, 2), left_region=1, right_region=2, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(2, 2), end=(0, 2), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(0, 2), end=(0, 0), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=2))
    lines.append(Line(start=(2, 0), end=(4, 0), left_region=2, right_region=0, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(4, 0), end=(4, 2), left_region=2, right_region=0, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(4, 2), end=(2, 2), left_region=2, right_region=0, h=edge_mesh_size, boundary_index=1))
    regions: List[Region] = []
    regions.append(Region(region_id=1, mesh_inner=domain_mesh_size[0]))
    regions.append(Region(region_id=2, mesh_inner=domain_mesh_size[1]))
    geometry = Geometry(lines=lines, regions=regions)

    mesh = generate_mesh(geometry, max_gradient=0.07)
    space = H1Space(mesh, order)

    u, mass, f_vector = solve_bvp(0, 1, space, u_bnd, f_domain)
    ax, mini, maxi = show_grid_function(u, space, vrange=(-0.05, 0.05), dx=plot_spacing, dy=plot_spacing)
    show_boundary_function(u_bnd, mesh, ax)
    print(mini, maxi)
    ax.set_zlim([-0.05, 0.05])  # type:ignore
plt.show()  # type:ignore
