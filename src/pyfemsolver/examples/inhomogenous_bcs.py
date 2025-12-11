"""Example solving Laplacian BVP with inhomogeneous boundary conditions."""

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

def u_bnd(x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:  # pylint:disable=C0116
    return (x - 0.5) ** 3 + (y - 0.5) ** 3
g = VariableCoefficientFunction({1:u_bnd, 2:u_bnd, 3:u_bnd, 4:u_bnd})
f = ConstantCoefficientFunction(0)

orders = [1, 4]
edge_mesh_sizes = [0.4, 0.5]
domain_mesh_sizes = [0.4, 0.5]
for order, edge_mesh_size, domain_mesh_size in zip(orders, edge_mesh_sizes, domain_mesh_sizes):
    lines: List[Line] = []
    lines.append(Line(start=(-1, -1), end=(1, -1), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(1, -1), end=(1, 1), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=2))
    lines.append(Line(start=(1, 1), end=(-1, 1), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=3))
    lines.append(Line(start=(-1, 1), end=(-1, -1), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=4))
    regions: List[Region] = []
    regions.append(Region(region_id=1, mesh_inner=domain_mesh_size))
    geometry = Geometry(lines=lines, regions=regions)

    mesh = generate_mesh(geometry, max_gradient=0.07)
    space = H1Space(mesh, order)

    u, mass2, f_vector2 = solve_bvp(0, 1, space, g, f)
    ax, mini, maxi = show_grid_function(u, space, vrange=(-6.01, 0.01), dx=0.125, dy=0.125)
    show_boundary_function(u_bnd, mesh, ax)
    plt.show()  # type:ignore
