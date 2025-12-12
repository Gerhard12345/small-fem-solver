"""Example solving Laplacian BVP with inhomogeneous boundary conditions."""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from ..solverlib.space import H1Space
from ..solverlib.solving import solve_bvp, set_boundary_values
from ..visual.visual import show_grid_function
from ..solverlib.meshing import generate_mesh
from ..solverlib.geometry import Line, Region, Geometry
from ..solverlib.coefficientfunction import VariableCoefficientFunction, ConstantCoefficientFunction
from ..solverlib.forms import BilinearForm, LinearForm
from ..solverlib.integrators import Laplace


def u_bnd(x: float, y: float) -> float:  # pylint:disable=C0116
    return (x - 0.5) ** 3 + (y - 0.5) ** 3


g = VariableCoefficientFunction({1: u_bnd, 2: u_bnd, 3: u_bnd, 4: u_bnd}, f_shape=(1, 1))
f = ConstantCoefficientFunction(0)
f_mass = ConstantCoefficientFunction(0)


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

    laplace = Laplace(ConstantCoefficientFunction(1), space, is_boundary=False)
    bilinearform = BilinearForm([laplace])
    linearform = LinearForm([])

    # set boundary values
    u = np.zeros((space.ndof, 1))
    set_boundary_values(u, space, g)

    solve_bvp(bilinearform, linearform, u, space)
    ax, mini, maxi = show_grid_function(u, space, vrange=(-6.75, 0.25), dx=0.125, dy=0.125)
    plt.show()  # type:ignore
