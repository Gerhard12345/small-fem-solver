"""
Example solving Laplacian BVP with sinusoidal source term and homogeneous boundary conditions.
Can be used to compute an approximation error.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from ..solverlib.space import H1Space
from ..solverlib.solving import solve_bvp
from ..visual.visual import show_grid_function
from ..solverlib.meshing import generate_mesh
from ..solverlib.geometry import Line, Region, Geometry
from ..solverlib.coefficientfunction import VariableCoefficientFunction, ConstantCoefficientFunction
from ..solverlib.forms import BilinearForm, LinearForm
from ..solverlib.integrators import Laplace, Source


def f1(x: float, y: float) -> float:  # pylint:disable=C0116
    return np.sin(0.75 * np.pi * x) * np.cos(1.5 * np.pi * y)


def f2(x: float, y: float) -> float:  # pylint:disable=C0116
    return -np.sin(0.25 * np.pi * x) * np.cos(1.5 * np.pi * y)


f_domain = VariableCoefficientFunction({1: f1, 2: f2}, f_shape=(1, 1))
u_bnd = ConstantCoefficientFunction(0)

orders = [6, 6]
edge_mesh_sizes = [1.25, 1.25]
domain_mesh_sizes = [[1.0, 1.0], [1.0, 1.0]]
plot_spacings = [0.05, 0.05]
all_dirichlet_indices = [[2, 3], [1]]
plot_ranges = [(-0.053, 0.053), (-0.042, 0.042)]
for order, edge_mesh_size, domain_mesh_size, plot_spacing, dirichlet_indices, plot_range in zip(
    orders, edge_mesh_sizes, domain_mesh_sizes, plot_spacings, all_dirichlet_indices, plot_ranges
):
    lines: List[Line] = []
    lines.append(Line(start=(0, 0), end=(2, 0), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(2, 0), end=(2, 2), left_region=1, right_region=2, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(2, 2), end=(0, 2), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(0, 2), end=(0, 0), left_region=1, right_region=0, h=edge_mesh_size, boundary_index=2))
    lines.append(Line(start=(2, 0), end=(4, 0), left_region=2, right_region=0, h=edge_mesh_size, boundary_index=1))
    lines.append(Line(start=(4, 0), end=(4, 2), left_region=2, right_region=0, h=edge_mesh_size, boundary_index=3))
    lines.append(Line(start=(4, 2), end=(2, 2), left_region=2, right_region=0, h=edge_mesh_size, boundary_index=1))
    regions: List[Region] = []
    regions.append(Region(region_id=1, mesh_inner=domain_mesh_size[0]))
    regions.append(Region(region_id=2, mesh_inner=domain_mesh_size[1]))
    geometry = Geometry(lines=lines, regions=regions)

    mesh = generate_mesh(geometry, max_gradient=0.07)
    space = H1Space(mesh, order, dirichlet_indices=dirichlet_indices)

    source = Source(coefficient=f_domain, space=space, is_boundary=False)
    linearform = LinearForm([source])

    laplace = Laplace(coefficient=ConstantCoefficientFunction(1), space=space, is_boundary=False)
    bilinearform = BilinearForm([laplace])

    u = np.zeros((space.ndof, 1))
    solve_bvp(bilinearform=bilinearform, linearform=linearform, u=u, space=space)
    ax, mini, maxi = show_grid_function(u, space, vrange=plot_range, dx=plot_spacing, dy=plot_spacing)
    print(mini, maxi)
plt.show()  # type:ignore
