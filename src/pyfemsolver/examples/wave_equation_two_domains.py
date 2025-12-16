"""Demonstrates the ability to solve wave equations"""

from typing import List
import matplotlib.pyplot as plt

from ..solverlib.space import H1Space
from ..solverlib.solving import solve_bvp, set_boundary_values
from ..visual.visual import show_grid_function
from ..solverlib.meshing import generate_mesh
from ..solverlib.geometry import Line, Region, Geometry
from ..solverlib.coefficientfunction import DomainConstantCoefficientFunction, ConstantCoefficientFunction
from ..solverlib.forms import BilinearForm, LinearForm
from ..solverlib.integrators import Laplace, Mass

height = 2.0  # pylint:disable=C0103
width = 2.0  # pylint:disable=C0103
center_x = 0
center_y = 0
lines: List[Line] = []
lines.append(Line(start=(-10, -10), end=(10, -10), left_region=3, right_region=0, h=0.5, boundary_index=1))
lines.append(Line(start=(10, -10), end=(10, 10), left_region=3, right_region=0, h=0.5, boundary_index=1))
lines.append(Line(start=(10, 10), end=(-10, 10), left_region=3, right_region=0, h=0.5, boundary_index=1))
lines.append(Line(start=(-10, 10), end=(-10, -10), left_region=3, right_region=0, h=0.5, boundary_index=1))

lines.append(Line(start=(-8, -8), end=(8, -8), left_region=2, right_region=3, h=0.5, boundary_index=1))
lines.append(Line(start=(8, -8), end=(8, 8), left_region=2, right_region=3, h=0.5, boundary_index=1))
lines.append(Line(start=(8, 8), end=(-8, 8), left_region=2, right_region=3, h=0.5, boundary_index=1))
lines.append(Line(start=(-8, 8), end=(-8, -8), left_region=2, right_region=3, h=0.5, boundary_index=1))

lines.append(Line(start=(-4, -4), end=(4, -4), left_region=1, right_region=2, h=0.5, boundary_index=1))
lines.append(Line(start=(4, -4), end=(4, 4), left_region=1, right_region=2, h=0.5, boundary_index=1))
lines.append(Line(start=(4, 4), end=(-4, 4), left_region=1, right_region=2, h=0.5, boundary_index=1))
lines.append(Line(start=(-4, 4), end=(-4, -4), left_region=1, right_region=2, h=0.5, boundary_index=1))


# Plate 1
lines.append(
    Line(
        start=(center_x - width * 0.5, center_y - height * 0.5),
        end=(center_x + width * 0.5, center_y - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.5,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x + width * 0.5, center_y - height * 0.5),
        end=(center_x + width * 0.5, center_y + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.5,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x + width * 0.5, center_y + height * 0.5),
        end=(center_x - width * 0.5, center_y + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.5,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x - width * 0.5, center_y + height * 0.5),
        end=(center_x - width * 0.5, center_y - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.5,
        boundary_index=2,
    )
)

regions = [Region(region_id=1, mesh_inner=0.5), Region(region_id=2, mesh_inner=0.75), Region(region_id=3, mesh_inner=0.5)]
geometry = Geometry(lines=lines, regions=regions)

mesh = generate_mesh(geometry, max_gradient=0.05)
space = H1Space(mesh, 4, dirichlet_indices=[1, 2])

u_bound = DomainConstantCoefficientFunction(values={2: 1.0, 1: 0.0})

laplace = Laplace(coefficient=DomainConstantCoefficientFunction({1: 0.05, 2: 0.15, 3: 0.05}), space=space, is_boundary=False)
mass = Mass(coefficient=ConstantCoefficientFunction(-1), space=space, is_boundary=False)
bilinearform = BilinearForm([laplace, mass])
linearform = LinearForm([])

u = space.create_gridfunction()
set_boundary_values(dof_vector=u, space=space, g=u_bound)
solve_bvp(bilinearform=bilinearform, linearform=linearform, u=u, space=space)
ax, mini, maxi = show_grid_function(u, space, vrange=(-1.75, 1.8), n_subdivision=16)
plt.show()  # type:ignore
