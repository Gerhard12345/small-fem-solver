"""Example solving for the electrostatic potential with two embedded charged plates."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..solverlib.space import H1Space
from ..solverlib.solving import solve_bvp, set_boundary_values
from ..visual.visual import show_grid_function, show_gradient_of_grid_function
from ..solverlib.meshing import generate_mesh
from ..solverlib.geometry import Line, Region, Geometry
from ..solverlib.coefficientfunction import ConstantCoefficientFunction, DomainConstantCoefficientFunction
from ..solverlib.forms import BilinearForm, LinearForm
from ..solverlib.integrators import Laplace


height = 0.6  # pylint:disable=C0103
width = 2.4  # pylint:disable=C0103
center_x = [0, 0]
center_y = [-2, 2]

lines: List[Line] = []
lines.append(Line(start=(-6, -6), end=(6, -6), left_region=1, right_region=0, h=4, boundary_index=1))
lines.append(Line(start=(6, -6), end=(6, 6), left_region=1, right_region=0, h=4, boundary_index=1))
lines.append(Line(start=(6, 6), end=(-6, 6), left_region=1, right_region=0, h=4, boundary_index=1))
lines.append(Line(start=(-6, 6), end=(-6, -6), left_region=1, right_region=0, h=4, boundary_index=1))
lines.append(
    Line(
        start=(center_x[0] - width * 0.5, center_y[0] - height * 0.5),
        end=(center_x[0] + width * 0.5, center_y[0] - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.25,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] + width * 0.5, center_y[0] - height * 0.5),
        end=(center_x[0] + width * 0.5, center_y[0] + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.25,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] + width * 0.5, center_y[0] + height * 0.5),
        end=(center_x[0] - width * 0.5, center_y[0] + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.25,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] - width * 0.5, center_y[0] + height * 0.5),
        end=(center_x[0] - width * 0.5, center_y[0] - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.25,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[1] - width * 0.5, center_y[1] - height * 0.5),
        end=(center_x[1] + width * 0.5, center_y[1] - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.25,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] + width * 0.5, center_y[1] - height * 0.5),
        end=(center_x[1] + width * 0.5, center_y[1] + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.25,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] + width * 0.5, center_y[1] + height * 0.5),
        end=(center_x[1] - width * 0.5, center_y[1] + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.25,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] - width * 0.5, center_y[1] + height * 0.5),
        end=(center_x[1] - width * 0.5, center_y[1] - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.25,
        boundary_index=3,
    )
)
regions: List[Region] = []
regions.append(Region(region_id=1, mesh_inner=2))
geometry = Geometry(lines=lines, regions=regions)

mesh = generate_mesh(geometry, max_gradient=0.4)
u_bound = DomainConstantCoefficientFunction(values={3: 100, 2: -100, 1: 0.0})
for dirichlet_indices in ([1, 2, 3], [2, 3]):
    space = H1Space(mesh, 4, dirichlet_indices=dirichlet_indices)

    laplace = Laplace(coefficient=ConstantCoefficientFunction(1), space=space, is_boundary=False)
    bilinearform = BilinearForm([laplace])

    u = np.zeros((space.ndof, 1))
    set_boundary_values(dof_vector=u, space=space, g=u_bound)

    solve_bvp(bilinearform=bilinearform, linearform=LinearForm([]), u=u, space=space)
    ax, mini, maxi = show_grid_function(u, space, vrange=(-100, 100), n_subdivision=40)
    ax_x, ax_y, mini, maxi = show_gradient_of_grid_function(u, space, vrange=(-100, 100), n_subdivision=40)
plt.show()  # type:ignore
