"""Demonstrates the ability to solve wave equations"""

from typing import List
import matplotlib.pyplot as plt

from ..solverlib.space import H1Space
from ..solverlib.solving import solve_bvp
from ..visual.visual import show_grid_function
from ..solverlib.meshing import generate_mesh
from ..solverlib.geometry import Line, Region, Geometry
from ..solverlib.coefficientfunction import DomainConstantCoefficientFunction, ConstantCoefficientFunction


height = 0.6  # pylint:disable=C0103
width = 2.4  # pylint:disable=C0103
center_x = [0, 0]
center_y = [-2, 2]
lines: List[Line] = []
lines.append(Line(start=(-6, -6), end=(6, -6), left_region=1, right_region=0, h=0.5, boundary_index=1))
lines.append(Line(start=(6, -6), end=(6, 6), left_region=1, right_region=0, h=0.5, boundary_index=1))
lines.append(Line(start=(6, 6), end=(-6, 6), left_region=1, right_region=0, h=0.5, boundary_index=1))
lines.append(Line(start=(-6, 6), end=(-6, -6), left_region=1, right_region=0, h=0.5, boundary_index=1))
# Plate 1
lines.append(
    Line(
        start=(center_x[0] - width * 0.5, center_y[0] - height * 0.5),
        end=(center_x[0] + width * 0.5, center_y[0] - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.2,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] + width * 0.5, center_y[0] - height * 0.5),
        end=(center_x[0] + width * 0.5, center_y[0] + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.2,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] + width * 0.5, center_y[0] + height * 0.5),
        end=(center_x[0] - width * 0.5, center_y[0] + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.2,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] - width * 0.5, center_y[0] + height * 0.5),
        end=(center_x[0] - width * 0.5, center_y[0] - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.2,
        boundary_index=2,
    )
)
# Plate 2
lines.append(
    Line(
        start=(center_x[1] - width * 0.5, center_y[1] - height * 0.5),
        end=(center_x[1] + width * 0.5, center_y[1] - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.2,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] + width * 0.5, center_y[1] - height * 0.5),
        end=(center_x[1] + width * 0.5, center_y[1] + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.2,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] + width * 0.5, center_y[1] + height * 0.5),
        end=(center_x[1] - width * 0.5, center_y[1] + height * 0.5),
        left_region=0,
        right_region=1,
        h=0.2,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] - width * 0.5, center_y[1] + height * 0.5),
        end=(center_x[1] - width * 0.5, center_y[1] - height * 0.5),
        left_region=0,
        right_region=1,
        h=0.2,
        boundary_index=3,
    )
)
regions = [Region(region_id=1, mesh_inner=0.5)]
geometry = Geometry(lines=lines, regions=regions)

mesh = generate_mesh(geometry, max_gradient=0.07)
space = H1Space(mesh, 3)

u_bound = DomainConstantCoefficientFunction(values={3: 0.1, 2: -0.1, 1: 0.0})
f = ConstantCoefficientFunction(value=0)

u1, mass, f_vector = solve_bvp(-1, 0.6125 * 0.125, space, u_bound, f)
ax, mini, maxi = show_grid_function(u1, space, vrange=(-0.65, 0.65), dx=0.125, dy=0.125)
plt.show()  # type:ignore
