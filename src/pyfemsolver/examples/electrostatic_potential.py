"""Example solving for the electrostatic potential with two embedded charged plates."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..solverlib.space import H1Space
from ..solverlib.solving import solve_bvp
from ..visual.visual import show_grid_function, show_gradient_of_grid_function
from ..solverlib.meshing import generate_mesh
from ..solverlib.geometry import Line, Region, Geometry
from ..solverlib.coefficientfunction import ConstantCoefficientFunction, DomainConstantCoefficientFunction

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
        h=1,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] + width * 0.5, center_y[0] - height * 0.5),
        end=(center_x[0] + width * 0.5, center_y[0] + height * 0.5),
        left_region=0,
        right_region=1,
        h=1,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] + width * 0.5, center_y[0] + height * 0.5),
        end=(center_x[0] - width * 0.5, center_y[0] + height * 0.5),
        left_region=0,
        right_region=1,
        h=1,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[0] - width * 0.5, center_y[0] + height * 0.5),
        end=(center_x[0] - width * 0.5, center_y[0] - height * 0.5),
        left_region=0,
        right_region=1,
        h=1,
        boundary_index=2,
    )
)
lines.append(
    Line(
        start=(center_x[1] - width * 0.5, center_y[1] - height * 0.5),
        end=(center_x[1] + width * 0.5, center_y[1] - height * 0.5),
        left_region=0,
        right_region=1,
        h=1,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] + width * 0.5, center_y[1] - height * 0.5),
        end=(center_x[1] + width * 0.5, center_y[1] + height * 0.5),
        left_region=0,
        right_region=1,
        h=1,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] + width * 0.5, center_y[1] + height * 0.5),
        end=(center_x[1] - width * 0.5, center_y[1] + height * 0.5),
        left_region=0,
        right_region=1,
        h=1,
        boundary_index=3,
    )
)
lines.append(
    Line(
        start=(center_x[1] - width * 0.5, center_y[1] + height * 0.5),
        end=(center_x[1] - width * 0.5, center_y[1] - height * 0.5),
        left_region=0,
        right_region=1,
        h=1,
        boundary_index=3,
    )
)
regions: List[Region] = []
regions.append(Region(region_id=1, mesh_inner=4))
geometry = Geometry(lines=lines, regions=regions)

mesh = generate_mesh(geometry, max_gradient=0.4)
space = H1Space(mesh, 9)


def u_bound(x: NDArray[np.floating] | np.floating, y: NDArray[np.floating] | np.floating) -> NDArray[np.floating]:  # pylint:disable=C0116
    safety = 0.01
    if isinstance(x, np.floating):
        x = np.array([x])
        y = np.array([y])
    vals = np.zeros(x.shape)
    for i, point in enumerate(zip(x.flatten(), y.flatten())):
        if np.abs(point[0] - center_x[0]) < width / 2 + safety and np.abs(point[1] - center_y[0]) < height / 2 + safety:
            vals[i] = -100
        elif np.abs(point[0] - center_x[1]) < width / 2 + safety and np.abs(point[1] - center_y[1]) < height / 2 + safety:
            vals[i] = 100
        else:
            vals[i] = 0
    return vals


u_bound = DomainConstantCoefficientFunction(values={3: 100, 2: -100, 1: 0.0})
f = ConstantCoefficientFunction(0)

u1, mass1, f_vector1 = solve_bvp(0, 1, space, u_bound, f)
ax, mini, maxi = show_grid_function(u1, space, vrange=(-100, 100), dx=0.05, dy=0.05)
ax.set_zlim([-100, 100])  # type:ignore
ax_x, ax_y, mini, maxi = show_gradient_of_grid_function(u1, space, vrange=(-100, 100), dx=0.1, dy=0.1)
ax_x.set_zlim([-500, 500])  # type:ignore
ax_y.set_zlim([-500, 500])  # type:ignore
plt.show()  # type:ignore
