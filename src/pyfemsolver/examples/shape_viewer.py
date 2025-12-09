"""Visualization of shape functions on a mesh."""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from ..solverlib.space import H1Space
from ..visual.visual import show_shape, show_edge_shape
from ..solverlib.meshing import generate_mesh
from ..solverlib.geometry import Geometry, Line, Region


lines: List[Line] = []

lines.append(Line(start=(-6, -6), end=(6, -6), left_region=1, right_region=0, h=4, boundary_index=1))
lines.append(Line(start=(6, -6), end=(6, 6), left_region=1, right_region=0, h=4, boundary_index=1))
lines.append(Line(start=(6, 6), end=(-6, 6), left_region=1, right_region=0, h=4, boundary_index=1))
lines.append(Line(start=(-6, 6), end=(-6, -6), left_region=1, right_region=0, h=4, boundary_index=1))

regions: List[Region] = []
regions.append(Region(region_id=1, mesh_inner=4))
geometry = Geometry(lines=lines, regions=regions)


t = generate_mesh(geometry, max_gradient=0.4)


space = H1Space(t, 2)
u = np.zeros((space.ndof, 1))
dof = 3  # pylint:disable=C0103
ax, mini, maxi = show_shape(dof, space, vrange=(0, 1), dx=0.2, dy=0.2)
show_edge_shape(15, space)
ax.set_zlim([0, 1])  # type:ignore
plt.show()  # type:ignore
