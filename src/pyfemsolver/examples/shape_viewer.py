"""Visualization of shape functions on a mesh."""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from ..solverlib.space import H1Space
from ..visual.visual import show_shape, show_edge_shape, show_mesh
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


space = H1Space(t, 3, dirichlet_indices=[])
u = np.zeros((space.ndof, 1))
dof = 38  # pylint:disable=C0103
ax, mini, maxi = show_shape(dof, space, vrange=(-0.192, 0.192), n_subdivision=80)
show_mesh(tri=t, ax=ax)
show_edge_shape(15, space)
plt.show()  # type:ignore
