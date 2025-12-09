import matplotlib.pyplot as plt
import numpy as np
from pyfemsolver.solverlib.space import H1Space
from pyfemsolver.visual.visual import show_shape, show_edge_shape
from pyfemsolver.solverlib.meshing import generate_mesh


height = 0.6
width = 2.4
center_x = [0, 0]
center_y = [-2, 2]


data = {
    "lines": [
        # Region 1's outer boundary
        {"start": [-6, -6], "end": [6, -6], "left_region": 1, "right_region": 0, "h": 4, "boundary_index": 1},
        {"start": [6, -6], "end": [6, 6], "left_region": 1, "right_region": 0, "h": 4, "boundary_index": 1},
        {"start": [6, 6], "end": [-6, 6], "left_region": 1, "right_region": 0, "h": 4, "boundary_index": 1},
        {"start": [-6, 6], "end": [-6, -6], "left_region": 1, "right_region": 0, "h": 4, "boundary_index": 1},
    ],
    "regions": [
        {"region_id": 1, "mesh_inner": 4},
    ],
}

max_gradient = 0.4
t = generate_mesh(data, max_gradient)


space = H1Space(t, 2)

safety = 0.01

u = np.zeros((space.ndof, 1))
dof = 3
ax, mini, maxi = show_shape(dof, space, vrange=(0, 1), dx=0.2, dy=0.2)
show_edge_shape(15, space)
ax.set_zlim([0, 1])  # type:ignore
plt.show()  # type:ignore
