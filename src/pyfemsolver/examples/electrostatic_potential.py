import matplotlib.pyplot as plt
import numpy as np
from pyfemsolver.solverlib.space import H1Space
from pyfemsolver.solverlib.solving import solve_bvp
from pyfemsolver.visual.visual import show_grid_function
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
        # plate 1
        {
            "start": [center_x[0] - width * 0.5, center_y[0] - height * 0.5],
            "end": [center_x[0] + width * 0.5, center_y[0] - height * 0.5],
            "left_region": 0,
            "right_region": 1,
            "h": 1,
            "boundary_index": 3,
        },
        {
            "start": [center_x[0] + width * 0.5, center_y[0] - height * 0.5],
            "end": [center_x[0] + width * 0.5, center_y[0] + height * 0.5],
            "left_region": 0,
            "right_region": 1,
            "h": 1,
            "boundary_index": 3,
        },
        {
            "start": [center_x[0] + width * 0.5, center_y[0] + height * 0.5],
            "end": [center_x[0] - width * 0.5, center_y[0] + height * 0.5],
            "left_region": 0,
            "right_region": 1,
            "h": 1,
            "boundary_index": 3,
        },
        {
            "start": [center_x[0] - width * 0.5, center_y[0] + height * 0.5],
            "end": [center_x[0] - width * 0.5, center_y[0] - height * 0.5],
            "left_region": 0,
            "right_region": 1,
            "h": 1,
            "boundary_index": 3,
        },
        # plate 2
        {
            "start": [center_x[1] - width * 0.5, center_y[1] - height * 0.5],
            "end": [center_x[1] + width * 0.5, center_y[1] - height * 0.5],
            "left_region": 0,
            "right_region": 1,
            "h": 1,
            "boundary_index": 3,
        },
        {
            "start": [center_x[1] + width * 0.5, center_y[1] - height * 0.5],
            "end": [center_x[1] + width * 0.5, center_y[1] + height * 0.5],
            "left_region": 0,
            "right_region": 1,
            "h": 1,
            "boundary_index": 3,
        },
        {
            "start": [center_x[1] + width * 0.5, center_y[1] + height * 0.5],
            "end": [center_x[1] - width * 0.5, center_y[1] + height * 0.5],
            "left_region": 0,
            "right_region": 1,
            "h": 1,
            "boundary_index": 3,
        },
        {
            "start": [center_x[1] - width * 0.5, center_y[1] + height * 0.5],
            "end": [center_x[1] - width * 0.5, center_y[1] - height * 0.5],
            "left_region": 0,
            "right_region": 1,
            "h": 1,
            "boundary_index": 3,
        },
    ],
    "regions": [
        {"region_id": 1, "mesh_inner": 4},
    ],
}

max_gradient = 0.4
t = generate_mesh(data, max_gradient)


space = H1Space(t, 9)

safety = 0.01


def u_bound(x, y):
    if type(x) == np.float64:
        x = np.array([x])
        y = np.array([y])
    vals = np.zeros(x.shape)
    for i, point in enumerate(zip(x, y)):
        if np.abs(point[0] - center_x[0]) < width / 2 + safety and np.abs(point[1] - center_y[0]) < height / 2 + safety:
            vals[i] = -100
        elif np.abs(point[0] - center_x[1]) < width / 2 + safety and np.abs(point[1] - center_y[1]) < height / 2 + safety:
            vals[i] = 100
        else:
            vals[i] = 0
    return vals


u1, mass1, f_vector1 = solve_bvp(0, 1, space, u_bound, lambda x, y: np.zeros(x.shape))
ax, mini, maxi = show_grid_function(u1, space, vrange=[-100, 100], dx=0.05, dy=0.05)
ax.set_zlim([-100, 100])
plt.show()
