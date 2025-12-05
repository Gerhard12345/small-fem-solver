import matplotlib.pyplot as plt
import numpy as np
from space import H1Space
from solving import solve_bvp
from visual import show_grid_function
from meshing import generate_mesh


height = 0.6
width = 2.4
center_x = [0,0]
center_y = [-2,2]


data = {
    "lines": [
        # Region 1's outer boundary
        {"start": [-6, -6], "end": [6, -6], "left_region": 1, "right_region": 0, "h": 0.5, "boundary_index": 1},
        {"start": [6, -6], "end": [6, 6], "left_region": 1, "right_region": 0, "h": 0.5, "boundary_index": 1},
        {"start": [6, 6], "end": [-6, 6], "left_region": 1, "right_region": 0, "h": 0.5, "boundary_index": 1},
        {"start": [-6, 6], "end": [-6, -6], "left_region": 1, "right_region": 0, "h": 0.5, "boundary_index": 1},
        # plate 1
        {"start": [center_x[0]-width*0.5, center_y[0]-height*0.5], "end": [center_x[0]+width*0.5, center_y[0]-height*0.5], "left_region": 0, "right_region": 1, "h": 0.2, "boundary_index": 3},
        {"start": [center_x[0]+width*0.5, center_y[0]-height*0.5], "end": [center_x[0]+width*0.5, center_y[0]+height*0.5], "left_region": 0, "right_region": 1, "h": 0.2, "boundary_index": 3},
        {"start": [center_x[0]+width*0.5, center_y[0]+height*0.5], "end": [center_x[0]-width*0.5, center_y[0]+height*0.5], "left_region": 0, "right_region": 1, "h": 0.2, "boundary_index": 3},
        {"start": [center_x[0]-width*0.5, center_y[0]+height*0.5], "end": [center_x[0]-width*0.5, center_y[0]-height*0.5], "left_region": 0, "right_region": 1, "h": 0.2, "boundary_index": 3},
        # plate 2
        {"start": [center_x[1]-width*0.5, center_y[1]-height*0.5], "end": [center_x[1]+width*0.5, center_y[1]-height*0.5], "left_region": 0, "right_region": 1, "h": 0.2, "boundary_index": 3},
        {"start": [center_x[1]+width*0.5, center_y[1]-height*0.5], "end": [center_x[1]+width*0.5, center_y[1]+height*0.5], "left_region": 0, "right_region": 1, "h": 0.2, "boundary_index": 3},
        {"start": [center_x[1]+width*0.5, center_y[1]+height*0.5], "end": [center_x[1]-width*0.5, center_y[1]+height*0.5], "left_region": 0, "right_region": 1, "h": 0.2, "boundary_index": 3},
        {"start": [center_x[1]-width*0.5, center_y[1]+height*0.5], "end": [center_x[1]-width*0.5, center_y[1]-height*0.5], "left_region": 0, "right_region": 1, "h": 0.2, "boundary_index": 3},
    ],
    "regions": [
        {"region_id": 1, "mesh_inner": 0.5},
    ]
}

max_gradient = 0.07
t = generate_mesh(data, max_gradient)


space = H1Space(t, 2)

safety=0.01

def g_bound(x,y):
    if type(x) == np.float64:
        x = np.array([x])
        y = np.array([y])
    vals = np.zeros(x.shape)
    for i,point in enumerate(zip(x,y)):
        if (np.abs(point[0]-center_x[0])<width/2+safety and np.abs(point[1]-center_y[0])<height/2+safety):
            vals[i] = -0.1
        elif (np.abs(point[0]-center_x[1])<width/2+safety and np.abs(point[1]-center_y[1])<height/2+safety):
            vals[i] = 0.1
        else:
            vals[i] = 0
    return vals

# u0, mass1, f_vector1 = solve_bvp(10,1,space2,g_bound,lambda x,y:np.zeros(x.shape))
u1, mass1, f_vector1 = solve_bvp(0,1,space,g_bound,lambda x,y:np.zeros(x.shape))
ax, mini, maxi = show_grid_function(u1,space,vrange=[-0.1,0.1],dx=0.5,dy=0.5)
ax.set_zlim([-5.0,5.0])
plt.show()