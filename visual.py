import numpy as np
import matplotlib.pyplot as plt
from meshing import Triangulation
from element import barycentric_coordinates, barycentric_coordinates_line, duffy
from space import H1Space


def show_grid_function(u, space:H1Space, vrange, dx = 0.01, dy=0.01):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    trigs = [trig.points for trig in space.tri.trigs]
    x_coords = [point.coordinates[0] for point in space.tri.points]
    y_coords = [point.coordinates[1] for point in space.tri.points]
    ax.triplot(x_coords, y_coords, trigs)
    #ax.plot(tri.points[:, 0], tri.points[:, 1], "o")
    min_val = 1e16
    max_val = -1e16
    for i, trig in enumerate(space.tri.trigs):
        x = np.arange(-1, 1 + dx, dx)
        y = np.arange(-1, 1 + dy, dy)
        X, Y = np.meshgrid(x, y)
        X_t, Y_t = duffy(X, Y)
        s = barycentric_coordinates(X_t.flatten(), Y_t.flatten())
        A = np.array(space.tri.points[trig.points[0]].coordinates)
        A.shape = (2, 1)
        B = np.array(space.tri.points[trig.points[1]].coordinates)
        B.shape = (2, 1)
        C = np.array(space.tri.points[trig.points[2]].coordinates)
        C.shape = (2, 1)
        trig_nodes = A * s[0, :] + B * s[1, :] + C * s[2, :]
        fel = space.elements[i]
        shape = fel.shape_functions(X_t.flatten(), Y_t.flatten())
        values = np.matrix(shape.T) * u[space.dofs[i]]
        min_val = np.min([min_val, np.min(values)])
        max_val = np.max([max_val, np.max(values)])
        ax.plot_surface(
            trig_nodes[0, :].reshape(len(x), len(y)),
            trig_nodes[1, :].reshape(len(x), len(y)),
            values.reshape(len(x), len(y)),
            cmap="jet",
            linestyle="None",
            vmin=vrange[0],
            vmax=vrange[1],
        )
    return ax, min_val, max_val


def show_shape(dof, tri:Triangulation, space:H1Space, vrange=[0, 2]):
    u = np.zeros((space.ndof, 1))
    u[dof, 0] = 1
    ax, mini, maxi = show_grid_function(u, tri, space, vrange)
    print(f"Minimum value of shape function = {mini}, maximum value of shape function = {maxi}")
    ax.set_title(f"dof = {dof}")


def show_edge_shape(trig, space:H1Space):
    t = np.arange(-1, 1.025, 0.025)
    t.shape = (t.shape[0], 1)
    x, y = barycentric_coordinates_line(t)
    fig = plt.figure()
    for i, edge in enumerate(space.trig_edges[trig]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        shape = space.elements[trig].edge_shape_functions(t.flatten())
        print("edge ", i, "is_flipped: ", space.elements[trig].flipped_edge[i])
        print(edge in space.boundary_edges)
        xy = (
            space.tri.points[space.tri.edges[edge][0], :] * x
            + space.tri.points[space.tri.edges[edge][1], :] * y
        )
        for j in shape:
            ax.plot(xy[:, 0], xy[:, 1], j)
        ax.legend(space.edge_dofs[trig][i])
        ax.triplot(space.tri.points[:, 0], space.tri.points[:, 1], space.tri.simplices)
        ax.plot(space.tri.points[:, 0], space.tri.points[:, 1], "o")


def show_boundary_function(g, tri:Triangulation, ax):
    t = np.arange(-1, 1.025, 0.025)
    t.shape = (t.shape[0], 1)
    x, y = barycentric_coordinates_line(t)
    for edge in tri.boundary_edges:
        xy = (
            np.array(tri.points[tri.edges[edge.global_edge_nr].points[0]].coordinates) * x + np.array(tri.points[tri.edges[edge.global_edge_nr].points[1]].coordinates) * y
        )
        vals = g(xy[:, 0], xy[:, 1])
        ax.plot(xy[:, 0], xy[:, 1], vals, linewidth=7)

