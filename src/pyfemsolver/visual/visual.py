"""Simple visualization routines for grid functions and shape functions."""

from typing import Tuple, Callable

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore

from ..solverlib.meshing import Triangulation
from ..solverlib.integrationrules import duffy
from ..solverlib.element import barycentric_coordinates, barycentric_coordinates_line
from ..solverlib.space import H1Space
from ..solverlib.elementtransformation import ElementTransformationTrig


def show_grid_function(
    u: NDArray[np.floating], space: H1Space, vrange: Tuple[float, float], dx: float = 0.01, dy: float = 0.01
) -> Tuple[Axes3D, float, float]:
    """
    Display a grid function as a 3D surface plot.

    :param u: Coefficient vector for the grid function
    :type u: NDArray[np.floating]
    :param space: H1 space instance
    :type space: H1Space
    :param vrange: (min_value, max_value) for color scaling
    :type vrange: Tuple[float, float]
    :param dx: Grid spacing in x-direction (default 0.01)
    :type dx: float
    :param dy: Grid spacing in y-direction (default 0.01)
    :type dy: float
    :return: Tuple of Axes object minimum value, maximum value
    :rtype: Tuple[plt.Axes, float, float]
    """
    fig = plt.figure()  # type: ignore
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    trigs = [trig.points for trig in space.tri.trigs]
    x_coords = [point.coordinates[0] for point in space.tri.points]
    y_coords = [point.coordinates[1] for point in space.tri.points]
    ax.triplot(x_coords, y_coords, trigs)  # type: ignore
    ax.plot(x_coords, y_coords, "o")  # type: ignore
    min_val: float = 1e16
    max_val: float = -1e16
    for i, trig in enumerate(space.tri.trigs):
        x = np.arange(-1, 1 + dx, dx)
        y = np.arange(-1, 1 + dy, dy)
        X, Y = np.meshgrid(x, y)
        X_t, Y_t = duffy(X, Y)
        s = barycentric_coordinates(X_t.flatten(), Y_t.flatten())
        node_0 = np.array(space.tri.points[trig.points[0]].coordinates)
        node_0.shape = (2, 1)
        node_1 = np.array(space.tri.points[trig.points[1]].coordinates)
        node_1.shape = (2, 1)
        node_2 = np.array(space.tri.points[trig.points[2]].coordinates)
        node_2.shape = (2, 1)
        trig_nodes = node_0 * s[0, :] + node_1 * s[1, :] + node_2 * s[2, :]
        fel = space.elements[i]
        shape = fel.shape_functions(X_t.flatten(), Y_t.flatten())
        values = np.matrix(shape.T) * u[space.dofs[i]]
        min_val = np.min([min_val, np.min(values)])
        max_val = np.max([max_val, np.max(values)])
        ax.plot_surface(  # type: ignore
            trig_nodes[0, :].reshape(len(x), len(y)),
            trig_nodes[1, :].reshape(len(x), len(y)),
            values.reshape(len(x), len(y)),
            cmap="jet",
            linestyle="None",
            vmin=vrange[0],
            vmax=vrange[1],
        )
    return ax, min_val, max_val


def show_shape(dof_number: int, space: H1Space, vrange: Tuple[float, float], dx: float = 0.3, dy: float = 0.3) -> Tuple[Axes3D, float, float]:
    """
    Display the shape function corresponding to a given degree of freedom number.

    :param dof_number: Degree of freedom number
    :type dof_number: int
    :param space: H1 space instance
    :type space: H1Space
    :param vrange: (min_value, max_value) for color scaling
    :type vrange: Tuple[float, float]
    :param dx: Grid spacing in x-direction (default 0.3)
    :type dx: float
    :param dy: Grid spacing in y-direction (default 0.3)
    :type dy: float
    :return: Tuple of Axes object, minimum value, maximum value
    """
    u = np.zeros((space.ndof, 1))
    u[dof_number, 0] = 1
    ax, mini, maxi = show_grid_function(u, space, vrange, dx, dy)
    print(f"Minimum value of shape function = {mini}, maximum value of shape function = {maxi}")
    ax.set_title(f"dof number = {dof_number}")  # type: ignore
    return ax, mini, maxi


def show_edge_shape(trig_number: int, space: H1Space, ax: Axes3D | None = None):
    """
    Display the edge shape functions of a given triangle element.

    :param trig_number: Triangle element number
    :type trig_number: int
    :param space: H1 space instance
    :type space: H1Space
    :param ax: Optional Axes3D object to plot on. If None, a
                new figure with subplots will be created.
    :type ax: plt.Axes | None
    :return: None
    """
    t = np.arange(-1, 1.025, 0.025)
    t.shape = (t.shape[0], 1)
    x, y = barycentric_coordinates_line(t)
    use_new_axes = False
    fig = None
    if not ax:
        use_new_axes = True
        fig = plt.figure()  # type: ignore
    trig = space.tri.trigs[trig_number]
    for i, edge_nr in enumerate(trig.edges):
        if use_new_axes:
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")  # type: ignore
        shape = space.elements[trig_number].edge_shape_functions(t.flatten())
        edge = space.tri.edges[edge_nr]
        xy = space.tri.points[edge.points[0]].coordinates * x + space.tri.points[edge.points[1]].coordinates * y
        for j in shape:
            ax.plot(xy[:, 0], xy[:, 1], j)  # type: ignore
        trigs = [trig.points for trig in space.tri.trigs]
        x_coords = [point.coordinates[0] for point in space.tri.points]
        y_coords = [point.coordinates[1] for point in space.tri.points]
        ax.triplot(x_coords, y_coords, trigs)  # type: ignore
        ax.plot(x_coords, y_coords, "o")  # type: ignore


def show_boundary_function(g: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]], tri: Triangulation, ax: Axes3D):
    """
    Display a boundary function on the boundary edges of a triangulation.

    :param g: Boundary function to display
    :type g: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
    :param tri: Triangulation instance
    :type tri: Triangulation
    :param ax: Axes3D object to plot on
    :type ax: Axes3D
    :return: None
    """
    t = np.arange(-1, 1.025, 0.025)
    t.shape = (t.shape[0], 1)
    x, y = barycentric_coordinates_line(t)
    for edge in tri.boundary_edges:
        xy = (
            np.array(tri.points[tri.edges[edge.global_edge_nr].points[0]].coordinates) * x
            + np.array(tri.points[tri.edges[edge.global_edge_nr].points[1]].coordinates) * y
        )
        vals = g(xy[:, 0], xy[:, 1])
        ax.plot(xy[:, 0], xy[:, 1], vals, linewidth=7)  # type: ignore


def show_gradient_of_grid_function(u: NDArray[np.floating], space: H1Space, vrange: Tuple[int, int], dx: float = 0.01, dy: float = 0.01):
    """Display x and y components of the gradient of a grid function as surfaces.

    Creates a figure with two 3D surface plots side-by-side:
    1) Left plot: x-component of the gradient
    2) Right plot: y-component of the gradient

    :param u: Coefficient vector for the grid function
    :type u: NDArray[np.floating]
    :param space: H1 space instance
    :type space: H1Space
    :param vrange: [min_value, max_value] for color scaling
    :type vrange: Tuple[int, int]
    :param dx: Grid spacing in x-direction (default 0.01)
    :type dx: float
    :param dy: Grid spacing in y-direction (default 0.01)
    :type dy: float
    :return: Tuple of (ax_dx, ax_dy, min_val, max_val) where ax_dx and ax_dy are the axes
                for the x and y gradient components, and min_val and max_val are the
                minimum and maximum gradient values across both components.
    :rtype: Tuple[Axes3D, Axes3D, float, float]
    """

    fig = plt.figure(figsize=(14, 6))  # type: ignore
    ax_dx = fig.add_subplot(1, 2, 1, projection="3d")
    ax_dy = fig.add_subplot(1, 2, 2, projection="3d")

    trigs = [trig.points for trig in space.tri.trigs]
    x_coords = [point.coordinates[0] for point in space.tri.points]
    y_coords = [point.coordinates[1] for point in space.tri.points]

    # Setup both axes with mesh
    for ax in [ax_dx, ax_dy]:
        ax.triplot(x_coords, y_coords, trigs)  # type: ignore
        ax.plot(x_coords, y_coords, "o")  # type: ignore

    min_val = 1e16
    max_val = -1e16

    for i, trig in enumerate(space.tri.trigs):
        x = np.arange(-1, 1 + dx, dx)
        y = np.arange(-1, 1 + dy, dy)
        X, Y = np.meshgrid(x, y)
        X_t, Y_t = duffy(X, Y)
        s = barycentric_coordinates(X_t.flatten(), Y_t.flatten())

        node_0 = np.array(space.tri.points[trig.points[0]].coordinates)
        node_0.shape = (2, 1)
        node_1 = np.array(space.tri.points[trig.points[1]].coordinates)
        node_1.shape = (2, 1)
        node_2 = np.array(space.tri.points[trig.points[2]].coordinates)
        node_2.shape = (2, 1)
        trig_nodes = node_0 * s[0, :] + node_1 * s[1, :] + node_2 * s[2, :]

        fel = space.elements[i]
        dshape = fel.dshape_functions(X_t.flatten(), Y_t.flatten())

        # Apply Jacobian inverse transformation to map gradients from reference to physical space
        # Create element transformation to get Jacobian inverse
        points = np.array([space.tri.points[p].coordinates for p in trig.points])
        eltrans = ElementTransformationTrig(points)
        Jinv = eltrans.get_jacobian_inverse()

        # Transform dshape with Jacobian inverse
        nip = X_t.flatten().shape[0]
        dshape_transformed = dshape.copy()
        for j in range(fel.ndof):
            temp = dshape[j, :].copy()
            temp.shape = (2, nip)
            temp2 = Jinv @ temp
            temp2.shape = (2 * nip,)
            dshape_transformed[j, :] = temp2

        # Compute gradient components with transformed derivatives
        dx_vals = np.matrix(dshape_transformed[:, :nip].T) * u[space.dofs[i]]
        dy_vals = np.matrix(dshape_transformed[:, nip:].T) * u[space.dofs[i]]

        min_val = np.min([min_val, np.min(dx_vals), np.min(dy_vals)])
        max_val = np.max([max_val, np.max(dx_vals), np.max(dy_vals)])

        # Plot x-gradient
        ax_dx.plot_surface(  # type: ignore
            trig_nodes[0, :].reshape(len(x), len(y)),
            trig_nodes[1, :].reshape(len(x), len(y)),
            dx_vals.reshape(len(x), len(y)),
            cmap="jet",
            linestyle="None",
            vmin=vrange[0],
            vmax=vrange[1],
        )

        # Plot y-gradient
        ax_dy.plot_surface(  # type: ignore
            trig_nodes[0, :].reshape(len(x), len(y)),
            trig_nodes[1, :].reshape(len(x), len(y)),
            dy_vals.reshape(len(x), len(y)),
            cmap="jet",
            linestyle="None",
            vmin=vrange[0],
            vmax=vrange[1],
        )

    ax_dx.set_title("∂u/∂x")  # type: ignore
    ax_dy.set_title("∂u/∂y")  # type: ignore

    return ax_dx, ax_dy, min_val, max_val
