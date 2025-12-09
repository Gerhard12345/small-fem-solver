from typing import Callable

import numpy as np
from numpy.typing import NDArray
from pyfemsolver.solverlib.space import H1Space


def set_boundary_values(space: H1Space, g: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]):
    """
    Set boundary values for the finite element space.

    :param space: H1 finite element space instance
    :type space: H1Space
    :param g: Function defining boundary values
    :type g: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
    :return: boundary dof values
    :rtype: NDArray[np.floating]
    """
    print("set boundary vals")
    boundary_mass = np.matrix(np.zeros((space.ndof, space.ndof)))
    boundary_f_vector = np.zeros((space.ndof, 1))
    u_bnd = np.zeros((len(space.unique_boundary_dofs), 1))
    space.assemble_boundary_mass(boundary_mass)
    space.assemble_boundary_element_vector(boundary_f_vector, g)
    k = 0
    for j, node in enumerate(space.tri.points):
        if node.is_boundary_point:
            u_bnd[k] = g(*node.coordinates)
            k += 1

    vertex_dofs = range(len(space.tri.boundary_points))
    edge_dofs = range(len(space.tri.boundary_points), len(space.unique_boundary_dofs))

    X, Y = np.meshgrid(space.unique_boundary_dofs, space.unique_boundary_dofs)

    boundary_mass = boundary_mass[Y, X]
    boundary_f_vector = boundary_f_vector[space.unique_boundary_dofs]
    edge_contribution = boundary_mass[:, vertex_dofs] * u_bnd[vertex_dofs]
    boundary_f_vector -= edge_contribution

    boundary_mass_diag = np.diag(boundary_mass).copy()

    for i in range(len(vertex_dofs), len(space.unique_boundary_dofs)):
        boundary_f_vector[i] /= np.sqrt(boundary_mass_diag[i])
        for j in range(len(vertex_dofs), len(space.unique_boundary_dofs)):
            boundary_mass[i, j] /= np.sqrt(boundary_mass_diag[i]) * np.sqrt(boundary_mass_diag[j])
    u_bnd[edge_dofs] = boundary_mass[edge_dofs, :][:, edge_dofs] ** -1 * boundary_f_vector[edge_dofs]
    for i in range(len(vertex_dofs), len(space.unique_boundary_dofs)):
        u_bnd[i] /= np.sqrt(boundary_mass_diag[i])

    print("done")
    return u_bnd


def solve_by_condensation(
    space: H1Space, system_matrix: NDArray[np.floating], f_vector: NDArray[np.floating], show_condition_number: bool = False
):
    """
    Solve the linear system of equations A * u = f for the inner dofs using static condensation.

    :param space: H1 finite element space instance
    :type space: H1Space
    :param system_matrix: The system matrix A to be inverted
    :type system_matrix: NDArray[np.floating]
    :param f_vector: The right-hand side vector f
    :type f_vector: NDArray[np.floating]
    :param show_condition_number: Whether to print the condition number of the condensed matrix (costly operation)
    :type show_condition_number: bool
    :return: Solution vector
    :rtype: NDArray[np.floating]
    """
    print("Solve by static condensation")
    u = np.matrix(np.zeros((len(space.inner_dofs), 1)))
    ndof_bubble = 1 / 2 * len(space.tri.trigs) * (space.p - 1) * (space.p - 2)
    ndof_edge = int(len(space.inner_dofs) - ndof_bubble)
    a_ee = system_matrix[:ndof_edge, :ndof_edge]
    a_ii = system_matrix[ndof_edge:, ndof_edge:]
    a_ei = system_matrix[ndof_edge:, :ndof_edge]
    a_ie = system_matrix[:ndof_edge, ndof_edge:]
    f_e = f_vector[:ndof_edge]
    f_i = f_vector[ndof_edge:]
    print("Invert bubble function matrix")
    a_inner_inverse = np.linalg.inv(a_ii)
    print("done")
    condensed_matrix = a_ee - a_ie @ a_inner_inverse @ a_ei
    if show_condition_number:
        print("Cond(condensed_matrix) = ", np.linalg.cond(condensed_matrix))
    condensed_f = f_e - a_ie @ a_inner_inverse @ f_i
    print("inverting")
    u[:ndof_edge] = np.linalg.inv(condensed_matrix) @ condensed_f
    print("done inverting")
    u[ndof_edge:] = a_inner_inverse @ (f_i - a_ei @ u[:ndof_edge])
    print("done, max(|S*u-f|) = ", np.max(np.abs(system_matrix * u - f_vector)))
    return u


def solve_bvp(
    a_1: float,
    a_2: float,
    space: H1Space,
    u_bnd: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]],
    f: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]],
    show_condition_number: bool = False,
):
    mass = np.zeros((space.ndof, space.ndof))
    gradu_gradv = np.zeros((space.ndof, space.ndof))
    f_vector = np.zeros((space.ndof, 1))
    u = np.zeros((space.ndof, 1))
    print("Assembling")
    space.assemble_mass(mass)
    space.assemble_gradu_gradv(gradu_gradv)
    space.assemble_element_vector(f_vector, f)
    print("Done")
    # set boundary values
    u[space.unique_boundary_dofs] = set_boundary_values(space, u_bnd)

    # the system matrix is the sum of all involved bilinear forms
    print("diagonally precondition")
    system_matrix = a_1 * mass + a_2 * gradu_gradv
    diag_system_matrix = np.diag(system_matrix).copy()
    print("got diag mass")
    # incoprorate bc on right side
    boundary_contribution = system_matrix[:, space.unique_boundary_dofs] @ u[space.unique_boundary_dofs]
    f_vector -= boundary_contribution
    print("updated boundary condition")
    # diagonally scale system matrix and rhs
    for i in space.inner_dofs:
        f_vector[i] /= np.sqrt(diag_system_matrix[i])
        # for j in space.inner_dofs:
        system_matrix[i, :] /= np.sqrt(diag_system_matrix[i])
        system_matrix[:, i] /= np.sqrt(diag_system_matrix[i])
    print("done")
    # solve
    u[space.inner_dofs] = solve_by_condensation(
        space, system_matrix[space.inner_dofs, :][:, space.inner_dofs], f_vector[space.inner_dofs], show_condition_number
    )
    # diagonally unscale solution
    for i in space.inner_dofs:
        u[i] /= np.sqrt(diag_system_matrix[i])
    if show_condition_number:
        print(f"Cond(system matrix) = {np.linalg.cond(system_matrix[space.inner_dofs,:][:,space.inner_dofs])}")
    return u, system_matrix, f_vector
