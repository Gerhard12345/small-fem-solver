"""
Finite element solver module. Provides functions to set boundary values,
solve by static condensation, and solve boundary value problems using H1 finite element spaces.
"""

import numpy as np
from numpy.typing import NDArray
from .space import H1Space
from .coefficientfunction import CoefficientFunction, ConstantCoefficientFunction
from .forms import LinearForm
from .forms import BilinearForm
from .integrators import EdgeMass, EdgeSource


def set_boundary_values(dof_vector: NDArray[np.floating], space: H1Space, g: CoefficientFunction):
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
    u_bnd = np.zeros((len(space.dirichlet_dofs), 1))
    edge_mass = EdgeMass(coefficient=ConstantCoefficientFunction(1), space=space, is_boundary=True)
    edge_mass.assemble(boundary_mass)
    edge_source = EdgeSource(coefficient=g, space=space, is_boundary=True)
    edge_source.assemble(boundary_f_vector)

    X, Y = np.meshgrid(space.dirichlet_dofs, space.dirichlet_dofs)

    boundary_mass = boundary_mass[Y, X]
    boundary_f_vector = boundary_f_vector[space.dirichlet_dofs]
    boundary_mass_diag = np.diag(boundary_mass).copy()

    for i in range(len(space.dirichlet_dofs)):
        boundary_f_vector[i] /= np.sqrt(boundary_mass_diag[i])
        for j in range(len(space.dirichlet_dofs)):
            boundary_mass[i, j] /= np.sqrt(boundary_mass_diag[i]) * np.sqrt(boundary_mass_diag[j])
    u_bnd = boundary_mass**-1 * boundary_f_vector
    for i in range(len(space.dirichlet_dofs)):
        u_bnd[i] /= np.sqrt(boundary_mass_diag[i])

    print("done")
    dof_vector[space.dirichlet_dofs] = u_bnd


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
    u = np.matrix(np.zeros((len(space.free_dofs), 1)))
    ndof_bubble = 1 / 2 * len(space.tri.trigs) * (space.p - 1) * (space.p - 2)
    ndof_edge = int(len(space.free_dofs) - ndof_bubble)
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
    bilinearform: BilinearForm,
    linearform: LinearForm,
    u: NDArray[np.floating],
    space: H1Space,
    show_condition_number: bool = False,
):
    system_matrix = np.zeros((space.ndof, space.ndof))
    f_vector = np.zeros((space.ndof, 1))
    # u = np.zeros((space.ndof, 1))
    print("Assembling")
    bilinearform.assemble(system_matrix)
    linearform.assemble(f_vector)
    print("Done")

    # the system matrix is the sum of all involved bilinear forms
    print("diagonally precondition")
    diag_system_matrix = np.diag(system_matrix).copy()
    print("got diag mass")
    # incoprorate bc on right side
    boundary_contribution = system_matrix[:, space.dirichlet_dofs] @ u[space.dirichlet_dofs]
    f_vector -= boundary_contribution
    print("updated boundary condition")
    # diagonally scale system matrix and rhs
    for i in space.free_dofs:
        f_vector[i] /= np.sqrt(diag_system_matrix[i])
        # for j in space.inner_dofs:
        system_matrix[i, :] /= np.sqrt(diag_system_matrix[i])
        system_matrix[:, i] /= np.sqrt(diag_system_matrix[i])
    print("done")
    # solve
    u[space.free_dofs] = solve_by_condensation(
        space, system_matrix[space.free_dofs, :][:, space.free_dofs], f_vector[space.free_dofs], show_condition_number
    )
    # diagonally unscale solution
    for i in space.free_dofs:
        u[i] /= np.sqrt(diag_system_matrix[i])
    if show_condition_number:
        print(f"Cond(system matrix) = {np.linalg.cond(system_matrix[space.free_dofs,:][:,space.free_dofs])}")
