"""
Finite element solver module. Provides functions to set boundary values,
solve by static condensation, and solve boundary value problems using H1 finite element spaces.
"""

import numpy as np
from numpy.typing import NDArray
import time
from scipy.sparse import linalg, csr_array, diags_array
from .space import H1Space
from .coefficientfunction import CoefficientFunction, ConstantCoefficientFunction
from .forms import LinearForm
from .forms import BilinearForm
from .integrators import EdgeMass, EdgeSource

s = 0


def cg_callback(u: NDArray[np.floating]):
    pass
    # global s
    # s += 1
    # print(s)


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
    boundary_mass = np.zeros((space.ndof, space.ndof))
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
    u_bnd = np.linalg.inv(boundary_mass) @ boundary_f_vector
    for i in range(len(space.dirichlet_dofs)):
        u_bnd[i] /= np.sqrt(boundary_mass_diag[i])

    print("done")
    dof_vector[space.dirichlet_dofs] = u_bnd


def invert_block_diagonal_matrix(a: NDArray[np.floating], block_size: int):
    n_blocks = a.shape[0] // block_size
    for block in range(n_blocks):
        a[(block * block_size) : ((block + 1) * block_size), (block * block_size) : ((block + 1) * block_size)] = np.linalg.inv(
            a[(block * block_size) : ((block + 1) * block_size), (block * block_size) : ((block + 1) * block_size)].toarray()
        )


def solve_linear_equations(a: csr_array, f: NDArray[np.floating], precond: csr_array) -> NDArray[np.floating]:
    # u = np.zeros((a.shape[0],))
    # f = f.reshape(f.size)
    # r = f - a @ u
    # d = r
    # for k in range(3 * a.shape[0]):
    #     z = a @ d
    #     alpha = np.dot(r, r) / np.dot(d, r)
    #     u += alpha * d
    #     r_new = r - alpha * z
    #     beta = np.dot(r_new, r_new) / np.dot(r, r)
    #     d = r_new + beta * d
    #     r = r_new
    #     res = np.dot(r, a @ r)
    #     print(res)
    t0 = time.time()
    direct_solve = False
    if direct_solve:
        a = a.toarray()
        print("Solving linear equations with direct solver")
        u = np.linalg.inv(a) @ f
    else:
        print("Solving linear equations with cg solver")
        # ev_max = linalg.eigs(diag_precond@a@diag_precond, k=1, which="LM", return_eigenvectors=False)
        # ev_min = linalg.eigs(diag_precond@a@diag_precond, k=1, which="SM", return_eigenvectors=False)
        # print(rf"extremal eigenvalues are \lambda_min = {ev_min} and \lambda_max = {ev_max}. Condition = {np.abs(ev_max)/np.abs(ev_min)}")
        u, _ = linalg.cg(A=a, b=f, M=precond, rtol=1e-14, callback=cg_callback)
        u = u.reshape(u.size, 1)
    t1 = time.time()
    print(f"Inverted system in {t1-t0} seconds.")
    print("max(|S*u-f|) = ", np.max(np.abs(a @ u - f)))
    return u


def solve_by_condensation(
    space: H1Space, system_matrix: NDArray[np.floating], f_vector: NDArray[np.floating], precond: csr_array, show_condition_number: bool = False
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
    u = np.zeros((len(space.free_dofs), 1))
    bubble_functions_per_trig = int(1 / 2 * (space.p - 1) * (space.p - 2))
    ndof_bubble = len(space.tri.trigs) * bubble_functions_per_trig
    ndof_edge = int(len(space.free_dofs) - ndof_bubble)
    a_ee = system_matrix[:ndof_edge, :ndof_edge]
    a_ii = system_matrix[ndof_edge:, ndof_edge:]
    a_ei = system_matrix[ndof_edge:, :ndof_edge]
    a_ie = system_matrix[:ndof_edge, ndof_edge:]
    f_e = f_vector[:ndof_edge]
    f_i = f_vector[ndof_edge:]
    precond_ee = precond[:ndof_edge, :ndof_edge]
    precond_ii = precond[ndof_edge:, ndof_edge:]
    precond_ei = precond[ndof_edge:, :ndof_edge]
    precond_ie = precond[:ndof_edge, ndof_edge:]
    print("Invert bubble function matrix")
    if space.p >= 3:
        invert_block_diagonal_matrix(a_ii, block_size=bubble_functions_per_trig)
    print("done")
    condensed_matrix = a_ee - a_ie @ a_ii @ a_ei
    condensed_precond = precond_ee - precond_ie @ precond_ii @ precond_ei
    if show_condition_number:
        print("Cond(condensed_matrix) = ", np.linalg.cond(condensed_matrix))
    condensed_f = f_e - a_ie @ a_ii @ f_i
    u[:ndof_edge] = solve_linear_equations(condensed_matrix, condensed_f, condensed_precond)
    u[ndof_edge:] = a_ii @ (f_i - a_ei @ u[:ndof_edge])
    return u


def get_precond(space: H1Space, a: csr_array):
    blockwise = True
    if blockwise == False:
        precond = diags_array(a.diagonal() ** -1).tocsr()
    else:
        precond = 0.0 * csr_array(a).copy()
        blocks = []
        blocks.extend(space.vertex_dofs)
        blocks.extend(space.edge_dofs)
        blocks.extend(space.bubble_dofs)
        import scipy.linalg

        for block in blocks:
            BX, BY = np.meshgrid(block, block)
            temp = scipy.linalg.sqrtm(np.linalg.inv((a[:, block][block, :].toarray())))
            precond[BX, BY] = temp
    return precond


def solve_bvp(
    bilinearform: BilinearForm,
    linearform: LinearForm,
    u: NDArray[np.floating],
    space: H1Space,
    show_condition_number: bool = False,
):
    # system_matrix = np.zeros((space.ndof, space.ndof))
    system_matrix = space.init_system_matrix()
    f_vector = np.zeros((space.ndof, 1))
    print("Assembling")
    bilinearform.assemble(system_matrix)
    linearform.assemble(f_vector)
    print("Done")

    # the system matrix is the sum of all involved bilinear forms with
    # incoprorate bc on right side
    boundary_contribution = system_matrix[:, space.dirichlet_dofs] @ u[space.dirichlet_dofs]
    f_vector -= boundary_contribution
    precond = get_precond(space, system_matrix)
    system_matrix = system_matrix[:, space.free_dofs][space.free_dofs, :]
    precond = precond[:, space.free_dofs][space.free_dofs, :]
    f_vector = f_vector[space.free_dofs]
    print("updated boundary condition")

    # solve
    use_static_condensation = True
    if use_static_condensation:
        u[space.free_dofs] = solve_by_condensation(space, system_matrix, f_vector, precond, show_condition_number)
    else:
        u[space.free_dofs] = solve_linear_equations(system_matrix, f_vector, precond)
    if show_condition_number:
        print(f"Cond(system matrix) = {np.linalg.cond(system_matrix)}")
