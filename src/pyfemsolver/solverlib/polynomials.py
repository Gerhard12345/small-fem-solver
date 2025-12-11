"""
Module for polynomial evaluations used in finite element shape functions. Provides Jacobi polynomials,
integrated Jacobi polynomials, barycentric coordinates, and helper functions for edge and bubble shape functions.
"""

from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray


def jacobi_polynomial(n: int, x: NDArray[np.floating], alpha: float | int) -> NDArray[np.floating]:
    """
    Evaluate Jacobi polynomials of order n at points x with parameter alpha.

    :param n: Order of the Jacobi polynomial
    :type n: int
    :param x: Evaluation points
    :type x: NDArray[np.floating]
    :param alpha: The alpha parameter of the Jacobi polynomial
    :type alpha: float | int
    :return: The first n Jacobi polynomials evaluated at x
    :rtype: NDArray[float64]
    """
    vals = np.zeros((n + 1, len(x)))
    vals[0, :] = 1
    if n == 0:
        return vals
    vals[1, :] = 0.5 * (alpha + (alpha + 2) * x)
    for j in range(1, n):
        a_1 = (2 * j + alpha + 1) / ((2 * j + 2) * (j + alpha + 1) * (2 * j + alpha))
        a_2 = (2 * j + alpha + 2) * (2 * j + alpha)
        a_3 = j * (j + alpha) * (2 * j + alpha + 2) / ((j + 1) * (j + alpha + 1) * (2 * j + alpha))
        vals[j + 1, :] = a_1 * (a_2 * x + alpha**2) * vals[j, :] - a_3 * vals[j - 1, :]
    return vals


def integrated_jacobi_polynomial(n: int, x: NDArray[np.floating], alpha: float | int) -> NDArray[np.floating]:
    """
    Evaluate integrated Jacobi polynomials of order n at points x with parameter alpha.

    :param n: Order of the integrated Jacobi polynomial
    :type n: int
    :param x: Evaluation points
    :type x: NDArray[np.floating]
    :param alpha: The alpha parameter of the Jacobi polynomial
    :type alpha: float | int
    :return: The first n Jacobi polynomials evaluated at x
    :rtype: NDArray[float64]
    """
    vals = np.zeros((n + 1, len(x)))
    vals[0, :] = 1
    if n == 0:
        return vals
    vals[1, :] = x + 1
    if n == 1:
        return vals
    jacobi_poly_vals = jacobi_polynomial(n + 1, x, alpha)
    for j in range(2, n + 1):
        a_1 = (2 * j + 2 * alpha) / ((2 * j + alpha - 1) * (2 * j + alpha))
        a_2 = 2 * alpha / ((2 * j + alpha - 2) * (2 * j + alpha))
        a_3 = (2 * j - 2) / ((2 * j + alpha - 1) * (2 * j + alpha - 2))
        vals[j, :] = a_1 * jacobi_poly_vals[j, :] + a_2 * jacobi_poly_vals[j - 1, :] - a_3 * jacobi_poly_vals[j - 2, :]
    return vals


def barycentric_coordinates(x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Computes the barycentric coordinates for a triangle with corners (-1,-1),(1,-1),(0,1).

    :param x: Evaluation points in x direction
    :type x: NDArray[np.floating]
    :param y: Evaluation points in y direction
    :type y: NDArray[np.floating]
    :return: Evaluated barycentric coordinates
    :rtype: NDArray[float64]
    """
    return 1 / 4 * np.array([1 - 2 * x - y, 1 + 2 * x - y, 2 + 2 * y])


def barycentric_coordinates_line(t: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Computes the barycentric coordinates for a line with corners (-1,0),(1,0).

    :param t: Evaluation points along the line
    :type t: NDArray[np.floating]
    :return: Evaluated barycentric coordinates
    :rtype: NDArray[float64]
    """
    # barycentric coordinates, corner sorting: (-1,0),(1,0)
    return 1 / 2 * np.array([1 - t, 1 + t])


def edge_based_polynomials(p: int, E: Tuple[int, int], x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Helper function to compute edge shape functions.
    The funtion vanishes on all edges except edge E.

    :param p: Order of the polynomial on the edge
    :type p: int
    :param E: The edge defined by its two vertex indices (0,1), (1,2), or (2,0)
    :type E: Tuple[int, int]
    :param x: Evaluation points in x direction
    :type x: NDArray[np.floating]
    :param y: Evaluation points in y direction
    :type y: NDArray[np.floating]
    :return: Evaluated edge shape functions
    :rtype: NDArray[float64]
    """
    e_1 = E[0]
    e_2 = E[1]
    l = barycentric_coordinates(x, y)
    l1 = l[e_2] + l[e_1]
    l2 = l[e_2] - l[e_1]
    with np.errstate(divide="ignore", invalid="ignore"):
        #x = l2 / l1
        x = np.where(l1 == 0, 1, l2 / l1)
    vals_1 = integrated_jacobi_polynomial(p, x, 0)[2:, :]
    vals_2 = np.array([l1**j for j in range(2, p + 1)])
    return vals_1 * vals_2


def h(p: int, x: NDArray[np.floating], y: NDArray[np.floating]) -> List[NDArray[np.floating]]:
    """
    Helper function to compute bubble shape functions. The function vanishes on edge (0,1),
    thus compensating the edge functions.

    :param p: Order of the polynomial in the interior
    :type p: int
    :param x: Evaluation points in x direction
    :type x: NDArray[np.floating]
    :param y: Evaluation points in y direction
    :type y: NDArray[np.floating]
    :return: Evaluated helper functions
    :rtype: List[NDArray[float64]]
    """
    l = barycentric_coordinates(x, y)
    l1 = 2 * l[2] - 1
    vals_1: List[NDArray[np.floating]] = []
    for i in range(2, p):
        s = integrated_jacobi_polynomial(p - i, l1, 2 * i - 1)[1:, :]
        vals_1.append(s)

    return vals_1
