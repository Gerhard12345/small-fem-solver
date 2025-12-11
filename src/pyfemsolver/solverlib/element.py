"""
The file provides a H! Finite element and associated methods for shape functions,
derivatives, mass matrix, stiffness matrix, and element vectors.
The basis functions are constructed using integrated Jacobi polynomials as described in:
https://www3.risc.jku.at/publications/download/risc_4253/buch.pdf
"""

import numpy as np
from numpy.typing import NDArray

from .coefficientfunction import CoefficientFunction
from .elementtransformation import ElementTransformationTrig, ElementTransformationLine
from .integrationrules import get_integration_rule_trig, get_integration_rule_line
from .polynomials import integrated_jacobi_polynomial, barycentric_coordinates, barycentric_coordinates_line, edge_based_polynomials, h


class H1Fel:
    """
    H1 Finite element class. Provides methods for shape functions, derivatives,
    mass matrix, stiffness matrix, and element vectors.
    """

    # Class-level integration rule instances (shared across all instances)
    def __init__(self, order: int):
        self.p = order
        self.edges = [(0, 1), (1, 2), (2, 0)]
        self.flipped_edge = [False, False, False]
        self.ndof_vertex = 3
        self.ndof_faces = 3 * (self.p - 1)
        self.ndof_facet = self.p - 1
        self.ndof_inner = int((self.p - 1) * (self.p - 2) / 2)
        self.ndof = self.ndof_vertex + self.ndof_faces + self.ndof_inner

    def flip_edge(self, i: int) -> None:
        """
        Flips the orientation of edge i by reversing its start and end point.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param i: The index of the edge to be flipped (0, 1, or 2)
        :type i: int
        """
        self.edges[i] = (self.edges[i][1], self.edges[i][0])

    def shape_functions(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Computes the shape functions at given points (x,y) in the reference triangle.
        The first three shape functions correspond to the vertices,
        the next 3*(p-1) to the edges, and the remaining to the interior bubble
        functions. Shape functions are constructed using integrated Jacobi polynomials.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param x: Evaluation points in x direction
        :type x: NDArray[np.floating]
        :param y: Evaluation points in y direction
        :type y: NDArray[np.floating]
        :return: Evaluated shape functions (row index corresponds to shape function, column index to evaluation point)
        :rtype: NDArray[float64]
        """
        x = x.reshape(x.size)
        y = y.reshape(y.size)
        shape = np.zeros((3 * self.p + int((self.p - 2) * (self.p - 1) / 2), *x.shape))
        # Vertex functions
        shape[:3, :] = barycentric_coordinates(x, y)
        if self.p == 1:
            return shape
        # Edge functions
        for i in range(3):
            shape[(3 + i * (self.p - 1)) : (3 + (i + 1) * (self.p - 1)), :] = edge_based_polynomials(self.p, self.edges[i], x, y)
        if self.p < 3:
            return shape
        hs = h(self.p, x, y)
        i0 = 3 * self.p
        d0 = self.p - 2
        # Bubble functions
        for i in range(2, self.p):
            shape[i0 : i0 + d0, :] = shape[3 + i - 2] * hs[i - 2]
            i0 = i0 + d0
            d0 -= 1
        return shape

    def dshape_functions(self, x: NDArray[np.floating], y: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Computes the shape functions gradients at given points (x,y) in the reference triangle.
        The first three shape functions correspond to the vertices,
        the next 3*(p-1) to the edges, and the remaining to the interior bubble
        functions.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param x: Evaluation points in x direction
        :type x: NDArray[np.floating]
        :param y: Evaluation points in y direction
        :type y: NDArray[np.floating]
        :return: Evaluated gradients of shape functions. First half are d/dx, second half d/dy (row wise)
        :rtype: NDArray[float64]
        """
        x = x.reshape(x.size)
        y = y.reshape(y.size)
        nip = x.size
        dshape = np.zeros((3 * self.p + int((self.p - 2) * (self.p - 1) / 2), 2 * nip))
        delta = 1e-7
        dshape[:3, :nip] = (barycentric_coordinates(x + delta, y) - barycentric_coordinates(x - delta, y)) / (2 * delta)
        dshape[:3, nip:] = (barycentric_coordinates(x, y + delta) - barycentric_coordinates(x, y - delta)) / (2 * delta)
        if self.p == 1:
            return dshape
        for i in range(3):
            dshape[(3 + i * (self.p - 1)) : (3 + (i + 1) * (self.p - 1)), :nip] = (
                1
                / (2 * delta)
                * (edge_based_polynomials(self.p, self.edges[i], x + delta, y) - edge_based_polynomials(self.p, self.edges[i], x - delta, y))
            )
            dshape[(3 + i * (self.p - 1)) : (3 + (i + 1) * (self.p - 1)), nip:] = (
                1
                / (2 * delta)
                * (edge_based_polynomials(self.p, self.edges[i], x, y + delta) - edge_based_polynomials(self.p, self.edges[i], x, y - delta))
            )
        if self.p < 3:
            return dshape
        tsp = h(self.p, x + delta, y)
        tsm = h(self.p, x - delta, y)
        tspy = h(self.p, x, y + delta)
        tsmy = h(self.p, x, y - delta)

        i0 = 3 * self.p
        d0 = self.p - 2
        gp = edge_based_polynomials(self.p, self.edges[0], x + delta, y)
        gm = edge_based_polynomials(self.p, self.edges[0], x - delta, y)
        gpy = edge_based_polynomials(self.p, self.edges[0], x, y + delta)
        gmy = edge_based_polynomials(self.p, self.edges[0], x, y - delta)

        for i in range(2, self.p):
            dshape[i0 : i0 + d0, :nip] = (gp[i - 2] * tsp[i - 2] - gm[i - 2] * tsm[i - 2]) / (2 * delta)
            dshape[i0 : i0 + d0, nip:] = (gpy[i - 2] * tspy[i - 2] - gmy[i - 2] * tsmy[i - 2]) / (2 * delta)
            i0 = i0 + d0
            d0 -= 1

        return dshape

    def edge_shape_functions(self, t: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Computes the edge shape functions at given points t in the reference line.
        The first two shape functions correspond to the vertices, the next (p-1) to
        integrated Jacobi polynomials.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param t: Evaluation points along the line
        :type t: NDArray[np.floating]
        :return: Evaluated edge shape functions
        :rtype: NDArray[float64]
        """
        t = t.reshape(t.size)
        shape = np.zeros((self.p + 1, *t.shape))
        shape[:2, :] = barycentric_coordinates_line(t)
        shape[2 : (self.p + 1), :] = integrated_jacobi_polynomial(self.p, t, 0)[2:, :]
        return shape

    def calc_mass_matrix(self, eltrans: ElementTransformationTrig, f: CoefficientFunction) -> NDArray[np.floating]:
        """
        Computes the mass matrix for the element defined by the given transformation.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param eltrans: The element transformation for the triangle
        :type eltrans: ElementTransformationTrig
        :return: The mass matrix of the element
        :rtype: NDArray[float64]
        """
        X, Y, omega = get_integration_rule_trig(self.p + 1)
        omega *= eltrans.getjacobian_determinant()
        shape = self.shape_functions(X, Y)
        x_phys, y_phys = eltrans.transform_points(X, Y)
        f_vals = f(x_phys, y_phys, eltrans.region)
        mass = (shape * omega) @ (f_vals * shape.T)
        mass[np.abs(mass) < 1e-16] = 0
        return mass

    def calc_gradu_gradv_matrix(self, eltrans: ElementTransformationTrig) -> NDArray[np.floating]:
        """
        Computes the stiffness matrix (grad u, grad v) for the
        element defined by the given transformation.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param eltrans: The element transformation for the triangle
        :type eltrans: ElementTransformationTrig
        :return: The stiffness matrix of the element
        :rtype: NDArray[float64]
        """
        X, Y, omega = get_integration_rule_trig(self.p + 1)
        omega *= eltrans.getjacobian_determinant()
        omega = np.concatenate([omega, omega])
        dshape = self.dshape_functions(X, Y)
        Jinv = eltrans.get_jacobian_inverse()
        for i in range(self.ndof):
            temp = dshape[i, :].reshape((2, len(omega) // 2))
            np.matmul(Jinv, temp, out=temp)
        gradu_gradv = (dshape * omega.T) @ dshape.T
        gradu_gradv[np.abs(gradu_gradv) < 1e-16] = 0
        return gradu_gradv

    def calc_element_vector(self, eltrans: ElementTransformationTrig, f: CoefficientFunction) -> NDArray[np.floating]:
        """
        Computes the element vector for the element defined by the given transformation
        and the function f.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param eltrans: The element transformation for the triangle
        :type eltrans: ElementTransformationTrig
        :param f: The function to be integrated over the element
        :type f: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
        :return: The element vector for the element and function f.
        :rtype: NDArray[float64]
        """
        X, Y, omega = get_integration_rule_trig(self.p + 1)
        omega *= eltrans.getjacobian_determinant()
        shape = self.shape_functions(X, Y)
        x_phys, y_phys = eltrans.transform_points(X, Y)
        f_vals = f(x_phys, y_phys, eltrans.region)
        element_vector = (shape * omega.T) @ f_vals
        element_vector[np.abs(element_vector) < 1e-16] = 0
        return element_vector

    def calc_edge_mass_matrix(self, eltrans: ElementTransformationLine) -> NDArray[np.floating]:
        """
        Computes the mass matrix for an edge defined by the given transformation.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param eltrans: The element transformation for the line
        :type eltrans: ElementTransformationLine
        :return: The mass matrix for the edge.
        :rtype: NDArray[float64]
        """
        X, omega = get_integration_rule_line(self.p + 1)
        omega *= eltrans.getjacobian_determinant()
        shape = self.edge_shape_functions(X)
        mass = (shape * omega) @ shape.T
        mass[np.abs(mass) < 1e-16] = 0
        return mass

    def calc_edge_element_vector(self, eltrans: ElementTransformationLine, f: CoefficientFunction) -> NDArray[np.floating]:
        """
        Computes the element vector for an edge defined by the given transformation
        and the function f.

        :param self: The H1 finite element instance
        :type self: H1Fel
        :param eltrans: The element transformation for the line
        :type eltrans: ElementTransformationLine
        :param f: The function to be integrated over the edge
        :type f: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
        :return: The element vector for the edge and function f.
        :rtype: NDArray[float64]
        """
        X, omega = get_integration_rule_line(self.p + 1)
        x_phys, y_phys = eltrans.transform_points(X)
        f_vals = f(x_phys, y_phys, eltrans.region)
        omega *= eltrans.getjacobian_determinant()
        shape = self.edge_shape_functions(X)
        element_vector = (shape * omega) @ f_vals
        element_vector[np.abs(element_vector) < 1e-16] = 0
        return element_vector
