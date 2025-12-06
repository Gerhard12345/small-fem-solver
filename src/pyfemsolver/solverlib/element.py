import numpy as np
from numpy.typing import NDArray
from copy import copy
from typing import Tuple


def jacobi_polynomial(n, x, alpha):
    vals = np.zeros((n + 1, len(x)))
    vals[0, :] = 1
    vals[1, :] = 0.5 * (alpha + (alpha + 2) * x)
    for j in range(1, n):
        a_1 = (2 * j + alpha + 1) / ((2 * j + 2) * (j + alpha + 1) * (2 * j + alpha))
        a_2 = (2 * j + alpha + 2) * (2 * j + alpha)
        a_3 = j * (j + alpha) * (2 * j + alpha + 2) / ((j + 1) * (j + alpha + 1) * (2 * j + alpha))
        vals[j + 1, :] = a_1 * (a_2 * x + alpha**2) * vals[j, :] - a_3 * vals[j - 1, :]
    return vals

def integrated_jacobi_polynomial(n: int, x: float | int, alpha: float | int):
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

def barycentric_coordinates(x, y):
    # barycentric coordinates, corner sorting: (-1,-1),(1,-1),(0,1)
    return 1 / 4 * np.array([1 - 2 * x - y, 1 + 2 * x - y, 2 + 2 * y])

def barycentric_coordinates_line(t):
    # barycentric coordinates, corner sorting: (-1,0),(1,0)
    return 1 / 2 * np.array([1 - t, 1 + t])

def g(p, E, x, y):
    e_1 = E[0]
    e_2 = E[1]
    l = barycentric_coordinates(x, y)
    l1 = l[e_2] + l[e_1]
    l2 = l[e_2] - l[e_1]
    with np.errstate(divide="ignore", invalid="ignore"):
        # x = l2 / l1
        x = np.where(l1 == 0, 0, l2 / l1)
    vals_1 = integrated_jacobi_polynomial(p, x, 0)[2:, :]
    vals_2 = np.array([l1**j for j in range(2, p + 1)])
    return vals_1 * vals_2

def h(p, x, y):
    l = barycentric_coordinates(x, y)
    l1 = 2 * l[2] - 1
    vals_1 = []
    for i in range(2, p):
        s = integrated_jacobi_polynomial(p - i, l1, 2 * i - 1)[1:, :]
        vals_1.append(s)

    return vals_1

def duffy(zeta, eta) -> Tuple[NDArray, NDArray]:
    return 0.5 * zeta * (1 - eta), eta

class H1Fel:
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
        self.edges[i] = tuple(reversed(self.edges[i]))

    def shape_functions(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        shape = np.zeros((3 * self.p + int((self.p - 2) * (self.p - 1) / 2), *x.shape))
        # Vertex functions
        shape[:3, :] = barycentric_coordinates(x, y)
        if self.p == 1:
            return shape
        # Edge functions
        for i in range(3):
            shape[(3 + i * (self.p - 1)) : (3 + (i + 1) * (self.p - 1)), :] = g(self.p, self.edges[i], x, y)
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

    def dshape_functions(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        nip = x.size
        Z = np.zeros((3 * self.p + int((self.p - 2) * (self.p - 1) / 2), 2 * nip))
        delta = 1e-7
        Z[:3, :nip] = (barycentric_coordinates(x + delta, y) - barycentric_coordinates(x - delta, y)) / (2 * delta)
        Z[:3, nip:] = (barycentric_coordinates(x, y + delta) - barycentric_coordinates(x, y - delta)) / (2 * delta)
        if self.p == 1:
            return Z
        for i in range(3):
            Z[(3 + i * (self.p - 1)) : (3 + (i + 1) * (self.p - 1)), :nip] = (
                1 / (2 * delta) * (g(self.p, self.edges[i], x + delta, y) - g(self.p, self.edges[i], x - delta, y))
            )
            Z[(3 + i * (self.p - 1)) : (3 + (i + 1) * (self.p - 1)), nip:] = (
                1 / (2 * delta) * (g(self.p, self.edges[i], x, y + delta) - g(self.p, self.edges[i], x, y - delta))
            )
        if self.p < 3:
            return Z
        tsp = h(self.p, x + delta, y)
        tsm = h(self.p, x - delta, y)
        tspy = h(self.p, x, y + delta)
        tsmy = h(self.p, x, y - delta)

        i0 = 3 * self.p
        d0 = self.p - 2
        temp_edge = copy(self.edges[0])
        if self.flipped_edge[0]:
            temp_edge.reverse()
        gp = g(self.p, temp_edge, x + delta, y)
        gm = g(self.p, temp_edge, x - delta, y)
        gpy = g(self.p, temp_edge, x, y + delta)
        gmy = g(self.p, temp_edge, x, y - delta)

        for i in range(2, self.p):
            Z[i0 : i0 + d0, :nip] = (gp[i - 2] * tsp[i - 2] - gm[i - 2] * tsm[i - 2]) / (2 * delta)
            Z[i0 : i0 + d0, nip:] = (gpy[i - 2] * tspy[i - 2] - gmy[i - 2] * tsmy[i - 2]) / (2 * delta)
            i0 = i0 + d0
            d0 -= 1

        return Z

    def edge_shape_functions(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        Z = np.zeros((self.p + 1, *t.shape))
        Z[:2, :] = barycentric_coordinates_line(t)
        Z[2 : (self.p + 1), :] = integrated_jacobi_polynomial(self.p, t, 0)[2:, :]
        return Z

    def get_integration_rule_trig(self, p: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        nodes, weights = np.polynomial.legendre.leggauss(2 * p + 1)
        X, Y = np.meshgrid(nodes, nodes)
        X_t, Y_t = duffy(X, Y)
        X_t = X_t.flatten()
        Y_t = Y_t.flatten()
        omega = np.outer(weights * 1 / 2 * (1 - nodes), weights).flatten()
        return X_t, Y_t, omega

    def get_integration_rule_line(self, p: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        nodes, weights = np.polynomial.legendre.leggauss(2 * p + 1)
        return nodes, weights

    def calc_mass_matrix(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        J = np.array(
            [
                0.5 * (points[1, :] - points[0, :]),
                0.25 * (2 * points[2] - points[1] - points[0]),
            ]
        )
        X, Y, omega = self.get_integration_rule_trig(self.p)
        omega *= abs(np.linalg.det(J))
        shape = self.shape_functions(X, Y)
        mass = (shape * omega) @ shape.T
        mass[np.abs(mass) < 1e-16] = 0
        return mass

    def calc_gradu_gradv_matrix(self, points):
        J = np.matrix(
            [
                0.5 * (points[1, :] - points[0, :]),
                0.25 * (2 * points[2] - points[1] - points[0]),
            ]
        )
        X, Y, omega = self.get_integration_rule_trig(self.p + 1)
        omega *= abs(np.linalg.det(J))
        omegas_2 = np.zeros((2 * omega.shape[0],))
        omegas_2[: len(omega)] = omega
        omegas_2[len(omega) :] = omega
        dshape = self.dshape_functions(X, Y)
        J = np.matrix(J) ** -1
        for i in range(self.ndof):
            temp = np.matrix(dshape[i, :].copy())
            temp.shape = (2, len(omega))
            temp = np.matrix(temp)
            temp2 = J * temp
            temp2.shape = (1, len(omegas_2))
            dshape[i, :] = temp2
        gradu_gradv = (dshape * omegas_2) @ dshape.T
        gradu_gradv[np.abs(gradu_gradv) < 1e-16] = 0
        return gradu_gradv

    def calc_element_vector(self, points, f):
        J = np.matrix(
            [
                0.5 * (points[1, :] - points[0, :]),
                0.25 * (2 * points[2] - points[1] - points[0]),
            ]
        )

        X, Y, omega = self.get_integration_rule_trig(self.p)
        omega *= abs(np.linalg.det(J))
        shape = self.shape_functions(X, Y)
        X.shape = (X.shape[0], 1)
        Y.shape = (Y.shape[0], 1)
        x, y, z = barycentric_coordinates(X, Y)
        XY_phys = points[0, :] * x + points[1, :] * y + points[2, :] * z
        f_vals = f(XY_phys[:, 0], XY_phys[:, 1])
        f_vals.shape = (f_vals.shape[0], 1)
        element_vector = (shape * omega) @ f_vals
        element_vector[np.abs(element_vector) < 1e-16] = 0
        return element_vector

    def calc_edge_mass_matrix(self, points):
        J = 0.5 * np.linalg.norm(points[1, :] - points[0, :])
        X, omega = self.get_integration_rule_line(self.p)
        omega *= abs(J)
        shape = self.edge_shape_functions(X)
        mass = (shape * omega) @ shape.T
        mass[np.abs(mass) < 1e-16] = 0
        return mass

    def calc_edge_element_vector(self, points, f):
        J = 0.5 * np.linalg.norm(points[1, :] - points[0, :])
        X, omega = self.get_integration_rule_line(self.p)
        X.shape = (X.shape[0], 1)
        x, y = barycentric_coordinates_line(X)
        XY_phys = points[0, :] * x + points[1, :] * y
        f_vals = f(XY_phys[:, 0], XY_phys[:, 1])
        f_vals.shape = (f_vals.shape[0], 1)
        omega *= J
        shape = self.edge_shape_functions(X.flatten())
        element_vector = (shape * omega) @ f_vals
        element_vector[np.abs(element_vector) < 1e-16] = 0
        return element_vector

def print_matrix(temp):
    print(np.array2string(temp).replace("\n", "").replace("]", "]\n"))
