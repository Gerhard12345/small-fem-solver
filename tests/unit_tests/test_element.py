""" "Unit tests for H1 finite element class and related functions."""

import numpy as np

from src.pyfemsolver.solverlib.element import (
    H1Fel,
    barycentric_coordinates,
    barycentric_coordinates_line,
    jacobi_polynomial,
    integrated_jacobi_polynomial,
)
from src.pyfemsolver.solverlib.integrationrules import duffy
from src.pyfemsolver.solverlib.elementtransformation import ElementTransformationTrig, ElementTransformationLine


class TestElement:
    """Unit tests for H1 finite element class and related functions."""

    def setup_method(self):
        """Called before every test method."""

    def teardown_method(self):
        """Called after every test method."""

    def test_constructor(self):
        """Given: an order p=3 element
        When: the element is constructed
        Then: the element fields (p, ndof_*) match expected values
        """
        p = 3
        fel = H1Fel(order=p)
        assert fel.p == p
        assert fel.ndof_vertex == 3
        assert fel.ndof_faces == 3 * (p - 1)
        assert fel.ndof_facet == p - 1
        assert fel.ndof_inner == int((p - 1) * (p - 2) / 2)
        assert fel.ndof == 3 * p + int((p - 2) * (p - 1) / 2)

    def test_flip_edge(self):
        """Given: a second-order element and an edge index
        When: flip_edge is called for that edge
        Then: the edge tuple is reversed
        """
        p = 2
        fel = H1Fel(order=p)
        for edge_nr in (0, 1, 2):
            original_edge = fel.edges[edge_nr]
            fel.flip_edge(edge_nr)
            flipped_edge = fel.edges[edge_nr]
            assert flipped_edge == tuple(reversed(original_edge))

    def test_barycentric_coordinates(self):
        """Given: reference coordinates (x, y) = (0, 0)
        When: barycentric_coordinates is evaluated
        Then: the result sums to 1 and matches expected values
        """
        x = np.array([0.0])
        y = np.array([0.0])
        l = barycentric_coordinates(x, y)
        assert np.allclose(l.sum(axis=0), 1.0)
        assert np.allclose(l[:, 0], np.array([0.25, 0.25, 0.5]))

    def test_barycentric_coordinates_line(self):
        """Given: reference line coordinate t = 0
        When: barycentric_coordinates_line is evaluated
        Then: the result sums to 1 and matches expected values
        """
        t = np.array([0.0])
        l_line = barycentric_coordinates_line(t)
        assert np.allclose(l_line.sum(axis=0), 1.0)
        assert np.allclose(l_line[:, 0], np.array([0.5, 0.5]))

    def test_jacobi_polynomial(self):
        """Given: order 1 and simple input array
        When: jacobi_polynomial is evaluated with alpha=0
        Then: the output shape is correct and values match expected Jacobi polynomials
        """
        x = np.array([0.0, 0.5])
        jp = jacobi_polynomial(1, x, 0)
        # jp[0] == 1, jp[1] == x for alpha=0
        assert jp.shape == (2, 2)
        assert np.allclose(jp[0, :], 1.0)
        assert np.allclose(jp[1, :], x)

    def test_integrated_jacobi_polynomial(self):
        """Given: order 1 and simple input array
        When: integrated_jacobi_polynomial is evaluated with alpha=0
        Then: the second entry matches expected integrated values
        """
        x = np.array([0.0, 0.5])
        ij = integrated_jacobi_polynomial(1, x, 0)
        assert ij.shape[0] >= 2
        assert np.allclose(ij[1, :], x + 1)

    def test_duffy_mapping(self):
        """Given: simple zeta/eta inputs
        When: duffy mapping is applied
        Then: outputs are the expected mapped coordinates
        """
        zeta = np.array([1.0])
        eta = np.array([0.5])
        X_t, Y_t = duffy(zeta, eta)
        assert np.allclose(X_t, 0.5 * 1.0 * (1 - 0.5))
        assert np.allclose(Y_t, eta)

    def test_shape_functions(self):
        """Given: a cubic element at reference point (0, 0)
        When: shape_functions is evaluated
        Then: the first three entries match barycentric coordinates
        """
        fel = H1Fel(order=3)
        X = np.array([0.0])
        Y = np.array([0.0])
        shape = fel.shape_functions(X, Y)
        # vertex shape functions are the barycentric coordinates
        l = barycentric_coordinates(X, Y)
        assert np.allclose(shape[:3, :], l)

    def test_dshape_functions(self):
        """Given: a cubic element at reference point (0, 0)
        When: dshape_functions is evaluated
        Then: the derivative of barycentric coordinates matches expected values
        """
        fel = H1Fel(order=3)
        X = np.array([0.0])
        Y = np.array([0.0])
        dshape = fel.dshape_functions(X, Y)
        # derivative of barycentric coordinates wrt x is [-0.5, 0.5, 0]
        nip = X.size
        dx = dshape[:3, :nip]
        assert np.allclose(dx[:, 0], np.array([-0.5, 0.5, 0.0]), atol=1e-4)

    def test_calc_mass_matrix(self):
        """Given: a second-order element on reference triangle
        When: calc_mass_matrix is computed
        Then: the matrix is symmetric with non-negative diagonal
        """
        fel = H1Fel(order=2)
        points = np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]])
        eltrans = ElementTransformationTrig(points)
        mass = fel.calc_mass_matrix(eltrans)
        assert np.allclose(mass, mass.T)
        assert np.all(np.diag(mass) >= 0)

    def test_calc_gradu_gradv_matrix(self):
        """Given: a second-order element on reference triangle
        When: calc_gradu_gradv_matrix is computed
        Then: the stiffness matrix is symmetric
        """
        fel = H1Fel(order=2)
        points = np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]])
        eltrans = ElementTransformationTrig(points)
        stiff = fel.calc_gradu_gradv_matrix(eltrans)
        assert np.allclose(stiff, stiff.T)

    def test_calc_element_vector(self):
        """Given: a second-order element on reference triangle and constant function f=1
        When: calc_element_vector is computed
        Then: the sum of integrals matches 0.5. (Each linear shape function integrates to 1/3,
        the quadratic ones to -1/6)
        """
        fel = H1Fel(order=2)
        points = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        eltrans = ElementTransformationTrig(points)
        elem_vec = fel.calc_element_vector(eltrans, lambda x, _: np.ones_like(x))
        for i in range(3):
            assert np.allclose(elem_vec[i].sum(), 1.0 / 3.0, atol=1e-6)
        for i in range(3, 6):
            assert np.allclose(elem_vec[i].sum(), -1.0 / 6.0, atol=1e-6)
        assert np.allclose(elem_vec.sum(), 0.5, atol=1e-6)

    def test_calc_edge_mass_matrix(self):
        """Given: an edge of the reference triangle
        When: calc_edge_mass_matrix is computed
        Then: the edge mass matrix is symmetric
        """
        fel = H1Fel(order=2)
        edge_pts = np.array([[-1.0, -1.0], [1.0, -1.0]])
        eltrans = ElementTransformationLine(edge_pts)
        edge_mass = fel.calc_edge_mass_matrix(eltrans)
        assert np.allclose(edge_mass, edge_mass.T)

    def test_calc_edge_element_vector(self):
        """Given: an edge of the reference triangle and constant function f=1
        When: calc_edge_element_vector is computed
        Then: the sum of integrals matches 4/3. The linear shape functions integrate to 1,
        the quadratic one to -2/3.
        """
        fel = H1Fel(order=2)
        edge_pts = np.array([[-1.0, 0.0], [1.0, 0.0]])
        eltrans = ElementTransformationLine(edge_pts)
        edge_vec = fel.calc_edge_element_vector(eltrans, lambda x, _: np.ones_like(x))
        for i in range(2):
            assert np.allclose(edge_vec[i].sum(), 1.0, atol=1e-6)
        assert np.allclose(edge_vec[2].sum(), -2.0 / 3.0, atol=1e-6)
        assert np.allclose(edge_vec.sum(), 4.0 / 3.0, atol=1e-6)

    def test_edge_shape_functions_match_integrated_jacobi(self):
        """Given: an element of order p and a point on the 1D reference line
        When: edge_shape_functions() is called
        Then: the first two entries are barycentric and higher entries match integrated Jacobi
        """
        p = 3
        fel = H1Fel(order=p)
        t = np.array([0.0])
        es = fel.edge_shape_functions(t)
        # first two are barycentric line coords
        bl = barycentric_coordinates_line(t)
        assert np.allclose(es[:2, :], bl)
        # the rest should equal integrated_jacobi_polynomial(p, t, 0)[2:, :]
        ij = integrated_jacobi_polynomial(p, t, 0)[2:, :]
        assert np.allclose(es[2:, :], ij)
