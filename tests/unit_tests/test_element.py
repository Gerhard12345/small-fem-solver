"""Unit tests for H1 finite element class."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, Mock

from src.pyfemsolver.solverlib.element import H1Fel


class TestH1FelConstructor:
    """Unit tests for H1Fel constructor."""

    def test_constructor_order_1(self):
        """Given: order p=1 element
        When: the element is constructed
        Then: the element fields match expected values
        """
        p = 1
        fel = H1Fel(order=p)
        assert fel.p == p
        assert fel.ndof_vertex == 3
        assert fel.ndof_faces == 0
        assert fel.ndof_facet == 0
        assert fel.ndof_inner == 0
        assert fel.ndof == 3

    def test_constructor_order_2(self):
        """Given: order p=2 element
        When: the element is constructed
        Then: the element fields match expected values
        """
        p = 2
        fel = H1Fel(order=p)
        assert fel.p == p
        assert fel.ndof_vertex == 3
        assert fel.ndof_faces == 3 * (p - 1)
        assert fel.ndof_facet == p - 1
        assert fel.ndof_inner == int((p - 1) * (p - 2) / 2)
        assert fel.ndof == 3 * p + int((p - 2) * (p - 1) / 2)

    def test_constructor_order_3(self):
        """Given: order p=3 element
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

    def test_constructor_edges(self):
        """Given: any order element
        When: the element is constructed
        Then: edges are initialized correctly
        """
        fel = H1Fel(order=2)
        assert fel.edges == [(0, 1), (1, 2), (2, 0)]


class TestH1FelEdgeFlipping:
    """Unit tests for H1Fel edge flipping."""

    def test_flip_edge_first(self):
        """Given: an element and first edge
        When: flip_edge is called for edge 0
        Then: the edge tuple is reversed
        """
        fel = H1Fel(order=2)
        original_edge = fel.edges[0]
        fel.flip_edge(0)
        assert fel.edges[0] == tuple(reversed(original_edge))

    def test_flip_edge_second(self):
        """Given: an element and second edge
        When: flip_edge is called for edge 1
        Then: the edge tuple is reversed
        """
        fel = H1Fel(order=2)
        original_edge = fel.edges[1]
        fel.flip_edge(1)
        assert fel.edges[1] == tuple(reversed(original_edge))

    def test_flip_edge_third(self):
        """Given: an element and third edge
        When: flip_edge is called for edge 2
        Then: the edge tuple is reversed
        """
        fel = H1Fel(order=2)
        original_edge = fel.edges[2]
        fel.flip_edge(2)
        assert fel.edges[2] == tuple(reversed(original_edge))

    def test_flip_edge_multiple_times(self):
        """Given: an element with flipped edge
        When: flip_edge is called again on the same edge
        Then: the edge returns to original orientation
        """
        fel = H1Fel(order=2)
        original_edge = fel.edges[0]
        fel.flip_edge(0)
        fel.flip_edge(0)
        assert fel.edges[0] == original_edge


class TestH1FelShapeFunctions:
    """Unit tests for H1Fel shape functions."""

    @patch("src.pyfemsolver.solverlib.element.barycentric_coordinates")
    @patch("src.pyfemsolver.solverlib.element.edge_based_polynomials")
    @patch("src.pyfemsolver.solverlib.element.h")
    def test_shape_functions_order_1(self, mock_h, mock_edge, mock_bary):
        """Given: a first-order element
        When: shape_functions is called
        Then: returns only vertex shape functions
        """
        mock_bary.return_value = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        fel = H1Fel(order=1)
        X = np.array([0.0, 0.5])
        Y = np.array([0.0, 0.5])

        shape = fel.shape_functions(X, Y)

        assert shape.shape == (3, 2)
        mock_bary.assert_called_once()
        mock_edge.assert_not_called()
        mock_h.assert_not_called()

    @patch("src.pyfemsolver.solverlib.element.barycentric_coordinates")
    @patch("src.pyfemsolver.solverlib.element.edge_based_polynomials")
    @patch("src.pyfemsolver.solverlib.element.h")
    def test_shape_functions_order_2(self, mock_h, mock_edge, mock_bary):
        """Given: a second-order element
        When: shape_functions is called
        Then: returns vertex and edge shape functions
        """
        mock_bary.return_value = np.array([[0.5], [0.25], [0.25]])
        mock_edge.return_value = np.array([[0.2]])

        fel = H1Fel(order=2)
        X = np.array([0.0])
        Y = np.array([0.0])

        shape = fel.shape_functions(X, Y)

        assert shape.shape == (6, 1)
        mock_edge.assert_called()

    @patch("src.pyfemsolver.solverlib.element.barycentric_coordinates")
    @patch("src.pyfemsolver.solverlib.element.edge_based_polynomials")
    @patch("src.pyfemsolver.solverlib.element.h")
    def test_shape_functions_order_3(self, mock_h, mock_edge, mock_bary):
        """Given: a cubic element
        When: shape_functions is called
        Then: calls edge and bubble helper functions
        """
        mock_bary.return_value = np.array([[0.25], [0.25], [0.5]])
        mock_edge.return_value = np.array([[0.1], [0.3]])
        mock_h.return_value = [np.array([[0.05]]), np.array([[0.02]])]

        fel = H1Fel(order=3)
        X = np.array([0.0])
        Y = np.array([0.0])
        shape = fel.shape_functions(X, Y)

        assert shape.shape[1] == 1
        mock_h.assert_called()


class TestH1FelDerivatives:
    """Unit tests for H1Fel derivative functions."""

    @patch("src.pyfemsolver.solverlib.element.barycentric_coordinates")
    @patch("src.pyfemsolver.solverlib.element.edge_based_polynomials")
    @patch("src.pyfemsolver.solverlib.element.h")
    def test_dshape_functions_order_1(self, mock_h, mock_edge, mock_bary):
        """Given: a first-order element
        When: dshape_functions is called
        Then: returns derivatives only for vertex functions
        """
        mock_bary.return_value = np.array([[0.5], [0.25], [0.25]])

        fel = H1Fel(order=1)
        X = np.array([0.0])
        Y = np.array([0.0])

        dshape = fel.dshape_functions(X, Y)

        # 3 functions, 2 directions (dx and dy)
        assert dshape.shape[0] == 3
        assert dshape.shape[1] == 2  # 2 nip * 2 directions

    @patch("src.pyfemsolver.solverlib.element.barycentric_coordinates")
    @patch("src.pyfemsolver.solverlib.element.edge_based_polynomials")
    @patch("src.pyfemsolver.solverlib.element.h")
    def test_dshape_functions_order_2(self, mock_h, mock_edge, mock_bary):
        """Given: a second-order element
        When: dshape_functions is called
        Then: returns derivatives for vertex and edge functions
        """
        mock_bary.return_value = np.array([[0.5], [0.25], [0.25]])
        mock_edge.return_value = np.array([[0.2]])

        fel = H1Fel(order=2)
        X = np.array([0.0])
        Y = np.array([0.0])

        dshape = fel.dshape_functions(X, Y)

        # 6 functions, 2 directions
        assert dshape.shape == (6, 2)


class TestH1FelMatrices:
    """Unit tests for H1Fel matrix computation methods."""

    @patch("src.pyfemsolver.solverlib.element.get_integration_rule_trig")
    def test_calc_mass_matrix_symmetry(self, mock_rule):
        """Given: a second-order element
        When: calc_mass_matrix is computed
        Then: the matrix is symmetric
        """
        # Mock integration rule
        X = np.array([[0.0], [1.0], [-1.0]])
        Y = np.array([[0.0], [0.0], [0.0]])
        omega = np.array([0.1, 0.2, 0.2])
        mock_rule.return_value = (X, Y, omega)

        fel = H1Fel(order=2)

        # Create mock element transformation
        mock_eltrans = MagicMock()
        mock_eltrans.getjacobian_determinant.return_value = 1.0

        with patch.object(fel, "shape_functions") as mock_shape:
            # Return dummy shape functions
            mock_shape.return_value = np.ones((6, 3))

            mass = fel.calc_mass_matrix(mock_eltrans)

            assert np.allclose(mass, mass.T)

    @patch("src.pyfemsolver.solverlib.element.get_integration_rule_trig")
    def test_calc_gradu_gradv_matrix_symmetry(self, mock_rule):
        """Given: a second-order element
        When: calc_gradu_gradv_matrix is computed
        Then: the stiffness matrix is symmetric
        """
        X = np.array([[0.0], [1.0], [-1.0]])
        Y = np.array([[0.0], [0.0], [0.0]])
        omega = np.array([0.1, 0.2, 0.2])
        mock_rule.return_value = (X, Y, omega)

        fel = H1Fel(order=2)

        mock_eltrans = MagicMock()
        mock_eltrans.getjacobian_determinant.return_value = 1.0
        mock_eltrans.get_jacobian_inverse.return_value = np.eye(2)

        with patch.object(fel, "dshape_functions") as mock_dshape:
            mock_dshape.return_value = np.ones((6, 6))

            stiff = fel.calc_gradu_gradv_matrix(mock_eltrans)

            assert np.allclose(stiff, stiff.T)

    @patch("src.pyfemsolver.solverlib.element.get_integration_rule_trig")
    def test_calc_element_vector(self, mock_rule):
        """Given: a second-order element and a function
        When: calc_element_vector is computed
        Then: returns vector with correct shape
        """
        X = np.array([[0.0], [1.0], [-1.0]])
        Y = np.array([[0.0], [0.0], [0.0]])
        omega = np.array([0.1, 0.2, 0.2])
        mock_rule.return_value = (X, Y, omega)

        fel = H1Fel(order=2)

        mock_eltrans = MagicMock()
        mock_eltrans.getjacobian_determinant.return_value = 1.0
        mock_eltrans.transform_points.return_value = (X, Y)

        with patch.object(fel, "shape_functions") as mock_shape:
            mock_shape.return_value = np.ones((6, 3))

            f = lambda x, y: np.ones_like(x)
            elem_vec = fel.calc_element_vector(mock_eltrans, f)

            assert elem_vec.shape == (6, 1)


class TestH1FelEdgeFunctions:
    """Unit tests for H1Fel edge-related functions."""

    @patch("src.pyfemsolver.solverlib.element.barycentric_coordinates_line")
    @patch("src.pyfemsolver.solverlib.element.integrated_jacobi_polynomial")
    def test_edge_shape_functions(self, mock_ij, mock_bary_line):
        """Given: an element and a point on the reference line
        When: edge_shape_functions is called
        Then: returns shape functions with correct shape
        """
        mock_bary_line.return_value = np.array([[0.5], [0.5]])
        mock_ij.return_value = np.array([[1.0], [0.5], [0.25]])

        fel = H1Fel(order=3)
        t = np.array([0.0])

        es = fel.edge_shape_functions(t)

        # Should have p+1 = 4 shape functions
        assert es.shape == (4, 1)

    @patch("src.pyfemsolver.solverlib.element.get_integration_rule_line")
    def test_calc_edge_mass_matrix_symmetry(self, mock_rule):
        """Given: a second-order element edge
        When: calc_edge_mass_matrix is computed
        Then: the matrix is symmetric
        """
        X = np.array([[0.0], [1.0]])
        omega = np.array([1.0, 1.0])
        mock_rule.return_value = (X, omega)

        fel = H1Fel(order=2)

        mock_eltrans = MagicMock()
        mock_eltrans.getjacobian_determinant.return_value = 1.0

        with patch.object(fel, "edge_shape_functions") as mock_shape:
            mock_shape.return_value = np.ones((3, 2))

            edge_mass = fel.calc_edge_mass_matrix(mock_eltrans)

            assert np.allclose(edge_mass, edge_mass.T)

    @patch("src.pyfemsolver.solverlib.element.get_integration_rule_line")
    def test_calc_edge_element_vector(self, mock_rule):
        """Given: a second-order element edge and a function
        When: calc_edge_element_vector is computed
        Then: returns vector with correct shape
        """
        X = np.array([[0.0], [1.0]])
        omega = np.array([1.0, 1.0])
        mock_rule.return_value = (X, omega)

        fel = H1Fel(order=2)

        mock_eltrans = MagicMock()
        mock_eltrans.getjacobian_determinant.return_value = 1.0
        mock_eltrans.transform_points.return_value = (X, X)

        with patch.object(fel, "edge_shape_functions") as mock_shape:
            mock_shape.return_value = np.ones((3, 2))

            f = lambda x, y: np.ones_like(x)
            edge_vec = fel.calc_edge_element_vector(mock_eltrans, f)

            assert edge_vec.shape == (3, 1)
