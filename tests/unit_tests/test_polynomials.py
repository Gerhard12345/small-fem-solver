"""Unit tests for polynomial functions in the solverlib module."""

import numpy as np
import pytest

from src.pyfemsolver.solverlib.polynomials import (
    jacobi_polynomial,
    integrated_jacobi_polynomial,
    barycentric_coordinates,
    barycentric_coordinates_line,
)


class TestJacobiPolynomial:
    """Unit tests for jacobi_polynomial function."""

    def test_jacobi_polynomial_order_0(self):
        """Given: order 0 polynomial
        When: jacobi_polynomial is evaluated
        Then: the first entry (constant polynomial) is 1
        """
        x = np.array([0.0, 0.5, 1.0])
        jp = jacobi_polynomial(0, x, 0)
        assert jp.shape == (1, 3)
        assert np.allclose(jp[0, :], 1.0)

    def test_jacobi_polynomial_order_1(self):
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

    def test_jacobi_polynomial_with_different_alpha(self):
        """Given: order 1 polynomial with alpha=1
        When: jacobi_polynomial is evaluated
        Then: the values match the analytical formula
        """
        x = np.array([0.0, 0.5])
        for alpha in (-1, 2, 3):
            jp = jacobi_polynomial(1, x, alpha)
            # For alpha: P_1^(1) = 0.5 * (alpha + (alpha + 2) * x)
            expected = 0.5 * (alpha + (2 + alpha) * x)
            assert np.allclose(jp[1, :], expected)

    def test_jacobi_polynomial_higher_order(self):
        """Given: order 3 polynomial
        When: jacobi_polynomial is evaluated
        Then: shape and basic properties are correct
        """
        x = np.array([0.0, 0.5])
        jp = jacobi_polynomial(3, x, 0)
        assert jp.shape == (4, 2)
        assert np.allclose(jp[0, :], 1.0)  # First polynomial is always 1


class TestIntegratedJacobiPolynomial:
    """Unit tests for integrated_jacobi_polynomial function."""

    def test_integrated_jacobi_polynomial_order_0(self):
        """Given: order 0
        When: integrated_jacobi_polynomial is evaluated
        Then: the first entry is 1
        """
        x = np.array([0.0, 0.5])
        ij = integrated_jacobi_polynomial(0, x, 0)
        assert ij.shape == (1, 2)
        assert np.allclose(ij[0, :], 1.0)

    def test_integrated_jacobi_polynomial_order_1(self):
        """Given: order 1
        When: integrated_jacobi_polynomial is evaluated with alpha=0
        Then: the second entry matches x + 1
        """
        x = np.array([0.0, 0.5])
        ij = integrated_jacobi_polynomial(1, x, 0)
        assert ij.shape == (2, 2)
        assert np.allclose(ij[0, :], 1.0)
        assert np.allclose(ij[1, :], x + 1)

    def test_integrated_jacobi_polynomial_order_2(self):
        """Given: order 2
        When: integrated_jacobi_polynomial is evaluated
        Then: the shape is correct
        """
        x = np.array([0.0, 0.5])
        ij = integrated_jacobi_polynomial(2, x, 0)
        assert ij.shape[0] >= 3
        assert np.allclose(ij[0, :], 1.0)
        assert np.allclose(ij[1, :], x + 1)

    def test_integrated_jacobi_polynomial_with_different_alpha(self):
        """Given: order 2 polynomial with alpha=1
        When: integrated_jacobi_polynomial is evaluated
        Then: shape is correct
        """
        x = np.array([0.0, 0.5])
        ij = integrated_jacobi_polynomial(2, x, 1)
        assert ij.shape[0] >= 3


class TestBarycentricCoordinates:
    """Unit tests for barycentric_coordinates function."""

    def test_barycentric_coordinates_origin(self):
        """Given: reference coordinates (x, y) = (0, 0)
        When: barycentric_coordinates is evaluated
        Then: the result sums to 1 and matches expected values
        """
        x = np.array([0.0])
        y = np.array([0.0])
        l = barycentric_coordinates(x, y)
        assert np.allclose(l.sum(axis=0), 1.0)
        assert np.allclose(l[:, 0], np.array([0.25, 0.25, 0.5]))

    def test_barycentric_coordinates_vertex(self):
        """Given: reference coordinates at vertex (-1, -1)
        When: barycentric_coordinates is evaluated
        Then: the first coordinate is 1, others are 0
        """
        x = np.array([-1.0])
        y = np.array([-1.0])
        l = barycentric_coordinates(x, y)
        assert np.allclose(l[:, 0], np.array([1.0, 0.0, 0.0]), atol=1e-10)

    def test_barycentric_coordinates_another_vertex(self):
        """Given: reference coordinates at vertex (1, -1)
        When: barycentric_coordinates is evaluated
        Then: the second coordinate is 1, others are 0
        """
        x = np.array([1.0])
        y = np.array([-1.0])
        l = barycentric_coordinates(x, y)
        assert np.allclose(l[:, 0], np.array([0.0, 1.0, 0.0]), atol=1e-10)

    def test_barycentric_coordinates_third_vertex(self):
        """Given: reference coordinates at vertex (0, 1)
        When: barycentric_coordinates is evaluated
        Then: the third coordinate is 1, others are 0
        """
        x = np.array([0.0])
        y = np.array([1.0])
        l = barycentric_coordinates(x, y)
        assert np.allclose(l[:, 0], np.array([0.0, 0.0, 1.0]), atol=1e-10)

    def test_barycentric_coordinates_multiple_points(self):
        """Given: multiple reference coordinates
        When: barycentric_coordinates is evaluated
        Then: each point sums to 1
        """
        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([-1.0, 0.0, -1.0])
        l = barycentric_coordinates(x, y)
        assert np.allclose(l.sum(axis=0), 1.0)


class TestBarycentricCoordinatesLine:
    """Unit tests for barycentric_coordinates_line function."""

    def test_barycentric_coordinates_line_origin(self):
        """Given: reference line coordinate t = 0
        When: barycentric_coordinates_line is evaluated
        Then: the result sums to 1 and matches expected values
        """
        t = np.array([0.0])
        l_line = barycentric_coordinates_line(t)
        assert np.allclose(l_line.sum(axis=0), 1.0)
        assert np.allclose(l_line[:, 0], np.array([0.5, 0.5]))

    def test_barycentric_coordinates_line_left_vertex(self):
        """Given: reference line coordinate t = -1
        When: barycentric_coordinates_line is evaluated
        Then: the first coordinate is 1, the second is 0
        """
        t = np.array([-1.0])
        l_line = barycentric_coordinates_line(t)
        assert np.allclose(l_line[:, 0], np.array([1.0, 0.0]), atol=1e-10)

    def test_barycentric_coordinates_line_right_vertex(self):
        """Given: reference line coordinate t = 1
        When: barycentric_coordinates_line is evaluated
        Then: the first coordinate is 0, the second is 1
        """
        t = np.array([1.0])
        l_line = barycentric_coordinates_line(t)
        assert np.allclose(l_line[:, 0], np.array([0.0, 1.0]), atol=1e-10)

    def test_barycentric_coordinates_line_multiple_points(self):
        """Given: multiple reference line coordinates
        When: barycentric_coordinates_line is evaluated
        Then: each point sums to 1
        """
        t = np.array([-1.0, 0.0, 1.0])
        l_line = barycentric_coordinates_line(t)
        assert np.allclose(l_line.sum(axis=0), 1.0)
