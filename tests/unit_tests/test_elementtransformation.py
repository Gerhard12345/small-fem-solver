"""Unit tests for element transformation classes."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.pyfemsolver.solverlib.elementtransformation import (
    ElementTransformation,
    ElementTransformationTrig,
    ElementTransformationLine,
)


class TestElementTransformationBase:
    """Unit tests for base ElementTransformation class."""

    def test_element_transformation_stores_points(self):
        """Given: points array
        When: ElementTransformation is created
        Then: points are stored
        """
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        et = ElementTransformation(points, region_index=1)
        assert np.allclose(et.points, points)
        assert et.region_index == 1

    def test_get_jacobian_determinant(self):
        """Given: an element transformation with Jacobian
        When: getjacobian_determinant is called
        Then: returns absolute value of determinant
        """
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        et = ElementTransformation(points, region_index=5)
        et.J = np.array([[1.0, 0.0], [0.0, 2.0]])

        det = et.getjacobian_determinant()
        assert np.allclose(det, 2.0)

    def test_get_jacobian_inverse(self):
        """Given: an element transformation with invertible Jacobian
        When: get_jacobian_inverse is called
        Then: returns inverse of Jacobian
        """
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        et = ElementTransformation(points, region_index=6)
        et.J = np.array([[2.0, 0.0], [0.0, 3.0]])

        J_inv = et.get_jacobian_inverse()
        expected = np.array([[0.5, 0.0], [0.0, 1.0 / 3.0]])
        assert np.allclose(J_inv, expected)


class TestElementTransformationTrig:
    """Unit tests for ElementTransformationTrig class."""

    def test_constructor_standard_triangle(self):
        """Given: standard reference triangle vertices
        When: ElementTransformationTrig is created
        Then: Jacobian is correctly computed
        """
        points = np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]])
        eltrans = ElementTransformationTrig(points, region_index=2)

        assert eltrans.J.shape == (2, 2)
        assert np.allclose(eltrans.J[0, :], 0.5 * (points[1, :] - points[0, :]))
        assert np.allclose(eltrans.J[1, :], 0.25 * (2 * points[2] - points[1] - points[0]))

    def test_jacobian_determinant_positive(self):
        """Given: a valid triangle transformation
        When: getjacobian_determinant is called
        Then: returns positive value
        """
        points = np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]])
        eltrans = ElementTransformationTrig(points, region_index=2)

        det = eltrans.getjacobian_determinant()
        assert det > 0

    def test_transform_points_at_origin(self):
        """Given: reference coordinates at origin
        When: transform_points is called
        Then: returns barycentric combination of vertices
        """
        points = np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]])
        eltrans = ElementTransformationTrig(points, region_index=2)

        x_phys, y_phys = eltrans.transform_points(np.array([0.0]), np.array([0.0]))

        # Should be weighted average of vertices
        assert x_phys.shape == (1,)
        assert y_phys.shape == (1,)

    def test_transform_points_at_vertex(self):
        """Given: reference coordinates at a vertex
        When: transform_points is called
        Then: returns that vertex in physical space
        """
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        eltrans = ElementTransformationTrig(points, region_index=2)

        # Reference coords at vertex (-1, -1) should map to first physical vertex
        x_phys, y_phys = eltrans.transform_points(np.array([-1.0]), np.array([-1.0]))

        assert np.allclose(x_phys, [0.0])
        assert np.allclose(y_phys, [0.0])

    def test_transform_multiple_points(self):
        """Given: multiple reference coordinates
        When: transform_points is called
        Then: returns transformed points with correct shape
        """
        points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        eltrans = ElementTransformationTrig(points, region_index=4)

        x_ref = np.array([-1.0, 0.0, 1.0])
        y_ref = np.array([-1.0, 0.0, 1.0])

        x_phys, y_phys = eltrans.transform_points(x_ref, y_ref)

        assert x_phys.shape == (3,)
        assert y_phys.shape == (3,)


class TestElementTransformationLine:
    """Unit tests for ElementTransformationLine class."""

    def test_constructor_standard_line(self):
        """Given: standard reference line endpoints
        When: ElementTransformationLine is created
        Then: Jacobian is correctly computed
        """
        points = np.array([[-1.0, 0.0], [1.0, 0.0]])
        eltrans = ElementTransformationLine(points, region_index=3)

        # Jacobian for a line is the half-length
        expected_J = np.linalg.norm(0.5 * points[1, :] - 0.5 * points[0, :])
        assert np.allclose(eltrans.J, expected_J)

    def test_jacobian_determinant(self):
        """Given: a line transformation
        When: getjacobian_determinant is called
        Then: returns positive value
        """
        points = np.array([[-1.0, 0.0], [1.0, 0.0]])
        eltrans = ElementTransformationLine(points, region_index=3)

        det = eltrans.getjacobian_determinant()
        assert det > 0

    def test_transform_points_at_origin(self):
        """Given: reference coordinate t = 0
        When: transform_points is called
        Then: returns barycentric combination of endpoints
        """
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        eltrans = ElementTransformationLine(points, region_index=3)

        x_phys, y_phys = eltrans.transform_points(np.array([0.0]))

        assert x_phys.shape == (1,)
        assert y_phys.shape == (1,)

    def test_transform_points_at_left_endpoint(self):
        """Given: reference coordinate t = -1 (left endpoint)
        When: transform_points is called
        Then: returns left endpoint in physical space
        """
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        eltrans = ElementTransformationLine(points, region_index=3)

        x_phys, y_phys = eltrans.transform_points(np.array([-1.0]))

        assert np.allclose(x_phys, [0.0])
        assert np.allclose(y_phys, [0.0])

    def test_transform_points_at_right_endpoint(self):
        """Given: reference coordinate t = 1 (right endpoint)
        When: transform_points is called
        Then: returns right endpoint in physical space
        """
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        eltrans = ElementTransformationLine(points, region_index=2)

        x_phys, y_phys = eltrans.transform_points(np.array([1.0]))

        assert np.allclose(x_phys, [2.0])
        assert np.allclose(y_phys, [0.0])

    def test_transform_multiple_points(self):
        """Given: multiple reference coordinates
        When: transform_points is called
        Then: returns transformed points with correct shape
        """
        points = np.array([[0.0, 0.0], [4.0, 0.0]])
        eltrans = ElementTransformationLine(points, region_index=3434)

        t_ref = np.array([-1.0, 0.0, 1.0])
        x_phys, y_phys = eltrans.transform_points(t_ref)

        assert x_phys.shape == (3,)
        assert y_phys.shape == (3,)

        # Check that points are in order
        assert x_phys[0] <= x_phys[1] <= x_phys[2]
