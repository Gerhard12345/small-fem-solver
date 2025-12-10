"""
Element transformation module for finite element analysis. Implements ElementTransformation
and its subclasses for triangular and line elements. Classes can provide Jacobian, its determinant,
and inverse.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from .polynomials import barycentric_coordinates_line, barycentric_coordinates


class ElementTransformation:
    """
    Base class for element transformations in finite element analysis.
    Has no knowledge about the geometric type.
    """

    def __init__(self, points: NDArray[np.floating], region_index: int):
        self.points = points  # array of point coordinates
        self.region_index = region_index
        self.J: NDArray[np.floating]

    def getjacobian(self):
        """
        Get the Jacobian matrix of the element transformation.

        :return: Jacobian matrix
        :rtype: NDArray[np.floating]
        """
        return self.J

    def getjacobian_determinant(self):
        """
        Get the determinant of the Jacobian matrix.

        :return: Determinant of the Jacobian matrix
        :rtype: np.floating
        """
        J = self.getjacobian()
        return np.abs(np.linalg.det(J))

    def get_jacobian_inverse(self):
        """
        Get the inverse of the Jacobian matrix.

        :return: Inverse of the Jacobian matrix
        :rtype: NDArray[np.floating]
        """
        J = self.getjacobian()
        return np.linalg.inv(J)


class ElementTransformationTrig(ElementTransformation):
    """Triangular element transformation class."""

    def __init__(self, points: NDArray[np.floating], region_index: int):
        super().__init__(points, region_index)
        self.J = np.array(
            [
                0.5 * (points[1, :] - points[0, :]),
                0.25 * (2 * points[2] - points[1] - points[0]),
            ]
        )

    def transform_points(
        self, ref_points_x: NDArray[np.floating], ref_points_y: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Transform reference coordinates to physical ones

        :param ref_points_x: x values of reference coordinates
        :type ref_points_x: NDArray[np.floating]
        :param ref_points_y: y values of reference coordinates
        :type ref_points_y: NDArray[np.floating]
        :return: transformed points in physical domain
        :rtype: Tuple[NDArray[np.floating], NDArray[np.floating]]
        """
        x, y, z = barycentric_coordinates(ref_points_x, ref_points_y)
        x_phys = self.points[0, 0] * x + self.points[1, 0] * y + self.points[2, 0] * z
        y_phys = self.points[0, 1] * x + self.points[1, 1] * y + self.points[2, 1] * z
        return x_phys, y_phys


class ElementTransformationLine(ElementTransformation):
    """Line element transformation class."""

    def __init__(self, points: NDArray[np.float64], region_index: int):
        super().__init__(points, region_index)
        self.J = np.array(np.linalg.norm(0.5 * points[1, :] - 0.5 * points[0, :]))

    def getjacobian_determinant(self):
        return np.abs(self.J)

    def transform_points(self, ref_points_x: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Transform reference coordinates to physical ones

        :param ref_points_x: x values of reference coordinates
        :type ref_points_x: NDArray[np.floating]
        :return: transformed points in physical domain
        :rtype: Tuple[NDArray[np.floating], NDArray[np.floating]]
        """
        x, y = barycentric_coordinates_line(ref_points_x)
        x_phys = self.points[0, 0] * x + self.points[1, 0] * y
        y_phys = self.points[0, 1] * x + self.points[1, 1] * y
        return x_phys, y_phys
