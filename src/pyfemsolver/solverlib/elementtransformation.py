"""
Element transformation module for finite element analysis. Implements ElementTransformation
and its subclasses for triangular and line elements. Classes can provide Jacobian, its determinant,
and inverse.
"""

import numpy as np
from numpy.typing import NDArray


class ElementTransformation:
    """
    Base class for element transformations in finite element analysis.
    Has no knowledge about the geometric type.
    """

    def __init__(self, points: NDArray[np.floating]):
        self.points = points  # array of point coordinates
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

    def __init__(self, points: NDArray[np.floating]):
        super().__init__(points)
        self.J = np.array(
            [
                0.5 * (points[1, :] - points[0, :]),
                0.25 * (2 * points[2] - points[1] - points[0]),
            ]
        )


class ElementTransformationLine(ElementTransformation):
    """Line element transformation class."""

    def __init__(self, points: NDArray[np.float64]):
        super().__init__(points)
        self.J = np.array(np.linalg.norm(0.5 * points[1, :] - 0.5 * points[0, :]))

    def getjacobian_determinant(self):
        return np.abs(self.J)
