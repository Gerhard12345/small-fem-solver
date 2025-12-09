import numpy as np
from numpy.typing import NDArray


class ElementTransformation:
    def __init__(self, points: NDArray[np.floating]):
        self.points = points  # array of point coordinates
        self.J: NDArray[np.floating]

    def getjacobian(self):
        return self.J

    def getjacobian_determinant(self):
        J = self.getjacobian()
        return np.abs(np.linalg.det(J))

    def get_jacobian_inverse(self):
        J = self.getjacobian()
        return np.linalg.inv(J)


class ElementTransformationTrig(ElementTransformation):
    def __init__(self, points: NDArray[np.floating]):
        super().__init__(points)
        self.J = np.array(
            [
                0.5 * (points[1, :] - points[0, :]),
                0.25 * (2 * points[2] - points[1] - points[0]),
            ]
        )


class ElementTransformationLine(ElementTransformation):
    def __init__(self, points: NDArray[np.float64]):
        super().__init__(points)
        self.J = np.array(np.linalg.norm(0.5 * points[1, :] - 0.5 * points[0, :]))

    def getjacobian_determinant(self):
        return np.abs(self.J)
