from numpy.typing import NDArray
import numpy as np
from .coefficientfunction import CoefficientFunction
from .space import H1Space


class LinearFormIntegrator:
    def __init__(self, coefficient: CoefficientFunction, space: H1Space, is_boundary: bool):
        self.coefficient = coefficient
        self.space = space
        self.is_boundary = is_boundary

    def assemble(self, global_matrix: NDArray[np.floating]) -> None:
        """Compute the global linearform vector"""
        raise NotImplementedError("Method assemble is not implemented within base class integrator")


class BilinearFormIntegrator:
    def __init__(self, coefficient: CoefficientFunction, space: H1Space, is_boundary: bool):
        self.coefficient = coefficient
        self.space = space
        self.is_boundary = is_boundary

    def assemble(self, global_matrix: NDArray[np.floating]) -> None:
        """Compute the global bilinearform matrix"""
        raise NotImplementedError("Method assemble is not implemented within base class integrator")


class Mass(BilinearFormIntegrator):
    r"""Bilinearform integrator for \int u*v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        self.space.assemble_mass(global_matrix, self.coefficient)


class EdgeMass(BilinearFormIntegrator):
    r"""Bilinearform integrator for \int u*v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        self.space.assemble_boundary_mass(global_matrix, self.coefficient)


class Laplace(BilinearFormIntegrator):
    r"""Bilinearform integrator for \int grad_u*grad_v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        self.space.assemble_gradu_gradv(global_matrix, self.coefficient)


class Source(LinearFormIntegrator):
    r"""Linearform integrator for \int f*v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        self.space.assemble_element_vector(global_matrix, self.coefficient)


class EdgeSource(LinearFormIntegrator):
    r"""Linearform integrator for \int f*v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        self.space.assemble_boundary_element_vector(global_matrix, self.coefficient)
