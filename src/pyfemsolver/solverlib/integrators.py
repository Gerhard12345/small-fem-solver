from numpy.typing import NDArray
import numpy as np
from .coefficientfunction import CoefficientFunction
from .space import H1Space
from .elementtransformation import ElementTransformationLine, ElementTransformationTrig


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
        """
        Assemble the global mass matrix.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param global_mass: The global mass matrix to be assembled
        :type global_mass: NDArray[np.floating]
        :return: None
        """
        for i, trig in enumerate(self.space.tri.trigs):
            print(f"Mass matrix, element {i + 1}/{len(self.space.tri.trigs)}", end="\r")
            trig_coords = np.array([self.space.tri.points[p].coordinates for p in trig.points])
            eltrans = ElementTransformationTrig(trig_coords, trig.region)
            element_matrix = self.space.elements[i].calc_mass_matrix(eltrans, self.coefficient)
            self.space.local_to_global(element_matrix, global_matrix, i)
        print()


class EdgeMass(BilinearFormIntegrator):
    r"""Bilinearform integrator for \int u*v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        """
        Assemble the global boundary mass matrix.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param global_boundary_mass: The global boundary mass matrix to be assembled
        :type global_boundary_mass: NDArray[np.floating]
        :return: None
        """
        for i, edge in enumerate(self.space.tri.boundary_edges):
            print(f"Boundary mass, element {i + 1}/{len(self.space.tri.boundary_edges)}", end="\r")
            edge_coords = np.array([self.space.tri.points[p].coordinates for p in edge.points])
            eltrans = ElementTransformationLine(edge_coords, edge.region)
            element_matrix = self.space.elements[i].calc_edge_mass_matrix(eltrans, self.coefficient)
            self.space.local_to_global_boundary(element_matrix, global_matrix, i)
        print()


class Laplace(BilinearFormIntegrator):
    r"""Bilinearform integrator for \int grad_u*grad_v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        """
        Assemble the global stiffness matrix.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param global_gradu_gradv: The global stiffness matrix to be assembled
        :type global_gradu_gradv: NDArray[np.floating]
        :return: None
        """
        for i, trig in enumerate(self.space.tri.trigs):
            print(f"Stiffness, element {i + 1}/{len(self.space.tri.trigs)}", end="\r")
            trig_coords = np.array([self.space.tri.points[p].coordinates for p in trig.points])
            eltrans = ElementTransformationTrig(trig_coords, trig.region)
            element_matrix = self.space.elements[i].calc_gradu_gradv_matrix(eltrans, self.coefficient)
            self.space.local_to_global(element_matrix, global_matrix, i)
        print()


class Source(LinearFormIntegrator):
    r"""Linearform integrator for \int f*v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        """
        Assemble the global load vector.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param global_vector: The global load vector to be assembled
        :type global_vector: NDArray[np.floating]
        :param f: load function
        :type f: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
        :return: None
        """
        for i, trig in enumerate(self.space.tri.trigs):
            print(f"Load vector, element {i + 1}/{len(self.space.tri.trigs)}", end="\r")
            trig_coords = np.array([self.space.tri.points[p].coordinates for p in trig.points])
            eltrans = ElementTransformationTrig(trig_coords, trig.region)
            element_vector = self.space.elements[i].calc_element_vector(eltrans, self.coefficient)
            self.space.local_to_global_vector(element_vector, global_matrix, i)
        print()


class EdgeSource(LinearFormIntegrator):
    r"""Linearform integrator for \int f*v"""

    def assemble(self, global_matrix: NDArray[np.floating]):
        """
        Assemble the global boundary load vector.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param global_boundary_vector: The global boundary load vector to be assembled
        :type global_boundary_vector: NDArray[np.floating]
        :param f: load function
        :type f: Callable[[NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]]
        :return: None
        """
        for i, edge in enumerate(self.space.tri.boundary_edges):
            print(f"Boundary load vector, element {i + 1}/{len(self.space.tri.boundary_edges)}", end="\r")
            edge_coords = np.array([self.space.tri.points[p].coordinates for p in edge.points])
            eltrans = ElementTransformationLine(edge_coords, edge.region)
            element_vector = self.space.elements[i].calc_edge_element_vector(eltrans, self.coefficient)
            self.space.local_to_global_boundary_vector(element_vector, global_matrix, i)
        print()
