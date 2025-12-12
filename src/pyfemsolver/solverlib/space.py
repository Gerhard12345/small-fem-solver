""" "H1 Finite element space module. Defines the H1Space class for managing finite element spaces, dofs, and assembly."""

from typing import List

import numpy as np
from numpy.typing import NDArray

from .element import H1Fel
from .elementtransformation import ElementTransformationTrig, ElementTransformationLine
from .meshing import Triangulation
from .coefficientfunction import CoefficientFunction


class H1Space:
    """H1 Finite element space class. Manages elements, dofs, and assembly."""

    def __init__(self, tri: Triangulation, p: int):
        self.tri = tri
        self.p = p
        # For each triangle the list contains a Finite element:
        self.elements: List[H1Fel] = [H1Fel(order=p) for _ in range(len(tri.trigs))]
        # The number of vertex dofs is equal to the number of vertices in the triangulation
        self.ndof_vertex = len(tri.points)
        # For each edge there are p-1 edge basis functions, thus the number of edge/face dofs is
        self.ndof_faces = len(tri.edges) * (p - 1)
        # Finally, there are (p-1) * (p-2) / 2 inner basis functions in a triangle,
        # consequently the inner dofs compute as
        self.ndof_inner = int(len(tri.trigs) * (p - 1) * (p - 2) / 2)
        # The total dof number is the sum of different dof types
        self.ndof = self.ndof_vertex + self.ndof_faces + self.ndof_inner
        for i, trig in enumerate(tri.trigs):
            trigpoints = trig.points
            self.elements[i] = H1Fel(order=p)
            # For each local edge store if it needs to be flipped, i.e. store if
            # the global index of the start point is smaller than the global index of the endpoint
            for j, edge in enumerate(self.elements[i].edges):
                if trigpoints[edge[0]] > trigpoints[edge[1]]:
                    self.elements[i].flip_edge(j)

        dofs: List[List[int]] = [[] for _ in range(len(tri.trigs))]
        for i, trig in enumerate(tri.trigs):
            trigpoints = trig.points
            # For each triangle the first dofs are the vertex dofs. "Hat functions"
            # These are numbered according to the vertex number.
            dofs[i].extend([int(p) for p in trigpoints])
            # Next add those dofs associated with the edges of the element. These are functions vanish on all nodes,
            # and edges except for one edge.
            dofs[i].extend(
                [
                    self.ndof_vertex + edge * self.elements[i].ndof_facet + j
                    for edge in tri.trigs[i].edges
                    for j in range(self.elements[i].ndof_facet)
                ]
            )
            # Finally, add the dofs for element bubble functions. These fucntions do not couple to other triangles,
            # they vanish on the complete triangle boundary.
            dofs[i].extend(
                [self.ndof_vertex + self.ndof_faces + i * self.elements[i].ndof_inner + j for j in range(self.elements[i].ndof_inner)]
            )

        boundary_dofs: List[List[int]] = [[]] * len(self.tri.boundary_edges)
        for i, edge in enumerate(tri.boundary_edges):
            neighbour = edge.neighbouring_elements[0]
            boundary_dofs[i] = list(edge.points)
            boundary_dofs[i].extend(
                [
                    self.ndof_vertex + edge.global_edge_nr * self.elements[neighbour].ndof_facet + s
                    for s in range(self.elements[neighbour].ndof_facet)
                ]
            )

        self.boundary_dofs = boundary_dofs  # dofs assoziated with the domain boundary
        self.dofs = dofs  # all dofs
        self.unique_boundary_dofs = sorted(list(set([dof for dofs in self.boundary_dofs for dof in dofs])))
        self.inner_dofs = [i for i in range(self.ndof) if i not in self.unique_boundary_dofs]

    def local_to_global(self, element_matrix: NDArray[np.floating], global_matrix: NDArray[np.floating], trig_index: int):
        """
        Map the local element dofs to the global ones for matrices.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param element_matrix: The local element matrix to be mapped
        :type element_matrix: NDArray[np.floating]
        :param global_matrix: The global matrix to which the local matrix is mapped
        :type global_matrix: NDArray[np.floating]
        :return: None
        """
        dx, dy = np.meshgrid(self.dofs[trig_index], self.dofs[trig_index])
        global_matrix[dy, dx] += element_matrix

    def local_to_global_boundary(self, element_matrix: NDArray[np.floating], global_matrix: NDArray[np.floating], edge_index: int):
        """
        Map the local boundary dofs to the global ones for matrices.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param element_matrix: The local element matrix to be mapped
        :type element_matrix: NDArray[np.floating]
        :param global_matrix: The global matrix to which the local matrix is mapped
        :type global_matrix: NDArray[np.floating]
        :return: None
        """
        dx, dy = np.meshgrid(self.boundary_dofs[edge_index], self.boundary_dofs[edge_index])
        global_matrix[dy, dx] += element_matrix

    def local_to_global_vector(self, element_vector: NDArray[np.floating], global_vector: NDArray[np.floating], trig_index: int):
        """
        Map the local element dofs to the global ones.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param element_vector: The local element vector to be mapped
        :type element_vector: NDArray[np.floating]
        :param global_vector: The global vector to which the local vector is mapped
        :type global_vector: NDArray[np.floating]
        :return: None
        """
        global_vector[self.dofs[trig_index]] += element_vector

    def local_to_global_boundary_vector(self, element_vector: NDArray[np.floating], global_vector: NDArray[np.floating], edge_index: int):
        """
        Map the local boundary dofs to the global ones.

        :param self: H1 finite element space instance
        :type self: H1Space
        :param element_vector: The local element vector to be mapped
        :type element_vector: NDArray[np.floating]
        :param global_vector: The global vector to which the local vector is mapped
        :type global_vector: NDArray[np.floating]
        :return: None
        """
        global_vector[self.boundary_dofs[edge_index]] += element_vector
