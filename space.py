import numpy as np
from element import H1Fel
from typing import List
from meshing import Triangulation


class H1Space:
    def __init__(self, tri:Triangulation, p):
        self.tri = tri
        self.p = p
        # For each triangle the list contains a Finite element:
        self.elements: List[H1Fel] = [None] * len(tri.trigs)
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

        dofs = [[] for _ in range(len(tri.trigs))]
        for i, trig in enumerate(tri.trigs):
            trigpoints = trig.points
            # For each triangle the first dofs are the vertex dofs. 
            # These are numbered according to the vertex number.
            dofs[i].extend([int(p) for p in trigpoints])
            # Next add those dofs associated with the edges of the element. The global, first edge dof
            # has the number of vertex dofs as offset, since they come before. Then for each edge, we get a free range of dofs
            # by counting additionally offsetting with the dof numbers for a single edge (self.elements[i].ndof_facet) times
            # the global edge number (compare to linear indexing a 2d point cloud - taking ndof_vertex into account, 
            # it can even be compared by a linear counting of a 3d point cloud)
            dofs[i].extend(
                [
                    self.ndof_vertex + edge * self.elements[i].ndof_facet + j
                    for edge in tri.trigs[i].edges
                    for j in range(self.elements[i].ndof_facet)
                ]
            )
            # Finally, add the dofs for element bubble functions. The offset now is all vertex plus all edge dofs. Then for the i-th triangle we
            # find a free spot, by offsetting additionally with the triangle number times the number of inner dofs (self.elements[i].ndof_inner)
            dofs[i].extend(
                [self.ndof_vertex + self.ndof_faces + i * self.elements[i].ndof_inner + j for j in range(self.elements[i].ndof_inner)]
            )

        boundary_dofs = [[]] * len(self.tri.boundary_edges)
        for i, edge in enumerate(tri.boundary_edges):
            neighbour = edge.neighbouring_elements[0]
            boundary_dofs[i] = list(edge.points)
            boundary_dofs[i].extend([self.ndof_vertex + edge.global_edge_nr * self.elements[neighbour].ndof_facet + s for s in range(self.elements[neighbour].ndof_facet)])


        self.boundary_dofs = boundary_dofs  # dofs assoziated with the domain boundary
        self.dofs = dofs  # all dofs
        self.unique_boundary_dofs = sorted(list(set([dof for dofs in self.boundary_dofs for dof in dofs])))
        self.inner_dofs = [i for i in range(self.ndof) if i not in self.unique_boundary_dofs]

    def local_to_global(self, element_matrix, global_matrix, trig_index:int):
        """Map the local element matrix to the global one."""
        dx, dy = np.meshgrid(self.dofs[trig_index], self.dofs[trig_index])
        global_matrix[dy, dx] += element_matrix

    def local_to_global_boundary(self, element_matrix, global_matrix, edge_index):
        """Map the local boundary element matrix to the global one."""
        dx, dy = np.meshgrid(self.boundary_dofs[edge_index], self.boundary_dofs[edge_index])
        global_matrix[dy, dx] += element_matrix

    def local_to_global_vector(self, element_vector, global_vector, trig_index):
        global_vector[self.dofs[trig_index]] += element_vector

    def local_to_global_boundary_vector(self, element_vector, global_vector, edge_index):
        global_vector[self.boundary_dofs[edge_index]] += element_vector

    def assemble_mass(self, global_mass):
        for i, trig in enumerate(self.tri.trigs):
            trig_coords = np.array([self.tri.points[p].coordinates for p in trig.points])
            element_matrix = self.elements[i].calc_mass_matrix(trig_coords)
            self.local_to_global(element_matrix, global_mass, i)

    def assemble_gradu_gradv(self, global_gradu_gradv):
        for i, trig in enumerate(self.tri.trigs):
            trig_coords = np.array([self.tri.points[p].coordinates for p in trig.points])
            element_matrix = self.elements[i].calc_gradu_gradv_matrix(trig_coords)
            self.local_to_global(element_matrix, global_gradu_gradv, i)

    def assemble_element_vector(self, global_vector, f):
        for i, trig in enumerate(self.tri.trigs):
            trig_coords = np.array([self.tri.points[p].coordinates for p in trig.points])
            element_vector = self.elements[i].calc_element_vector(trig_coords, f)
            self.local_to_global_vector(element_vector, global_vector, i)

    def assemble_boundary_mass(self, global_boundary_mass):
        for i, edge in enumerate(self.tri.boundary_edges):
            edge_coords = np.array([self.tri.points[p].coordinates for p in edge.points])
            element_matrix = self.elements[i].calc_edge_mass_matrix(edge_coords)
            self.local_to_global_boundary(element_matrix, global_boundary_mass, i)

    def assemble_boundary_element_vector(self, global_boundary_vector, f):
        for i, edge in enumerate(self.tri.boundary_edges):
            edge_coords = np.array([self.tri.points[p].coordinates for p in edge.points])
            element_vector = self.elements[i].calc_edge_element_vector(edge_coords, f)
            self.local_to_global_boundary_vector(element_vector, global_boundary_vector, i)

