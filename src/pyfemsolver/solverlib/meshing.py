"""
Meshing module for generating and optimizing triangular meshes based on geometric definitions.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay  # pylint:disable=E0611 # type:ignore

from .geometry import Geometry, Line, Region


@dataclass
class Point:
    """
    Class representing a point in the mesh via its coordiantes.
    Provides an indicator if the point is a boundary point.
    """

    coordinates: NDArray[np.float64]
    is_boundary_point: bool


@dataclass
class Triangle:
    """
    Represents a triangle element in the mesh via its point and edge indices.
    """

    points: Tuple[int, int, int]
    edges: Tuple[int, int, int]


@dataclass
class Edge:
    """
    Represents an edge in the mesh via its point indices and neighbouring elements.
    Provides indicators for boundary edges.
    """

    points: Tuple[int, int]
    neighbouring_elements: List[int]
    is_boundary_edge: bool
    global_edge_nr: int


@dataclass
class Triangulation:
    """
    The triangulation data structure containing points, edges, triangles, and boundary information.
    """

    points: List[Point]
    boundary_points: List[Point]
    edges: List[Edge]
    trigs: List[Triangle]
    boundary_edges: List[Edge]


def generate_points_on_lines(geometry: Geometry, tolerance: float = 1e-6) -> Tuple[List[Tuple[float, float]], List[bool]]:
    """
    Generate points along the lines of the geometry with specified mesh sizes. Also generates an indicator list for inner points.

    :param geometry: Geometry object containing lines and regions.
    :type geometry: Geometry
    :param tolerance: Tolerance for point uniqueness, defaults to 1e-6
    :type tolerance: float, optional
    :return: Tuple containing list of points and corresponding inner point indicators.
    :rtype: Tuple[List[Tuple[float, float]], List[bool]]
    """
    points: List[NDArray[np.floating]] = []
    is_inner_point: List[bool] = []
    for line in geometry.lines:
        start = np.array(line.start)
        end = np.array(line.end)
        length = np.linalg.norm(end - start)
        num_points = int(round(length / line.h))
        line_points = [start + (end - start) * i / num_points for i in range(num_points + 1)]
        points.extend(line_points)
        local_is_inner_points = [line.left_region != 0 and line.right_region != 0] * len(line_points)
        is_inner_point.extend(local_is_inner_points)
        # Remove duplicates within a numerical tolerance. group points according to equivalence classes within rounding tolerance.
        # Store for each group all potential different values for "is_boundary". This can arise from a point being in multiple lines
        # forming a t crossing:
        # ------------------P--------------------
        #     l_1           |      l_2
        #                   |
        #                   |
        #                   |
        #                   |  l_3
        #                   |
        #
        # P has 3 different is_inner_point indicators. for l_1 and l_2 it is said to be a boundary point, for line l_3
        # it is an inner point (since the line is an inner line).
        # If P is in at least one outside line, then it is said to be an outside point.
    groups: Dict[Tuple[float, float], List[bool]] = {}
    for el, is_local_inner_point in zip(points, is_inner_point):
        if tuple(np.round(el, decimals=int(-np.log10(tolerance)))) in groups:
            groups[tuple(np.round(el, decimals=int(-np.log10(tolerance))))].append(is_local_inner_point)
        else:
            groups[tuple(np.round(el, decimals=int(-np.log10(tolerance))))] = [is_local_inner_point]
    reduced_is_inner_point = [min(vals) for vals in groups.values()]
    return list(groups.keys()), reduced_is_inner_point


def is_point_on_line_segment(point: Tuple[float, float], line: Line, tolerance: float = 1e-6) -> bool:
    """
    Verifes if a point lies on a given line segment within a specified tolerance.

    :param point: Coordinates of the point to check.
    :type point: Tuple[float, float]
    :param line: Line segment to check against.
    :type line: Line
    :param tolerance: Tolerance for point-line distance, defaults to 1e-6
    :type tolerance: float, optional
    :return: True if the point lies on the line segment within the tolerance, False otherwise.
    :rtype: bool
    """
    # Check if a point lies on the line segment within a given tolerance
    line_vec = np.array(line.end) - np.array(line.start)
    point_vec = np.array(point) - np.array(line.start)
    line_length = np.linalg.norm(line_vec)
    projection = np.dot(point_vec, line_vec) / line_length
    closest_point_on_line = np.array(line.start) + line_vec * projection / line_length
    dist_to_line = np.linalg.norm(point - closest_point_on_line)

    return dist_to_line <= tolerance and -tolerance <= projection <= line_length + tolerance  # type:ignore


def generate_inner_points(region: Region, lines: List[Line], tolerance: float = 1e-6) -> List[Tuple[float, float]]:
    """
    Generate inner points for a given region based on its mesh size. Does not place points on the boundary lines of the region.

    :param region: Region object defining the area for point generation.
    :type region: Region
    :param lines: List of Line objects defining the boundaries of the region.
    :type lines: List[Line]
    :param tolerance: Tolerance for point-line distance, defaults to 1e-6
    :type tolerance: float, optional
    :return: List of tuples representing the coordinates of the generated inner points.
    :rtype: List[Tuple[float, float]]
    """
    points: List[Tuple[float, float]] = []
    region_id = region.region_id
    mesh_size = region.mesh_inner
    # Determine bounds from lines for the region
    min_x = min(line.start[0] for line in lines if region_id in (line.left_region, line.right_region))
    max_x = max(line.end[0] for line in lines if region_id in (line.left_region, line.right_region))
    min_y = min(line.start[1] for line in lines if region_id in (line.left_region, line.right_region))
    max_y = max(line.end[1] for line in lines if region_id in (line.left_region, line.right_region))

    # Iterate through the region bounds using calculated meshes
    x = min_x
    while x <= max_x:
        y = min_y + 0.5 * mesh_size  # Adjust y-offset for stagger
        j = 0
        while y <= max_y:
            # stagger by half mesh size for alternating rows
            adjusted_x = x + (0.5 * mesh_size if j % 2 else 0)
            point = (adjusted_x, y)

            # Check if the point is within the region and not on a line
            on_any_line = any(is_point_on_line_segment(point, line, tolerance) for line in lines)
            if not on_any_line and point_in_region(point, lines, region_id):
                points.append(point)

            y += mesh_size
            j += 1
        x += mesh_size

    return points


def create_delaunay_triangulation(points: NDArray[np.floating], geometry: Geometry) -> Tuple[NDArray[np.floating], List[List[int]]]:
    """
    Creates a Delaunay triangulation from given points and filters triangles based on the geometry regions, i.e.
    only keeping triangles whose centroids lie within the geometry.

    :param points: Array of points to triangulate.
    :type points: NDArray[np.floating]
    :param geometry: Geometry object defining regions for filtering triangles.
    :type geometry: Geometry
    :return: Tuple containing the array of points and a list of valid triangles (as lists of point indices).
    :rtype: Tuple[NDArray[np.floating], List[List[int]]]
    """
    delaunay = Delaunay(points)
    valid_triangles: List[List[int]] = []

    for simplex in delaunay.simplices:
        centroid = np.mean(points[simplex], axis=0)
        region_id = get_region_for_centroid(centroid, geometry)

        if region_id != 0:  # Exclude triangles whose centroids lie outside the defined regions
            valid_triangles.append(simplex)

    return np.array(points), valid_triangles


def point_in_region(point: Tuple[float, float], lines: List[Line], region_id: int) -> bool:
    """
    Determine if a point is within a specific region using line information.

    :param point: Coordinates of the point to check.
    :type point: Tuple[float, float]
    :param lines: List of Line objects defining the boundaries of the regions.
    :type lines: List[Line]
    :param region_id: ID of the region to check against.
    :type region_id: int
    :return: True if the point is within the specified region, False otherwise.
    :rtype: bool
    """
    x, y = point
    crossings = 0
    for line in lines:
        start = np.array(line.start)
        end = np.array(line.end)

        if region_id in (line.left_region, line.right_region):
            if ((start[1] <= y < end[1]) or (end[1] <= y < start[1])) and (
                x < (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0]
            ):
                crossings += 1

    return crossings % 2 == 1


def get_region_for_centroid(centroid: Tuple[float, float], geometry: Geometry) -> int:
    """
    Determine the region ID for a given centroid based on the geometry.

    :param centroid: Coordinates of the centroid.
    :type centroid: Tuple[float, float]
    :param geometry: Geometry object containing regions and lines.
    :type geometry: Geometry
    :return: Region ID where the centroid is located, or 0 if outside all regions.
    :rtype: int
    """
    for region in geometry.regions:
        if point_in_region(centroid, geometry.lines, region.region_id):
            return region.region_id
    return 0


def side_length_gradient(points: NDArray[np.floating], simplices: List[List[int]]) -> NDArray[np.floating]:
    """
    Calculate the gradient based on deviation of triangle side lengths.

    :param points: Array of point coordinates.
    :type points: NDArray[np.floating]
    :param simplices: List of triangles defined by point indices.
    :type simplices: List[List[int]]
    :return: Gradient array based on side length deviations.
    :rtype: NDArray[np.floating]
    """
    points = points.reshape(-1, 2)
    gradients = np.zeros_like(points)

    for simplex in simplices:
        coords = points[simplex]
        lengths = np.array([np.linalg.norm(coords[i] - coords[(i + 1) % 3]) for i in range(3)])
        avg_length = sum(lengths) / 3

        for i in range(3):
            # Gradient contribution based on length deviation
            gradient_contribution = (lengths[i] - avg_length) * (coords[i] - coords[(i + 1) % 3]) / lengths[i]
            gradients[simplex[i]] += gradient_contribution

    return gradients.flatten()


def iterate_mesh_optimization(
    points: NDArray[np.floating], simplices: List[List[int]], geometry: Geometry, inner_point_indicator: List[bool], step_size: float = 0.1
) -> NDArray[np.floating]:
    """
    Performs a single iteration of mesh point optimization based on side lengths with constraints.

    :param points: Array of point coordinates.
    :type points: NDArray[np.floating]
    :param simplices: List of triangles defined by point indices.
    :type simplices: List[List[int]]
    :param geometry: Geometry object containing regions and lines.
    :type geometry: Geometry
    :param inner_point_indicator: List indicating which points are inner points.
    :type inner_point_indicator: List[bool]
    :param step_size: Step size for the optimization, defaults to 0.1
    :type step_size: float, optional
    :return: Updated array of point coordinates after optimization.
    :rtype: NDArray[np.floating]
    """
    points_flattened = points.flatten()
    gradient = side_length_gradient(points_flattened, simplices)
    gradient = gradient.reshape(-1, 2)
    updated_points = points.copy()
    for i, point in enumerate(points):
        for region in geometry.regions:
            if inner_point_indicator[i]:
                potential_point = point - step_size * gradient[i]
                if point_in_region(potential_point, geometry.lines, region.region_id):
                    updated_points[i] = potential_point
    return updated_points


def point_to_line_distance(point: Tuple[float, float], line: Line) -> float:
    """
    Calculate the shortest distance from a point to a line segment.

    :param point: Coordinates of the point.
    :type point: Tuple[float, float]
    :param line: Line segment to calculate distance to.
    :type line: Line
    :return: Shortest distance from the point to the line segment.
    :rtype: float
    """
    line_vec = np.array(line.end) - np.array(line.start)
    point_vec = np.array(point) - np.array(line.start)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    t = max(0, min(1, t))
    nearest = line_vec * t
    dist = np.linalg.norm(nearest - point_vec)
    return float(dist)


def get_restricted_mesh_size(point: Tuple[float, float], geometry: Geometry, max_gradient: float) -> float:
    """
    Calculate the restricted mesh size at a given point based on its distance to the nearest boundary line and the maximum mesh sizegradient.

    :param point: Coordinates of the point.
    :type point: Tuple[float, float]
    :param geometry: Geometry object containing regions and lines.
    :type geometry: Geometry
    :param max_gradient: Maximum allowed gradient for mesh size changes.
    :type max_gradient: float
    :return: Calculated mesh size at the point.
    :rtype: float
    """
    region_id = get_region_for_centroid(point, geometry)
    if region_id == 0:
        return 0.001

    region_info = next(region for region in geometry.regions if region.region_id == region_id)
    inner_size = region_info.mesh_inner

    min_distance = float("inf")
    boundary_size = 1.0

    for line in geometry.lines:
        if region_id in (line.left_region, line.right_region):
            dist = point_to_line_distance(point, line)
            if dist < min_distance:
                min_distance = dist
                boundary_size = line.h

    weight = min(1, max_gradient / max(min_distance, 1e-12))
    mesh_size = weight * boundary_size + (1 - weight) * inner_size

    return mesh_size


def refine_triangulation(points: NDArray[np.floating], geometry: Geometry, max_gradient: float) -> Tuple[NDArray[np.floating], List[List[int]]]:
    """
    Refine triangulation based on mesh size restrictions.

    :param points: Array of point coordinates.
    :type points: NDArray[np.floating]
    :param geometry: Geometry object containing regions and lines.
    :type geometry: Geometry
    :param max_gradient: Maximum allowed gradient for mesh size changes.
    :type max_gradient: float
    :return: Tuple containing the refined array of point coordinates and a list of valid triangles (as lists of point indices).
    :rtype: Tuple[NDArray[np.floating], List[List[int]]]
    """
    points, simplices = create_delaunay_triangulation(points, geometry)
    max_iterations = 5
    iteration = 0
    while iteration < max_iterations:
        centroids = np.mean(points[simplices], axis=1)
        max_edge_lengths = np.max(
            [np.linalg.norm(points[simplices][:, i] - points[simplices][:, (i + 1) % 3], axis=1) for i in range(3)], axis=0
        )
        mesh_sizes = np.array([get_restricted_mesh_size(c, geometry, max_gradient) for c in centroids]) * 1.4
        needs_refinement = max_edge_lengths > mesh_sizes
        if not np.any(needs_refinement):
            break
        for simplex in np.array(simplices)[needs_refinement]:
            new_point = np.mean(points[simplex], axis=0)
            points = np.vstack([points, new_point])
        points, simplices = create_delaunay_triangulation(points, geometry)
        iteration += 1
    return points, simplices


def generate_mesh(geometry: Geometry, max_gradient: float = 0.05) -> Triangulation:
    """
    Compute a triangular mesh based on the provided geometry and maximum mesh size gradient.
    Generates points on lines and within regions, creates a Delaunay triangulation, and optimizes the mesh.
    Refines the triangulation based on mesh size restrictions and optimizes again.

    :param geometry: Geometry object defining the area to mesh.
    :type geometry: Geometry
    :param max_gradient: Maximum allowed gradient for mesh size changes, defaults to 0.05
    :type max_gradient: float, optional
    :return: Triangulation object containing points, edges, triangles, and boundary information.
    :rtype: Triangulation
    """
    # Generate line points
    points_from_lines, is_inner_point_lines = generate_points_on_lines(geometry)
    # Generate inner points for each region individually
    points_within_regions: List[Tuple[float, float]] = []
    for region in geometry.regions:
        points_within_regions.extend(generate_inner_points(region, geometry.lines))
    # Combine all points
    all_points = np.array(points_from_lines + points_within_regions)
    is_inner_point = [False] * len(points_from_lines) + [True] * len(points_within_regions)
    # Create Delaunay triangulation
    points, valid_triangles = create_delaunay_triangulation(all_points, geometry)
    # Perform iterative optimization restricted to inner points
    num_iterations = 15
    for _ in range(num_iterations):
        points = iterate_mesh_optimization(points, valid_triangles, geometry, is_inner_point)
    points2, simplices2 = refine_triangulation(points, geometry, max_gradient)
    is_inner_point = [False] * len(points_from_lines) + [True] * (len(points2) - len(points_from_lines))
    num_iterations = 5
    for _ in range(num_iterations):
        points2 = iterate_mesh_optimization(points2, simplices2, geometry, is_inner_point, step_size=0.1)

    edges = list(set([(int(a), int(b)) if a < b else (int(b), int(a)) for edge in simplices2 for a, b in zip(edge, np.roll(edge, -1))]))

    is_boundary_edge = [False] * len(edges)
    for i, edge in enumerate(edges):
        for line in geometry.lines:
            if is_point_on_line_segment(points2[edge[0]], line, tolerance=1e-4) and is_point_on_line_segment(
                points2[edge[1]], line, tolerance=1e-4
            ):
                if line.left_region == 0 or line.right_region == 0:
                    is_boundary_edge[i] = True
                break

    is_not_boundary = is_inner_point_lines + [True] * (len(points2) - len(points_from_lines))
    is_boundary = [not ip for ip in is_not_boundary]

    edge_definitions: List[Edge] = [None] * len(edges)  # type:ignore
    trig_definitions: List[Triangle] = []
    found_edges: List[int] = []
    for i, trig in enumerate(simplices2):
        # and map them to the global points
        trig_local_edges = [tuple(sorted((int(a), int(b)))) for a, b in zip(trig, np.roll(trig, -1))]
        # self.trig_edges stores for each triangle the index of the edges
        # in the global set of edges, i.e. self.trig_edges[i] = [a,b,c] means
        # the i-th triangle consists of the global edges with indices a,b and c.
        for edge in trig_local_edges:
            edge_nr = edges.index(edge)  # type:ignore
            if edge_nr in found_edges:
                first_neighbour = edge_definitions[edge_nr].neighbouring_elements[0]
                edge_definitions[edge_nr].neighbouring_elements = [first_neighbour, i]
            else:
                found_edges.append(edge_nr)
                edge_definitions[edge_nr] = Edge(edge, [i], is_boundary_edge=is_boundary_edge[edge_nr], global_edge_nr=edge_nr)  # type:ignore
        trig_definitions.append(
            Triangle(points=tuple(trig.astype(int).tolist()), edges=tuple(edges.index(edge) for edge in trig_local_edges))  # type:ignore
        )

    boundary_edges = [edge for edge in edge_definitions if edge.is_boundary_edge]

    point_definitions: List[Point] = []
    for i, point in enumerate(points2):
        point_definitions.append(Point(coordinates=tuple(point), is_boundary_point=is_boundary[i]))  # type:ignore

    boundary_points = [point for point in point_definitions if point.is_boundary_point]

    t = Triangulation(
        points=point_definitions, trigs=trig_definitions, edges=edge_definitions, boundary_edges=boundary_edges, boundary_points=boundary_points
    )
    return t
