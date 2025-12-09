from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay  # pylint:disable=E0611 # type:ignore

from .geometry import Geometry, Line, Region


@dataclass
class Point:
    coordinates: NDArray[np.float64]
    is_boundary_point: bool


@dataclass
class Triangle:
    points: Tuple[int]
    edges: Tuple[int]


@dataclass
class Edge:
    points: Tuple[int, int]
    neighbouring_elements: List[int]
    is_boundary_edge: bool
    global_edge_nr: int


@dataclass
class Triangulation:
    points: List[Point]
    boundary_points: List[Point]
    edges: List[Edge]
    trigs: List[Triangle]
    boundary_edges: List[Edge]


def generate_points_on_lines(geometry: Geometry, tolerance: float = 1e-6) -> Tuple[List[Tuple[float, float]], List[bool]]:
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
    # Store for each group all potential different values for "is_boundary". This can arise from a point being in multiple lines, forming a t crossing:
    # ------------------P--------------------
    #     l_1           |      l_2
    #                   |
    #                   |
    #                   |
    #                   |  l_3
    #                   |
    #
    # P has 3 different is_inner_point indicators. for l_1 and l_2 it is said to be a boundary point, for line l_3 it is an inner point (since the line is an inner line).
    # If P is in at least one outside line, then it is said to be an outside point.
    groups: Dict[Tuple[float, float], List[bool]] = {}
    for el, is_local_inner_point in zip(points, is_inner_point):
        if tuple(np.round(el, decimals=int(-np.log10(tolerance)))) in groups:
            groups[tuple(np.round(el, decimals=int(-np.log10(tolerance))))].append(is_local_inner_point)
        else:
            groups[tuple(np.round(el, decimals=int(-np.log10(tolerance))))] = [is_local_inner_point]
    reduced_is_inner_point = [min(vals) for vals in groups.values()]
    return list(groups.keys()), reduced_is_inner_point


def is_point_on_line_segment(point: Tuple[float, float], line: Line, tolerance: float = 1e-6):
    # Check if a point lies on the line segment within a given tolerance
    line_vec = np.array(line.end) - np.array(line.start)
    point_vec = np.array(point) - np.array(line.start)
    line_length = np.linalg.norm(line_vec)
    projection = np.dot(point_vec, line_vec) / line_length
    closest_point_on_line = np.array(line.start) + line_vec * projection / line_length
    dist_to_line = np.linalg.norm(point - closest_point_on_line)

    return dist_to_line <= tolerance and -tolerance <= projection <= line_length + tolerance


def generate_inner_points(region: Region, lines: List[Line], tolerance: float = 1e-6):
    points: List[Tuple[float, float]] = []
    region_id = region.region_id
    mesh_size = region.mesh_inner
    # Determine bounds from lines for the region
    min_x = min(line.start[0] for line in lines if line.left_region == region_id or line.right_region == region_id)
    max_x = max(line.end[0] for line in lines if line.left_region == region_id or line.right_region == region_id)
    min_y = min(line.start[1] for line in lines if line.left_region == region_id or line.right_region == region_id)
    max_y = max(line.end[1] for line in lines if line.left_region == region_id or line.right_region == region_id)

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
    delaunay = Delaunay(points)
    valid_triangles: List[List[int]] = []

    for simplex in delaunay.simplices:
        centroid = np.mean(points[simplex], axis=0)
        region_id = get_region_for_centroid(centroid, geometry)

        if region_id != 0:  # Exclude triangles whose centroids lie outside the defined regions
            valid_triangles.append(simplex)

    return np.array(points), valid_triangles


def point_in_region(point: Tuple[float, float], lines: List[Line], region_id: int) -> bool:
    """Determine if a point is within a specific region using line information."""
    x, y = point
    crossings = 0
    for line in lines:
        start = np.array(line.start)
        end = np.array(line.end)

        if line.right_region == region_id or line.left_region == region_id:
            if ((start[1] <= y < end[1]) or (end[1] <= y < start[1])) and (
                x < (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0]
            ):
                crossings += 1

    return crossings % 2 == 1


def get_region_for_centroid(centroid: Tuple[float, float], geometry: Geometry) -> int:
    for region in geometry.regions:
        if point_in_region(centroid, geometry.lines, region.region_id):
            return region.region_id
    return 0


def side_length_gradient(points: NDArray[np.floating], simplices: List[List[int]]) -> NDArray[np.floating]:
    """Calculate the gradient based on deviation of triangle side lengths."""
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
    """Performs a single iteration of mesh point optimization based on side lengths with constraints."""
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
    region_id = get_region_for_centroid(point, geometry)
    if region_id == 0:
        return 0.001

    region_info = next(region for region in geometry.regions if region.region_id == region_id)
    inner_size = region_info.mesh_inner

    min_distance = float("inf")
    boundary_size = 1.0

    for line in geometry.lines:
        if line.left_region == region_id or line.right_region == region_id:
            dist = point_to_line_distance(point, line)
            if dist < min_distance:
                min_distance = dist
                boundary_size = line.h

    weight = min(1, max_gradient / max(min_distance, 1e-12))
    mesh_size = weight * boundary_size + (1 - weight) * inner_size

    return mesh_size


def refine_triangulation(points: NDArray[np.floating], geometry: Geometry, max_gradient: float) -> Tuple[NDArray[np.floating], List[List[int]]]:
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
