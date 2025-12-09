from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ..solverlib.meshing import generate_points_on_lines
from ..solverlib.meshing import generate_inner_points
from ..solverlib.meshing import create_delaunay_triangulation
from ..solverlib.meshing import iterate_mesh_optimization
from ..solverlib.meshing import get_restricted_mesh_size
from ..solverlib.meshing import refine_triangulation
from ..solverlib.meshing import is_point_on_line_segment
from ..solverlib.geometry import Line, Region, Geometry

if __name__ == "__main__":
    # Configuration with defined hole regions
    lines: List[Line] = []
    lines.append(Line(start=(0, 0), end=(2, 0.5), left_region=1, right_region=0, h=0.1, boundary_index=1))
    lines.append(Line(start=(2, 0.5), end=(2, 1.5), left_region=1, right_region=2, h=0.1, boundary_index=1))
    lines.append(Line(start=(2, 1.5), end=(0, 2), left_region=1, right_region=0, h=0.1, boundary_index=1))
    lines.append(Line(start=(0, 2), end=(0, 0), left_region=1, right_region=0, h=0.1, boundary_index=2))
    # Hole in Region 1
    lines.append(Line(start=(0.8, 0.8), end=(1.2, 0.8), left_region=0, right_region=1, h=0.1, boundary_index=3))
    lines.append(Line(start=(1.2, 0.8), end=(1.2, 1.2), left_region=0, right_region=1, h=0.1, boundary_index=3))
    lines.append(Line(start=(1.2, 1.2), end=(0.8, 1.2), left_region=0, right_region=1, h=0.1, boundary_index=3))
    lines.append(Line(start=(0.8, 1.2), end=(0.8, 0.8), left_region=0, right_region=1, h=0.1, boundary_index=3))
    # Region 2's outer boundary
    lines.append(Line(start=(2, 0.5), end=(4, 0), left_region=2, right_region=0, h=0.1, boundary_index=1))
    lines.append(Line(start=(4, 0), end=(4, 2), left_region=2, right_region=0, h=0.1, boundary_index=1))
    lines.append(Line(start=(4, 2), end=(2, 1.5), left_region=2, right_region=0, h=0.1, boundary_index=1))
    # Hole in Region 2
    lines.append(Line(start=(2.8, 0.8), end=(3.2, 0.8), left_region=0, right_region=2, h=0.1, boundary_index=3))
    lines.append(Line(start=(3.2, 0.8), end=(3.2, 1.2), left_region=0, right_region=2, h=0.1, boundary_index=3))
    lines.append(Line(start=(3.2, 1.2), end=(2.8, 1.2), left_region=0, right_region=2, h=0.1, boundary_index=3))
    lines.append(Line(start=(2.8, 1.2), end=(2.8, 0.8), left_region=0, right_region=2, h=0.1, boundary_index=3))
    regions: List[Region] = []
    regions.append(Region(region_id=1, mesh_inner=0.1))
    regions.append(Region(region_id=2, mesh_inner=0.2))
    geometry = Geometry(lines=lines, regions=regions)

    # Generate line points
    points_from_lines, is_inner_point_local = generate_points_on_lines(geometry)
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
    plt.figure()  # type:ignore
    # Plot triangulation
    plt.triplot(points[:, 0], points[:, 1], valid_triangles)  # type:ignore
    plt.scatter(points[:, 0], points[:, 1], color="r")  # type:ignore
    plt.axis("equal")  # type:ignore

    x_coords = np.linspace(0.01, 3.99, 200)
    y_coords = np.linspace(0.01, 1.99, 100)
    mesh_sizes = np.zeros((y_coords.size, x_coords.size))
    max_gradient = 0.2
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = (x, y)
            mesh_sizes[j, i] = get_restricted_mesh_size(point, geometry, max_gradient)

    X, Y = np.meshgrid(x_coords, y_coords)
    # Plotting Mesh Size as a 3D Surface
    fig = plt.figure(figsize=(10, 6))  # type:ignore
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, mesh_sizes, cmap="viridis")  # type:ignore

    ax.set_xlabel("X Coordinate")  # type:ignore
    ax.set_ylabel("Y Coordinate")  # type:ignore
    ax.set_zlabel("Mesh Size")  # type:ignore
    ax.set_title("3D Surface Plot of Mesh Size Function")  # type:ignore
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

    plt.figure()  # type:ignore
    # Plot triangulation
    plt.triplot(points2[:, 0], points2[:, 1], simplices2)  # type:ignore
    plt.scatter(points2[:, 0], points2[:, 1], color="r")  # type:ignore
    plt.axis("equal")  # type:ignore
    plt.show()  # type:ignore
