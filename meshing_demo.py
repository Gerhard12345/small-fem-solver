import matplotlib.pyplot as plt
import numpy as np
from meshing import generate_points_on_lines
from meshing import generate_inner_points
from meshing import create_delaunay_triangulation
from meshing import iterate_mesh_optimization
from meshing import get_restricted_mesh_size
from meshing import refine_triangulation
from meshing import is_point_on_line_segment

if __name__ == "__main__":
    # Configuration with defined hole regions
    data = {
        "lines": [
            # Region 1's outer boundary
            {"start": [0, 0], "end": [2, 0.5], "left_region": 1, "right_region": 0, "h": 0.1, "boundary_index": 1},
            {"start": [2, 0.5], "end": [2, 1.5], "left_region": 1, "right_region": 2, "h": 0.1, "boundary_index": 1},
            {"start": [2, 1.5], "end": [0, 2], "left_region": 1, "right_region": 0, "h": 0.1, "boundary_index": 1},
            {"start": [0, 2], "end": [0, 0], "left_region": 1, "right_region": 0, "h": 0.1, "boundary_index": 2},

            # Hole in Region 1
            {"start": [0.8, 0.8], "end": [1.2, 0.8], "left_region": 0, "right_region": 1, "h": 0.1, "boundary_index": 3},
            {"start": [1.2, 0.8], "end": [1.2, 1.2], "left_region": 0, "right_region": 1, "h": 0.1, "boundary_index": 3},
            {"start": [1.2, 1.2], "end": [0.8, 1.2], "left_region": 0, "right_region": 1, "h": 0.1, "boundary_index": 3},
            {"start": [0.8, 1.2], "end": [0.8, 0.8], "left_region": 0, "right_region": 1, "h": 0.1, "boundary_index": 3},
            
            # Region 2's outer boundary
            {"start": [2, 0.5], "end": [4, 0], "left_region": 2, "right_region": 0, "h": 0.1, "boundary_index": 1},
            {"start": [4, 0], "end": [4, 2], "left_region": 2, "right_region": 0, "h": 0.1, "boundary_index": 1},
            {"start": [4, 2], "end": [2, 1.5], "left_region": 2, "right_region": 0, "h": 0.1, "boundary_index": 1},

            # Hole in Region 2
            {"start": [2.8, 0.8], "end": [3.2, 0.8], "left_region": 0, "right_region": 2, "h": 0.1, "boundary_index": 3},
            {"start": [3.2, 0.8], "end": [3.2, 1.2], "left_region": 0, "right_region": 2, "h": 0.1, "boundary_index": 3},
            {"start": [3.2, 1.2], "end": [2.8, 1.2], "left_region": 0, "right_region": 2, "h": 0.1, "boundary_index": 3},
            {"start": [2.8, 1.2], "end": [2.8, 0.8], "left_region": 0, "right_region": 2, "h": 0.1, "boundary_index": 3}
        ],
        "regions": [
            {"region_id": 1, "mesh_inner": 0.1},
            {"region_id": 2, "mesh_inner": 0.2}
        ]
    }

    # Generate line points
    points_from_lines, is_inner_point_local = generate_points_on_lines(data["lines"])
    # Generate inner points for each region individually
    points_within_regions = []
    for region in data["regions"]:
        points_within_regions.extend(generate_inner_points(region, data["lines"]))
    # Combine all points
    all_points = np.array(points_from_lines + points_within_regions)
    is_inner_point = [False]* len(points_from_lines) + [True] * len(points_within_regions)
    # Create Delaunay triangulation
    points, valid_triangles = create_delaunay_triangulation(all_points, data["regions"], data["lines"])
    # Perform iterative optimization restricted to inner points
    num_iterations = 15
    for _ in range(num_iterations):
        points = iterate_mesh_optimization(points, valid_triangles, data["regions"], data["lines"], is_inner_point)
    plt.figure()
    # Plot triangulation
    plt.triplot(points[:, 0], points[:, 1], valid_triangles)
    plt.scatter(points[:, 0], points[:, 1], color='r')    
    plt.axis('equal')


    x_coords = np.linspace(0.01, 3.99, 200)
    y_coords = np.linspace(0.01, 1.99, 100)
    mesh_sizes = np.zeros((y_coords.size, x_coords.size))
    max_gradient = 0.2
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            point = [x, y]
            mesh_sizes[j, i] = get_restricted_mesh_size(point, data["lines"], data["regions"], max_gradient)
            
    X, Y = np.meshgrid(x_coords, y_coords)
    # Plotting Mesh Size as a 3D Surface
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, mesh_sizes, cmap='viridis')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Mesh Size')
    ax.set_title('3D Surface Plot of Mesh Size Function')
    points2, simplices2 = refine_triangulation(points, data["regions"], data["lines"], max_gradient)
    is_inner_point = [False]* len(points_from_lines) + [True] * (len(points2) - len(points_from_lines))
    num_iterations = 5
    for _ in range(num_iterations):
        points2 = iterate_mesh_optimization(points2, simplices2, data["regions"], data["lines"], is_inner_point, step_size=0.1)


    edges = list(
        set(
            [
                (int(a), int(b)) if a < b else (int(b), int(a))
                for edge in simplices2
                for a, b in zip(edge, np.roll(edge, -1))
            ]
        )
    )

    is_boundary_edge = [False] * len(edges)
    for i, edge in enumerate(edges):
        for line in data["lines"]:
            if is_point_on_line_segment(points2[edge[0]], line["start"], line["end"], tolerance=1e-4) and is_point_on_line_segment(points2[edge[1]], line["start"], line["end"],tolerance=1e-4):
                if line["left_region"] == 0 or line["right_region"] == 0:
                    is_boundary_edge[i] = True
                break

    plt.figure()
    # Plot triangulation
    plt.triplot(points2[:, 0], points2[:, 1], simplices2)
    plt.scatter(points2[:, 0], points2[:, 1], color='r')
    plt.axis('equal')
    plt.show()