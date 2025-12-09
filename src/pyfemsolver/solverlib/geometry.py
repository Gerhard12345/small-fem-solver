"""
Geometry module defining classes for 2D geometric entities used in meshing.
"""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Line:
    """
    Class representing a line in 2d geometry.

    Attributes:
        left_region (int): ID of the region to the left of the line.
        right_region (int): ID of the region to the right of the line.
        start (Tuple[float, float]): Starting coordinates of the line.
        end (Tuple[float, float]): Ending coordinates of the line.
        h (float): Desired mesh size along the line.
        boundary_index (int): Index identifying the boundary condition type.
    """

    left_region: int
    right_region: int
    start: Tuple[float, float]
    end: Tuple[float, float]
    h: float
    boundary_index: int


@dataclass
class Region:
    """
    Class representing a regions property in a 2d geometry.

    Attributes:
        region_id (int): Unique identifier for the region.
        mesh_inner (float): Desired mesh size inside the region.
    """

    region_id: int
    mesh_inner: float


@dataclass
class Geometry:
    """
    Class representing a 2d geometry.

    Attributes:
        lines (List[Line]): List of lines defining the geometry.
        regions (List[Region]): List of regions in the geometry.
    """

    lines: List[Line]
    regions: List[Region]
