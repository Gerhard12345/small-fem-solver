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


class DoubleSlitGeometry(Geometry):
    """
    Class representing a double slit geometry.

    Attributes:
        lines (List[Line]): List of lines defining the geometry.
        regions (List[Region]): List of regions in the geometry.
    """
    def __init__(self, slit_width: float=2.4, slit_height: float=0.6, slit_distance: float=4.0, domain_size: float=12.0):
        """
        Initializes a DoubleSlitGeometry instance with 2 domains.

        Args:
            slit_width (float): Width of each slit.
            slit_height (float): Height of each slit.
            slit_distance (float): Distance between the centers of the two slits.
            inner_domain_size (float): Size of the (inner) square domain containing the slits.
            outer_domain_size (float): Size of the (outer) square domain containing the slits.
        """
        height = slit_height  # pylint:disable=C0103
        width = slit_width  # pylint:disable=C0103
        center_x = [0, 0]
        center_y = [-slit_distance/2, slit_distance/2]
        lines: List[Line] = []
        lines.append(Line(start=(-domain_size/2, -domain_size/2), end=(domain_size/2, -domain_size/2), left_region=1, right_region=0, h=0.5, boundary_index=1))
        lines.append(Line(start=(domain_size/2, -domain_size/2), end=(domain_size/2, domain_size/2), left_region=1, right_region=0, h=0.5, boundary_index=1))
        lines.append(Line(start=(domain_size/2, domain_size/2), end=(-domain_size/2, domain_size/2), left_region=1, right_region=0, h=0.5, boundary_index=1))
        lines.append(Line(start=(-domain_size/2, domain_size/2), end=(-domain_size/2, -domain_size/2), left_region=1, right_region=0, h=0.5, boundary_index=1))
        # Plate 1
        lines.append(
            Line(
                start=(center_x[0] - width * 0.5, center_y[0] - height * 0.5),
                end=(center_x[0] + width * 0.5, center_y[0] - height * 0.5),
                left_region=0,
                right_region=1,
                h=0.2,
                boundary_index=2,
            )
        )
        lines.append(
            Line(
                start=(center_x[0] + width * 0.5, center_y[0] - height * 0.5),
                end=(center_x[0] + width * 0.5, center_y[0] + height * 0.5),
                left_region=0,
                right_region=1,
                h=0.2,
                boundary_index=2,
            )
        )
        lines.append(
            Line(
                start=(center_x[0] + width * 0.5, center_y[0] + height * 0.5),
                end=(center_x[0] - width * 0.5, center_y[0] + height * 0.5),
                left_region=0,
                right_region=1,
                h=0.2,
                boundary_index=2,
            )
        )
        lines.append(
            Line(
                start=(center_x[0] - width * 0.5, center_y[0] + height * 0.5),
                end=(center_x[0] - width * 0.5, center_y[0] - height * 0.5),
                left_region=0,
                right_region=1,
                h=0.2,
                boundary_index=2,
            )
        )
        # Plate 2
        lines.append(
            Line(
                start=(center_x[1] - width * 0.5, center_y[1] - height * 0.5),
                end=(center_x[1] + width * 0.5, center_y[1] - height * 0.5),
                left_region=0,
                right_region=1,
                h=0.2,
                boundary_index=3,
            )
        )
        lines.append(
            Line(
                start=(center_x[1] + width * 0.5, center_y[1] - height * 0.5),
                end=(center_x[1] + width * 0.5, center_y[1] + height * 0.5),
                left_region=0,
                right_region=1,
                h=0.2,
                boundary_index=3,
            )
        )
        lines.append(
            Line(
                start=(center_x[1] + width * 0.5, center_y[1] + height * 0.5),
                end=(center_x[1] - width * 0.5, center_y[1] + height * 0.5),
                left_region=0,
                right_region=1,
                h=0.2,
                boundary_index=3,
            )
        )
        lines.append(
            Line(
                start=(center_x[1] - width * 0.5, center_y[1] + height * 0.5),
                end=(center_x[1] - width * 0.5, center_y[1] - height * 0.5),
                left_region=0,
                right_region=1,
                h=0.2,
                boundary_index=3,
            )
        )
        self.lines = lines
        self.regions = [Region(region_id=1, mesh_inner=0.5)]


class SingleSlitGeometryWith3Domains(Geometry):
    """
    Class representing a double slit geometry with inner, center and outer domain.

    Attributes:
        lines (List[Line]): List of lines defining the geometry.
        regions (List[Region]): List of regions in the geometry.
    """
    def __init__(self, slit_width: float=2.0, slit_height: float=2.0, inner_domain_size: float=8.0, intermediate_domain_size: float = 16.0, outer_domain_size: float=20.0):
        """
        Initializes a DoubleSlitGeometry instance.

        Args:
            slit_width (float): Width of each slit.
            slit_height (float): Height of each slit.
            slit_distance (float): Distance between the centers of the two slits.
            domain_size (float): Size of the square domain containing the slits.
        """
        height = slit_height  # pylint:disable=C0103
        width = slit_width  # pylint:disable=C0103
        center_x = 0
        center_y = 0
        lines: List[Line] = []
        lines.append(Line(start=(-outer_domain_size/2, -outer_domain_size/2), end=(outer_domain_size/2, -outer_domain_size/2), left_region=3, right_region=0, h=0.5, boundary_index=1))
        lines.append(Line(start=(outer_domain_size/2, -outer_domain_size/2), end=(outer_domain_size/2, outer_domain_size/2), left_region=3, right_region=0, h=0.5, boundary_index=1))
        lines.append(Line(start=(outer_domain_size/2, outer_domain_size/2), end=(-outer_domain_size/2, outer_domain_size/2), left_region=3, right_region=0, h=0.5, boundary_index=1))
        lines.append(Line(start=(-outer_domain_size/2, outer_domain_size/2), end=(-outer_domain_size/2, -outer_domain_size/2), left_region=3, right_region=0, h=0.5, boundary_index=1))

        lines.append(Line(start=(-intermediate_domain_size/2, -intermediate_domain_size/2), end=(intermediate_domain_size/2, -intermediate_domain_size/2), left_region=2, right_region=3, h=0.5, boundary_index=1))
        lines.append(Line(start=(intermediate_domain_size/2, -intermediate_domain_size/2), end=(intermediate_domain_size/2, intermediate_domain_size/2), left_region=2, right_region=3, h=0.5, boundary_index=1))
        lines.append(Line(start=(intermediate_domain_size/2, intermediate_domain_size/2), end=(-intermediate_domain_size/2, intermediate_domain_size/2), left_region=2, right_region=3, h=0.5, boundary_index=1))
        lines.append(Line(start=(-intermediate_domain_size/2, intermediate_domain_size/2), end=(-intermediate_domain_size/2, -intermediate_domain_size/2), left_region=2, right_region=3, h=0.5, boundary_index=1))

        lines.append(Line(start=(-inner_domain_size/2, -inner_domain_size/2), end=(inner_domain_size/2, -inner_domain_size/2), left_region=1, right_region=2, h=0.5, boundary_index=1))
        lines.append(Line(start=(inner_domain_size/2, -inner_domain_size/2), end=(inner_domain_size/2, inner_domain_size/2), left_region=1, right_region=2, h=0.5, boundary_index=1))
        lines.append(Line(start=(inner_domain_size/2, inner_domain_size/2), end=(-inner_domain_size/2, inner_domain_size/2), left_region=1, right_region=2, h=0.5, boundary_index=1))
        lines.append(Line(start=(-inner_domain_size/2, inner_domain_size/2), end=(-inner_domain_size/2, -inner_domain_size/2), left_region=1, right_region=2, h=0.5, boundary_index=1))


        # Plate 1
        lines.append(
            Line(
                start=(center_x - width * 0.5, center_y - height * 0.5),
                end=(center_x + width * 0.5, center_y - height * 0.5),
                left_region=0,
                right_region=1,
                h=0.5,
                boundary_index=2,
            )
        )
        lines.append(
            Line(
                start=(center_x + width * 0.5, center_y - height * 0.5),
                end=(center_x + width * 0.5, center_y + height * 0.5),
                left_region=0,
                right_region=1,
                h=0.5,
                boundary_index=2,
            )
        )
        lines.append(
            Line(
                start=(center_x + width * 0.5, center_y + height * 0.5),
                end=(center_x - width * 0.5, center_y + height * 0.5),
                left_region=0,
                right_region=1,
                h=0.5,
                boundary_index=2,
            )
        )
        lines.append(
            Line(
                start=(center_x - width * 0.5, center_y + height * 0.5),
                end=(center_x - width * 0.5, center_y - height * 0.5),
                left_region=0,
                right_region=1,
                h=0.5,
                boundary_index=2,
            )
        )

        self.lines = lines
        self.regions = [Region(region_id=1, mesh_inner=0.5), Region(region_id=2, mesh_inner=0.75), Region(region_id=3, mesh_inner=0.5)]