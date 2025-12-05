import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import LineString
from shapely.ops import triangulate


def uniform_sample(poly, n=100):
    """
    Uniformly sample the Delaunay triangulation of a polygon. If the polygon
    is convex, this will uniformly sample its area.
    Parameters
    ----------
    poly: Shapely Polygon
    n: Number of points
    Returns
    -------
    [n x 2] numpy array of x-y coordinates that are uniformly distributed
    over the Delaunay triangulation.
    """
    polys = triangulate(poly)
    # Normalize the areas
    areas = np.array([p.area for p in polys])
    areas /= areas.sum()

    # Randomly select a triangle weighted by area
    # t_inds is the index of the chosen triangle
    t_inds = np.searchsorted(np.cumsum(areas), np.random.random(n))

    # Randomly sample the area of each triangle according to
    # P = (1-sqrt(r1))A + (sqrt(r1)(1-r2))B + (sqrt(r1)r2)C
    # where r1, r2 are sampled uniform [0, 1] and A, B, C are the triangle
    # vertices
    # http://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle

    # Compute the coefficients
    sr1 = np.sqrt(np.random.random(n))  # sr1 is sqrt(r1) above
    r2 = np.random.random(n)
    c0 = 1 - sr1
    c1 = sr1 * (1 - r2)
    c2 = sr1 * r2

    # Grab the triangle vertices.
    # v is a 3-element list where each element is [len(polys) x 2]
    # array of each triangle vertex. It represents, A, B, C above.
    v = [np.array([p.exterior.coords[i] for p in polys]) for i in range(3)]

    # Compute the points. v[i] is [N x 2] and the coefficients are [N x 1]
    P = c0[:, np.newaxis] * v[0][t_inds, :] + c1[:, np.newaxis] * v[1][t_inds, :] + c2[:, np.newaxis] * v[2][t_inds, :]

    return P


# Generate a random convex polygon by taking the convex hull
# of a bunch of random points
n = 8
points = np.random.random((n, 2))
line = LineString(points)
poly = line.convex_hull
P = uniform_sample(poly, n=1000)

# Plot the polygon, the Delaunay triangulation and the points
# randomly sampled across its face.
shape = np.array(poly.exterior.coords)
# plt.hold(True)
for p in triangulate(poly):
    coords = np.array(p.exterior.coords)
    plt.plot(coords[:, 0], coords[:, 1], "g--")
plt.plot(shape[:, 0], shape[:, 1], "k")
plt.plot(P[:, 0], P[:, 1], "b.")
