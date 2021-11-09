"""
Utility functions for FlowKit
"""
import numpy as np
# noinspection PyUnresolvedReferences, PyProtectedMember
from .. import _utils_c as utils_c


def points_in_ellipsoid(
        ellipsoid_covariance_matrix,
        ellipsoid_means,
        ellipsoid_distance_square,
        points
):
    """
    Determines whether points in an array are inside an ellipsoid. Points on the
    edge are considered inclusive. True ellipsoids in n-dimensions are supported.

    :param ellipsoid_covariance_matrix: Covariance matrix for the ellipsoid shape (NxN array)
    :param ellipsoid_means: center point of the ellipsoid for n-dimensions
    :param ellipsoid_distance_square: square of the Mahalanobis distance, controlling
        the size of the ellipsoid. The distance square parameter is conceptually
        similar to the number of standard deviations representing the boundary
        for an n-dimensional distribution of points.
    :param points: NumPy array of data points to test for ellipsoid inclusion

    :return: NumPy 1-D array of boolean values for each point. True is inside ellipsoid.
    """
    # we only take points that have already been filtered by the correct
    # columns (i.e. those columns that are included in the ellipsoid

    # First, subtract ellipse centers (consider the ellipse at the origin)
    points_translated = points - ellipsoid_means

    # Get the inverse covariance matrix
    ell_cov_mat_inv = np.linalg.inv(ellipsoid_covariance_matrix)

    # Matrix multiplication of translated points by inverse covariance matrix,
    # rotates the points instead of rotating the ellipse
    points_rot = np.dot(points_translated, ell_cov_mat_inv)
    points_rot = points_rot * points_translated

    # Points are inclusive if they are <= than the distance square
    # since boundary points are considered inclusive
    results = points_rot.sum(axis=1) <= ellipsoid_distance_square

    return results


def points_in_polygon(poly_vertices, points):
    """
    Determines whether points in an array are inside a polygon. Points on the
    edge of the polygon are considered inclusive. This function uses the
    winding number method and is robust to complex polygons with crossing
    boundaries, including the presence of 'holes' created by boundary crosses.

    :param poly_vertices: Polygon vertices (NumPy array of 2-D points)
    :param points: NumPy array of data points to test for polygon inclusion

    :return: NumPy 1-D array of boolean values for each point. True is inside polygon.
    """
    wind_counts = utils_c.points_in_polygon(poly_vertices, len(poly_vertices), points, len(points))
    return wind_counts % 2 != 0


def rotate_point_around_point(point, cov_mat, center_point=(0, 0)):
    """
    Rotates given point around center_point

    :param point: Coordinates of point to rotate
    :param cov_mat: Covariance matrix for the rotation
    :param center_point: Coordinates of the reference rotation point. Default is the origin (0, 0)

    :return: Rotated point coordinates
    """
    point_translated = np.array([point[0] - center_point[0], point[1] - center_point[1]])
    point_rot = np.dot(point_translated, cov_mat)
    point_untranslated = point_rot + center_point

    return point_untranslated
