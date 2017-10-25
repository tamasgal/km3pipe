# coding=utf-8
# cython: profile=True
# Filename: math.pyx
# cython: embedsignature=True
# pylint: disable=C0103
"""
Maths, Geometry, coordinates.
"""
from __future__ import division, absolute_import, print_function

from matplotlib.path import Path
import numpy as np
import scipy.linalg


from .logger import logging

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


def mad(v):
    """MAD -- Median absolute deviation. More robust than standard deviation.
    """
    return np.median(np.abs(v - np.median(v)))


def zenith(v):
    """Return the zenith angle in radians.

    Defined as 'Angle respective to downgoing'.
    Downgoing event: zenith = 0
    Horizont: 90deg
    Upgoing: zenith = 180deg
    """
    return angle_between((0, 0, -1), v)


def azimuth(v):
    """Return the azimuth angle in radians.

    This is the 'normal' azimuth definition -- beware of how you
    define your coordinates. KM3NeT defines azimuth
    differently than e.g. SLALIB, astropy, the AAS.org
    """
    v = np.atleast_2d(v)
    phi = np.arctan2(v[:, 1], v[:, 0])
    phi[phi < 0] += 2 * np.pi
    if len(phi) == 1:
        return phi[0]
    return phi


def cartesian(phi, theta, radius=1):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack((x, y, z))


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793

    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # Don't use `np.dot`, does not work with all shapes
    angle = np.arccos(np.inner(v1_u, v2_u))
    return angle


def unit_vector(vector, **kwargs):
    """Returns the unit vector of the vector."""
    # This also works for a dataframe with columns ['x', 'y', 'z']
    # However, the division operation is picky about the shapes
    # So, remember input vector shape, cast all up to 2d,
    # do the (ugly) conversion, then return unit in same shape as input
    # of course, the numpy-ized version of the input...
    vector = np.array(vector)
    out_shape = vector.shape
    vector = np.atleast_2d(vector)
    unit = vector / scipy.linalg.norm(vector, axis=1, **kwargs)[:, None]
    return unit.reshape(out_shape)


def pld3(p1, p2, d2):
    """Calculate the point-line-distance for given point and line."""
    return scipy.linalg.norm(np.cross(d2, p2 - p1)) / scipy.linalg.norm(d2)


def lpnorm(x, p=2):
    return np.power(np.sum(np.power(x, p)), 1 / p)


def dist(x1, x2):
    return lpnorm(x2 - x1, p=2)


def com(points, masses=None):
    """Calculate center of mass for given points.
    If masses is not set, assume equal masses."""
    if masses is None:
        return np.average(points, axis=0)
    else:
        return np.average(points, axis=0, weights=masses)


def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations


def hsin(theta):
    """haversine"""
    return (1.0 - np.cos(theta)) / 2.


def space_angle(phi1, theta1, phi2, theta2):
    """Also called Great-circle-distance --
    use long-ass formula from wikipedia (last in section):
    https://en.wikipedia.org/wiki/Great-circle_distance#Computational_formulas

    Space angle only makes sense in lon-lat, so convert zenith -> latitude.
    """
    from numpy import pi, sin, cos, arctan2, sqrt, square
    lamb1 = pi / 2 - theta1
    lamb2 = pi / 2 - theta2
    lambdelt = lamb2 - lamb1
    under = sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(lambdelt)
    over = sqrt(
        np.square((cos(phi2) * sin(lambdelt))) + square(
            cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(lambdelt)
        )
    )
    angle = arctan2(over, under)
    return angle


def rotation_matrix(axis, theta):
    """The Euler–Rodrigues formula.

    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis: vector to rotate around
    theta: rotation angle, in rad
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
    ])


class Polygon(object):
    """A polygon, to implement containment conditions."""
    def __init__(self, vertices):
        self.poly = Path(vertices)

    def contains(self, points):
        points = np.atleast_2d(points)
        points_flat = points.reshape((-1, 2))
        is_contained = self.poly.contains_points(points_flat)
        return is_contained

    def contains_xy(self, x, y):
        xy = np.column_stack((x, y))
        return self.contains(xy)


class IrregularPrism(object):
    """Like a cylinder, but the top is an irregular Polygon."""
    def __init__(self, xy_vertices, z_min, z_max):
        self.poly = Polygon(xy_vertices)
        self.z_min = z_min
        self.z_max = z_max

    def _is_z_contained(self, z):
        return (self.z_min <= z) & (z <= self.z_max)

    def contains(self, points):
        points = np.atleast_2d(points)
        points_xy = points[:, [0, 1]]
        points_z = points[:, 2]
        is_xy_contained = self.poly.contains(points_xy)
        is_z_contained = self._is_z_contained(points_z)
        return is_xy_contained & is_z_contained

    def contains_xyz(self, x, y, z):
        xyz = np.column_stack((x, y, z))
        return self.contains(xyz)


class SparseCone(object):
    """A Cone, represented by sparse samples.

    This samples evenly spaced points from the base circle.

    Parameters
    ----------
    spike_pos: coordinates of the top
    bottom_center_pos: center of the bottom circle
    opening_angle: cone opening angle, in rad
        theta, axis to mantle, *not* mantle-mantle. So this is the angle
        to the axis, and mantle-to-mantle (aperture) is 2 theta.
    """
    def __init__(self, spike_pos, bottom_center_pos, opening_angle):
        self.spike_pos = np.asarray(spike_pos)
        self.bottom_center_pos = np.asarray(bottom_center_pos)
        self.opening_angle = opening_angle
        self.top_bottom_vec = self.bottom_center_pos - self.spike_pos
        self.height = np.linalg.norm(self.top_bottom_vec)
        self.mantle_length = self.height / np.cos(self.opening_angle)
        self.radius = self.height * np.tan(self.opening_angle * self.height)

    @classmethod
    def _equidistant_angles_from_circle(cls, n_angles=4):
        return np.linspace(0, 2 * np.pi, n_angles + 1)[:-1]

    @property
    def _random_circle_vector(self):
        k = self.top_bottom_vec
        r = self.radius
        x = np.random.randn(3)
        x -= x.dot(k) * k
        x *= r / np.linalg.norm(x)
        return x

    def sample_circle(self, n_angles=4):
        angles = self._equidistant_angles_from_circle(n_angles)
        random_circle_vector = self._random_circle_vector
        # rotate the radius vector around the cone axis
        points_on_circle = [
            np.dot(
                rotation_matrix(self.top_bottom_vec, theta),
                random_circle_vector
            )
            for theta in angles
        ]
        return points_on_circle

    @property
    def sample_axis(self):
        return [self.spike_pos, self.bottom_center_pos]

    def sample(self, n_circle=4):
        points = self.sample_circle(n_circle)
        points.extend(self.sample_axis)
        return points


def inertia(x, y, z, weight=None):
    """Inertia tensor, stolen of thomas"""
    if weight is None:
        weight = 1
    tensor_of_inertia = np.zeros((3, 3), dtype=float)
    tensor_of_inertia[0][0] = (y * y + z * z) * weight
    tensor_of_inertia[0][1] = (-1) * x * y * weight
    tensor_of_inertia[0][2] = (-1) * x * z * weight
    tensor_of_inertia[1][0] = (-1) * x * y * weight
    tensor_of_inertia[1][1] = (x * x + z * z) * weight
    tensor_of_inertia[1][2] = (-1) * y * z * weight
    tensor_of_inertia[2][0] = (-1) * x * z * weight
    tensor_of_inertia[2][1] = (-1) * z * y * weight
    tensor_of_inertia[2][2] = (x * x + y * y) * weight

    eigen_values = np.linalg.eigvals(tensor_of_inertia)
    small_inertia = eigen_values[2][2]
    middle_inertia = eigen_values[1][1]
    big_inertia = eigen_values[0][0]
    return small_inertia, middle_inertia, big_inertia


def g_parameter(time_residual):
    """stolen from thomas"""
    mean = np.mean(time_residual)
    time_residual_prime = (time_residual - np.ones(time_residual.shape) * mean)
    time_residual_prime *= time_residual_prime / (-2 * 1.5 * 1.5)
    time_residual_prime = np.exp(time_residual_prime)
    g = np.sum(time_residual_prime) / len(time_residual)
    return g


def gold_parameter(time_residual):
    """stolen from thomas"""
    gold = np.exp(
        -1 * time_residual * time_residual / (2 * 1.5 * 1.5)
    ) / len(time_residual)
    gold = np.sum(gold)
