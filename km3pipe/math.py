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


def zenith(v):
    """Return the zenith angle in radians"""
    return angle_between((0, 0, -1), v)


def azimuth(v):
    """Return the azimuth angle in radians"""
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


def add_empty_flow_bins(bins):
    """Add empty over- and underflow bins.
    """
    bins = list(bins)
    bins.insert(0, 0)
    bins.append(0)
    return np.array(bins)


def flat_weights(x, bins):
    """Get weights to produce a flat histogram.
    """
    bin_width = np.abs(bins[1] - bins[0])
    hist, _ = np.histogram(x, bins=bins)
    hist = hist.astype(float)
    hist = add_empty_flow_bins(hist)
    hist *= bin_width
    which = np.digitize(x, bins=bins, right=True)
    pop = hist[which]
    wgt = 1 / pop
    wgt *= len(wgt) / np.sum(wgt)
    return wgt


def hsin(theta):
    """haversine"""
    return (1.0 - np.cos(theta)) / 2.


def space_angle(zen_1, zen_2, azi_1, azi_2):
    """Space angle between two directions specified by zenith and azimuth."""
    return hsin(azi_2 - azi_1) + np.cos(azi_1) * np.cos(azi_2) * hsin(zen_2 - zen_1)


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
