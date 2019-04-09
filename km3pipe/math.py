#!usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: math.py
# pylint: disable=C0103
"""
Maths, Geometry, coordinates.
"""
from __future__ import absolute_import, print_function, division

import numpy as np

from .logger import get_logger

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = ['Vladimir Kulikovskiy']
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = get_logger(__name__)    # pylint: disable=C0103


def neutrino_to_source_direction(phi, theta, radian=True):
    """Flip the direction.

    Parameters
    ==========
    phi, theta: neutrino direction
    radian: bool [default=True]
        receive + return angles in radian? (if false, use degree)
    """
    phi = np.atleast_1d(phi).copy()
    theta = np.atleast_1d(theta).copy()
    if not radian:
        phi *= np.pi / 180
        theta *= np.pi / 180
    assert np.all(phi <= 2 * np.pi)
    assert np.all(theta <= np.pi)
    azimuth = (phi + np.pi) % (2 * np.pi)
    zenith = np.pi - theta
    if not radian:
        azimuth *= 180 / np.pi
        zenith *= 180 / np.pi
    return azimuth, zenith


def source_to_neutrino_direction(azimuth, zenith, radian=True):
    """Flip the direction.

    Parameters
    ==========
    zenith, azimuth: neutrino origin
    radian: bool [default=True]
        receive + return angles in radian? (if false, use degree)
    """
    azimuth = np.atleast_1d(azimuth).copy()
    zenith = np.atleast_1d(zenith).copy()
    if not radian:
        azimuth *= np.pi / 180
        zenith *= np.pi / 180
    phi = (azimuth - np.pi) % (2 * np.pi)
    theta = np.pi - zenith
    if not radian:
        phi *= 180 / np.pi
        theta *= 180 / np.pi
    return phi, theta


def theta(v):
    """Neutrino direction in polar coordinates.

    Downgoing event: theta = 180deg
    Horizont: 90deg
    Upgoing: theta = 0

    Angles in radians.
    """
    v = np.atleast_2d(v)
    dir_z = v[:, 2]
    return theta_separg(dir_z)


def theta_separg(dir_z):
    return np.arccos(dir_z)


def phi(v):
    """Neutrino direction in polar coordinates.

    ``phi``, ``theta`` is the opposite of ``zenith``, ``azimuth``.

    Angles in radians.
    """
    v = np.atleast_2d(v)
    dir_x = v[:, 0]
    dir_y = v[:, 1]
    return phi_separg(dir_x, dir_y)


def phi_separg(dir_x, dir_y):
    p = np.arctan2(dir_y, dir_x)
    p[p < 0] += 2 * np.pi
    return p


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

    ``phi``, ``theta`` is the opposite of ``zenith``, ``azimuth``.

    This is the 'normal' azimuth definition -- beware of how you
    define your coordinates. KM3NeT defines azimuth
    differently than e.g. SLALIB, astropy, the AAS.org
    """
    v = np.atleast_2d(v)
    azi = phi(v) - np.pi
    azi[azi < 0] += 2 * np.pi
    if len(azi) == 1:
        return azi[0]
    return azi


def cartesian(phi, theta, radius=1):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack((x, y, z))


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'.

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


def innerprod_1d(v1, v2):
    """1d Inner product for vector-of-vectors.

    Example:
    ========
    ```
    v1 = np.array([dir_x, dir_y, dir_z]).T
    v2 = ... # dito
    angle_between_v1_v1 = innerprod_1d(v1, v2)
    ```
    """
    return np.einsum('ij,ij->i', v1, v1)


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
    unit = vector / np.linalg.norm(vector, axis=1, **kwargs)[:, None]
    return unit.reshape(out_shape)


def pld3(pos, line_vertex, line_dir):
    """Calculate the point-line-distance for given point and line."""
    pos = np.atleast_2d(pos)
    line_vertex = np.atleast_1d(line_vertex)
    line_dir = np.atleast_1d(line_dir)
    c = np.cross(line_dir, line_vertex - pos)
    n1 = np.linalg.norm(c, axis=1)
    n2 = np.linalg.norm(line_dir)
    out = n1 / n2
    if out.ndim == 1 and len(out) == 1:
        return out[0]
    return out


def lpnorm(x, p=2):
    return np.power(np.sum(np.power(x, p)), 1 / p)


def dist(x1, x2, axis=0):
    """Return the distance between two points.

    Set axis=1 if x1 is a vector and x2 a matrix to get a vector of distances.
    """
    return np.linalg.norm(x2 - x1, axis=axis)


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
        np.square((cos(phi2) * sin(lambdelt))) +
        square(cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(lambdelt))
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
        from matplotlib.path import Path
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
            ) for theta in angles
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
    gold = np.exp(-1 * time_residual * time_residual /
                  (2 * 1.5 * 1.5)) / len(time_residual)
    gold = np.sum(gold)


def log_b(arg, base):
    """Logarithm to any base"""
    return np.log(arg) / np.log(base)


def qrot(vector, quaternion):
    """Rotate a 3D vector using quaternion algebra.

    Implemented by Vladimir Kulikovskiy.

    Parameters
    ----------
    vector: np.array
    quaternion: np.array

    Returns
    -------
    np.array

    """
    t = 2 * np.cross(quaternion[1:], vector)
    v_rot = vector + quaternion[0] * t + np.cross(quaternion[1:], t)
    return v_rot


def qeuler(yaw, pitch, roll):
    """Convert Euler angle to quaternion.

    Parameters
    ----------
    yaw: number
    pitch: number
    roll: number

    Returns
    -------
    np.array

    """
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q = np.array((
        cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp
    ))
    return q


def qrot_yaw(vector, heading):
    """Rotate vectors using quaternion algebra.


    Parameters
    ----------
    vector: np.array or list-like (3 elements)
    heading: the heading to rotate to [deg]

    Returns
    -------
    np.array

    """
    return qrot(vector, qeuler(heading, 0, 0))


def intersect_3d(p1, p2):
    """Find the closes point for a given set of lines in 3D.

    Parameters
    ----------
    p1 : (M, N) array_like
        Starting points
    p2 : (M, N) array_like
        End points.

    Returns
    -------
    x : (N,) ndarray
        Least-squares solution - the closest point of the intersections.

    Raises
    ------
    numpy.linalg.LinAlgError
        If computation does not converge.

    """
    v = p2 - p1
    normed_v = unit_vector(v)
    nx = normed_v[:, 0]
    ny = normed_v[:, 1]
    nz = normed_v[:, 2]
    xx = np.sum(nx**2 - 1)
    yy = np.sum(ny**2 - 1)
    zz = np.sum(nz**2 - 1)
    xy = np.sum(nx * ny)
    xz = np.sum(nx * nz)
    yz = np.sum(ny * nz)
    M = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
    x = np.sum(
        p1[:, 0] * (nx**2 - 1) + p1[:, 1] * (nx * ny) + p1[:, 2] * (nx * nz)
    )
    y = np.sum(
        p1[:, 0] * (nx * ny) + p1[:, 1] * (ny * ny - 1) + p1[:, 2] * (ny * nz)
    )
    z = np.sum(
        p1[:, 0] * (nx * nz) + p1[:, 1] * (ny * nz) + p1[:, 2] * (nz**2 - 1)
    )
    return np.linalg.lstsq(M, np.array((x, y, z)), rcond=None)[0]
