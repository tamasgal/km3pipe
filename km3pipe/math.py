# coding=utf-8
# cython: profile=True
# Filename: math.pyx
# cython: embedsignature=True
# pylint: disable=C0103
"""
Maths.

"""
from __future__ import division, absolute_import, print_function

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
    return np.power(np.sum(np.power(x, p)), 1/p)


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
