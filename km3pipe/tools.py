# coding=utf-8
# Filename: tools.py
# pylint: disable=locally-disabled
"""
Some frequently used logic.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

import collections
from collections import namedtuple

import numpy as np


def unpack_nfirst(seq, nfirst):
    """Unpack the nfrist items from the list and return the rest.

    >>> a, b, c, rest = unpack_nfirst((1, 2, 3, 4, 5), 3)
    >>> a, b, c
    (1, 2, 3)
    >>> rest
    (4, 5)

    """
    iterator = iter(seq)
    for item in xrange(nfirst):
        yield next(iterator, None)
    yield tuple(iterator)


def split(string, callback=None):
    """Split the string and execute the callback function on each part.

    >>> string = "1 2 3 4"
    >>> parts = split(string, int)
    >>> parts
    [1, 2, 3, 4]

    """
    if callback:
        return [callback(i) for i in string.split()]
    else:
        return string.split()

def namedtuple_with_defaults(typename, field_names, default_values=[]):
    """Create a namedtuple with default values

    >>> Node = namedtuple_with_defaults('Node', 'val left right')
    >>> Node()
    Node(val=None, left=None, right=None)
    >>> Node = namedtuple_with_defaults('Node', 'val left right', [1, 2, 3])
    >>> Node()
    Node(val=1, left=2, right=3)
    >>> Node = namedtuple_with_defaults('Node', 'val left right', {'right':7})
    >>> Node()
    Node(val=None, left=None, right=7)
    >>> Node(4)
    Node(val=4, left=None, right=7)
    """
    T = namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


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
    angle = np.arccos(np.dot(v1_u, v2_u))
    return angle

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def geant2pdg(geant_code):
    """Convert GEANT particle ID to PDG"""
    conversion_table = {
        1: 22,   # photon
        2: -11, # positron
        3: 11,   # electron
        5: -13,    # muplus
        6: 13,   # muminus
        7: 111, # pi0
        8: 211, # piplus
        9: -211, # piminus
        10: 130, # k0long
        11: 321, # kplus
        12: -321, # kminus
        13: 2112,  # neutron
        14: 2212, # proton
        16: 310, # kaon0short
        17: 221, # eta
        }
    try:
        return conversion_table[geant_code]
    except KeyError:
        return 0

def pdg2name(pdg_id):
    conversion_table = {
         11: 'e-',
        -11: 'e+',
         12: 'nu_e',
        -12: 'anu_e',
         13: 'mu-',
        -13: 'mu+',
         14: 'nu_mu',
        -14: 'anu_mu',
         15: 'tau-',
        -15: 'tau+',
         16: 'nu_tau',
        -16: 'anu_tau',
         22: 'photon',
        111: 'pi0',
        130: 'K0L',
        211: 'pi-',
       -211: 'pi+',
        310: 'K0S',
        311: 'K0',
        321: 'K+',
       -321: 'K-',
       2112: 'n',
       2212: 'p',
      -2212: 'p-',
    }
    try:
        return conversion_table[pdg_id]
    except KeyError:
        return "N/A"
