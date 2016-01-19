# coding=utf-8
# Filename: tools.py
# pylint: disable=C0103
"""
Some frequently used logic.

"""
from __future__ import division, absolute_import, print_function

__author__ = 'tamasgal'

import collections
from collections import namedtuple
from itertools import chain
from datetime import datetime
import time

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
    for _ in range(nfirst):
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
    the_tuple = namedtuple(typename, field_names)
    the_tuple.__new__.__defaults__ = (None,) * len(the_tuple._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = the_tuple(**default_values)
    else:
        prototype = the_tuple(*default_values)
    the_tuple.__new__.__defaults__ = tuple(prototype)
    return the_tuple


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


def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations


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
    """Convert PDG ID to human readable names"""
    # pylint: disable=C0330
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


class PMTReplugger(object):
    """Replugs PMTs and modifies the data according to the new setup."""

    def __init__(self, pmt_combs, angles, rates):
        self._pmt_combs = pmt_combs
        self._new_combs = []
        self._angles = angles
        self._rates = rates
        self._switch = None

    def angle_for(self, pmt_comb):
        """Return angle for given PMT combination"""
        combs = self.current_combs()
        print(combs)
        idx = combs.index(pmt_comb)
        return self._angles[idx]

    def current_combs(self):
        combs = self._new_combs if self._new_combs else self._pmt_combs
        return combs

    @property
    def angles(self):
        combs = self.current_combs()
        idxs = []
        for comb in self._pmt_combs:
            idxs.append(combs.index(comb))
        angles = []
        for idx in idxs:
            angles.append(self._angles[idx])
        return angles

    def switch(self, idxs1, idxs2):
        """Switch PMTs"""
        flat_combs = np.array(self.flatten(self._pmt_combs))
        operations = []
        for old, new in zip(idxs1, idxs2):
            operations.append((self.indices(old, flat_combs), new))
        for idxs, new_value in operations:
            flat_combs[idxs] = new_value
        it = iter(flat_combs)
        self._new_combs = []
        for pmt1, pmt2 in zip(it, it):
            if pmt1 > pmt2:
                self._new_combs.append((pmt2, pmt1))
            else:
                self._new_combs.append((pmt1, pmt2))

    def reset_switches(self):
        """Reset all switches"""
        self._new_combs = None

    def indices(self, item, items):
        values = np.array(items)
        indices = np.where(values == item)[0]
        return indices

    def flatten(self, items):
        return list(chain.from_iterable(items))


class Timer(object):
    """A very simple, accurate and easy to use timer context"""
    def __enter__(self):
        self.__start = time.time()

    def __exit__(self, type, value, traceback):
        self.__finish = time.time()

    def get_seconds(self):
        return self.__finish - self.__start

    def log(self):
        print("{0}s".format(self.get_seconds()))


class Cuckoo(object):
    "A timed callback caller, which only executes once in a given interval."
    def __init__(self, interval=0, callback=print):
        "Setup with interval in seconds and a callback function"
        self.interval = interval
        self.callback = callback
        self.timestamp = None

    def msg(self, message=None):
        "Only execute callback when interval is reached."
        if self.timestamp is None or self._interval_reached():
            self.callback(message)
            self.reset()

    def reset(self):
        "Reset the timestamp"
        self.timestamp = datetime.now()

    def _interval_reached(self):
        "Check if defined interval is reached"
        return (datetime.now() - self.timestamp).total_seconds() > self.interval
