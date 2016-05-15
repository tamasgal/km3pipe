# coding=utf-8
# cython: profile=True
# Filename: tools.pyx
# cython: embedsignature=True
# pylint: disable=C0103
"""
Some unsorted, frequently used logic.

"""
from __future__ import division, absolute_import, print_function

import resource
import sys
import subprocess
import collections
from collections import namedtuple
from itertools import chain
from datetime import datetime
import time
from timeit import default_timer as timer
from contextlib import contextmanager

import numpy as np

__author__ = 'tamasgal'


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
    return np.array(vector) / np.linalg.norm(vector)


def circ_permutation(items):
    """Calculate the circular permutation for a given list of items."""
    permutations = []
    for i in range(len(items)):
        permutations.append(items[i:] + items[:i])
    return permutations


def geant2pdg(geant_code):
    """Convert GEANT particle ID to PDG"""
    conversion_table = {
        1: 22,     # photon
        2: -11,    # positron
        3: 11,     # electron
        5: -13,    # muplus
        6: 13,     # muminus
        7: 111,    # pi0
        8: 211,    # piplus
        9: -211,   # piminus
        10: 130,   # k0long
        11: 321,   # kplus
        12: -321,  # kminus
        13: 2112,  # neutron
        14: 2212,  # proton
        16: 310,   # kaon0short
        17: 221,   # eta
        }
    try:
        return conversion_table[geant_code]
    except KeyError:
        return 0


def pdg2name(pdg_id):
    """Convert PDG ID to human readable names"""
    # pylint: disable=C0330
    conversion_table = {
      11:    'e-',
      -11:   'e+',
      12:    'nu_e',
      -12:   'anu_e',
      13:    'mu-',
      -13:   'mu+',
      14:    'nu_mu',
      -14:   'anu_mu',
      15:    'tau-',
      -15:   'tau+',
      16:    'nu_tau',
      -16:   'anu_tau',
      22:    'photon',
      111:   'pi0',
      130:   'K0L',
      211:   'pi-',
      -211:  'pi+',
      310:   'K0S',
      311:   'K0',
      321:   'K+',
      -321:  'K-',
      2112:  'n',
      2212:  'p',
      -2212: 'p-',
    }
    try:
        return conversion_table[pdg_id]
    except KeyError:
        return "N/A"


def total_seconds(td):
    """Convert the timedelta to seconds. (Python 2.6 backward compatibility)"""
    try:
        s = td.total_seconds()
    except AttributeError:
        s = (td.microseconds + (td.seconds + td.days*24*3600) * 10**6) / 10**6
    return s


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
    def __init__(self, message='It', precision=3):
        self.message = message
        self.precision = precision

    def __enter__(self):
        self.__start = timer()
        self.__start_cpu = time.clock()

    def __exit__(self, type, value, traceback):
        self.__finish = timer()
        self.__finish_cpu = time.clock()
        self.log()

    def get_seconds(self):
        return self.__finish - self.__start

    def get_seconds_cpu(self):
        return self.__finish_cpu - self.__start_cpu

    def log(self):
        seconds = self.get_seconds()
        seconds_cpu = self.get_seconds_cpu()
        print("{0} took {1:.{3}f}s (CPU {2:.{3}f}s)."
              .format(self.message, seconds, seconds_cpu, self.precision))


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
        return total_seconds(datetime.now() - self.timestamp) > self.interval


def ifiles(irods_path):
    """Return a list of filenames for given iRODS path (recursively)"""
    raw_output = subprocess.check_output("ils -r --bundle {0}"
                                         "    | grep 'Bundle file:'"
                                         "    | awk '{{print $3}}'"
                                         .format(irods_path), shell=True)
    filenames = raw_output.strip().split("\n")
    return filenames


def remain_file_pointer(function):
    """Remain the file pointer position after calling the decorated function

    This decorator assumes that the last argument is the file handler.

    """
    def wrapper(*args, **kwargs):
        """Wrap the function and remain its parameters and return values"""
        file_obj = args[-1]
        old_position = file_obj.tell()
        return_value = function(*args, **kwargs)
        file_obj.seek(old_position, 0)
        return return_value
    return wrapper


def peak_memory_usage():
    """Return peak memory usage in MB"""
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    factor_mb = 1 / 1024
    if sys.platform == 'darwin':
        factor_mb = 1 / (1024 * 1024)
    return mem * factor_mb


@contextmanager
def ignored(*exceptions):
    """Ignore-context for a given list of exceptions.

    Example:
        with ignored(AttributeError):
            foo.a = 1

    """
    try:
        yield
    except exceptions:
        pass


try:
    dict.iteritems
except AttributeError:
    # for Python 3
    def itervalues(d):
        return iter(d.values())
    def iteritems(d):
        return iter(d.items())
else:
    # for Python 2
    def itervalues(d):
        return d.itervalues()
    def iteritems(d):
        return d.iteritems()
