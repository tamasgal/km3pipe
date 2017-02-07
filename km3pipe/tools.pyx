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
import os
import base64
import subprocess
import collections
import socket
from collections import namedtuple
from itertools import chain
from datetime import datetime
import time
from timeit import default_timer as timer
from contextlib import contextmanager
import re
import warnings

import numpy as np
import scipy.linalg


from .logger import logging

__author__ = "Tamas Gal and Moritz Lotze"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal and Moritz Lotze"
__email__ = "tgal@km3net.de"
__status__ = "Development"

log = logging.getLogger(__name__)  # pylint: disable=C0103


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


def split(string, callback=None, sep=' '):
    """Split the string and execute the callback function on each part.

    >>> string = "1 2 3 4"
    >>> parts = split(string, int)
    >>> parts
    [1, 2, 3, 4]

    """
    if callback is not None:
        return [callback(i) for i in string.split(sep)]
    else:
        return string.split(sep)


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


_PDG2NAME = {
    1: 'd',
    2: 'u',
    3: 's',
    4: 'c',
    5: 'b',
    6: 't',
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

_NAME2PDG = {val: key for key, val in _PDG2NAME.items()}      # noqa


def pdg2name(pdg_id):
    """Convert PDG ID to human readable names"""
    # pylint: disable=C0330
    try:
        return _PDG2NAME[pdg_id]
    except KeyError:
        return "N/A"

def name2pdg(name):
    try:
        return _NAME2PDG[name]
    except KeyError:
        return 0


def total_seconds(td):
    """Convert the timedelta to seconds. (Python 2.6 backward compatibility)"""
    try:
        s = td.total_seconds()
    except AttributeError:
        s = (td.microseconds + (td.seconds + td.days*24*3600) * 10**6) / 10**6
    return s


class Timer(object):
    """A very simple, accurate and easy to use timer context"""
    def __init__(self, message='It', precision=3, callback=print):
        self.message = message
        self.precision = precision
        self.callback = callback

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        self.__start = timer()
        self.__start_cpu = time.clock()

    def stop(self):
        self.__finish = timer()
        self.__finish_cpu = time.clock()
        self.log()
        return self.seconds

    @property
    def seconds(self):
        return self.__finish - self.__start

    @property
    def cpu_seconds(self):
        return self.__finish_cpu - self.__start_cpu

    def log(self):
        self.callback("{0} took {1:.{3}f}s (CPU {2:.{3}f}s)."
                      .format(self.message,
                              self.seconds, self.cpu_seconds,
                              self.precision))


class Cuckoo(object):
    "A timed callback caller, which only executes once in a given interval."
    def __init__(self, interval=0, callback=print):
        "Setup with interval in seconds and a callback function"
        self.interval = interval
        self.callback = callback
        self.timestamp = None

    def msg(self, *args, **kwargs):
        "Only execute callback when interval is reached."
        if self.timestamp is None or self._interval_reached():
            self.callback(*args, **kwargs)
            self.reset()

    def reset(self):
        "Reset the timestamp"
        self.timestamp = datetime.now()

    def _interval_reached(self):
        "Check if defined interval is reached"
        return total_seconds(datetime.now() - self.timestamp) > self.interval

    def __call__(self, *args, **kwargs):
        "Run the msg function when called directly."
        self.msg(*args, **kwargs)


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


def token_urlsafe(nbytes=32):
    """Return a random URL-safe text string, in Base64 encoding.

    This is taken and slightly modified from the Python 3.6 stdlib.

    The string has *nbytes* random bytes.  If *nbytes* is ``None``
    or not supplied, a reasonable default is used.

    >>> token_urlsafe(16)  #doctest:+SKIP
    'Drmhze6EPcv0fN_81Bj-nA'

    """
    tok = os.urandom(nbytes)
    return base64.urlsafe_b64encode(tok).rstrip(b'=').decode('ascii')


def tai_timestamp():
    """Return current TAI timestamp."""
    timestamp = time.time()
    date = datetime.utcfromtimestamp(timestamp)
    if date.year < 1972:
        return timestamp
    offset = 10 + timestamp
    leap_seconds = [
        (1972, 1, 1), (1972, 7, 1), (1973, 1, 1),
        (1974, 1, 1), (1975, 1, 1), (1976, 1, 1),
        (1977, 1, 1), (1978, 1, 1), (1979, 1, 1),
        (1980, 1, 1), (1981, 7, 1), (1982, 7, 1),
        (1983, 7, 1), (1985, 7, 1), (1988, 1, 1),
        (1990, 1, 1), (1991, 1, 1), (1992, 7, 1),
        (1993, 7, 1), (1994, 7, 1), (1996, 1, 1),
        (1997, 7, 1), (1999, 1, 1), (2006, 1, 1),
        (2009, 1, 1), (2012, 7, 1), (2015, 7, 1),
        (2017, 1, 1),
    ]
    for idx, leap_date in enumerate(leap_seconds):
        if leap_date >= (date.year, date.month, date.day):
            return idx - 1 + offset
    return len(leap_seconds) - 1 + offset


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


def decamelise(text):
    """Convert CamelCase to lower_and_underscore."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def camelise(text, capital_first=True):
    """Convert lower_underscore to CamelCase."""
    def camelcase():
        if not capital_first:
            yield str.lower
        while True:
            yield str.capitalize

    c = camelcase()
    return "".join(next(c)(x) if x else '_' for x in text.split("_"))


def insert_prefix_to_dtype(arr, prefix):
    new_cols = [prefix + '_' + col for col in arr.dtype.names]
    arr.dtype.names = new_cols
    return arr


class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    >>> @deprecated('some warning')
    ... def some_function(): pass
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.
    # and stolen again from sklearn.utils

    def __init__(self, extra=''):
        """
        Parameters
        ----------
        extra: string
          to be added to the deprecation messages
        """
        self.extra = extra

    def __call__(self, obj):
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


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


def prettyln(text, fill='-', align='^', prefix='[ ', suffix=' ]', length=69):
    """Wrap `text` in a pretty line with maximum length."""
    text = '{prefix}{0}{suffix}'.format(text, prefix=prefix, suffix=suffix)
    print("{0:{fill}{align}{length}}"
          .format(text, fill=fill, align=align, length=length))


def irods_filepath(det_id, run_id):
    """Generate the iRODS filepath for given detector (O)ID and run ID"""
    data_path = "/in2p3/km3net/data/raw/sea"
    from km3pipe.db import DBManager
    if not isinstance(det_id, int):
        dts = DBManager().detectors
        det_id = int(dts[dts.OID == det_id].SERIALNUMBER.values[0])
    return data_path + "/KM3NeT_{0:08}/{2}/KM3NeT_{0:08}_{1:08}.root" \
           .format(det_id, run_id, run_id//1000)
