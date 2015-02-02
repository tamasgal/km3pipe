# coding=utf-8
# Filename: dataclasses.py
# pylint: disable=W0232,C0103,C0111
"""
...

"""
from __future__ import division, absolute_import, print_function

__all__ = ('Point', 'Position', 'Direction', 'Hit')

from collections import namedtuple

import numpy as np

from km3pipe.tools import pdg2name

class Point(np.ndarray):
    """Represents a point in a 3D space"""
    def __new__(cls, input_array=(np.nan, np.nan, np.nan)):
        """Add x, y and z to the ndarray"""
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value


class Position(Point):
    """Represents a point in a 3D space"""
    pass


class Direction(Point):
    """Represents a direction in a 3D space

    The direction vector always normalises itself when an attribute is changed.

    """
    def __new__(cls, input_array=(1, 0, 0)):
        """Add x, y and z to the ndarray"""
        normed_array = np.array(input_array) / np.linalg.norm(input_array)
        obj = np.asarray(normed_array).view(cls)
        return obj

    def _normalise(self):
        normed_array = self / np.linalg.norm(self)
        self[0] = normed_array[0]
        self[1] = normed_array[1]
        self[2] = normed_array[2]

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value
        self._normalise()

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value
        self._normalise()

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value
        self._normalise()

    def __str__(self):
        return "({0}, {1}, {2})".format(self.x, self.y, self.z)


class Track(object):
    def __init__(self, id, x, y, z, dx, dy, dz, p, t, particle_type, length=None):
        self.id = id
        self.pos = Point((x, y, z))
        self.dir = Direction((dx, dy, dz))
        self.momentum = p
        self.time = t
        self.particle_type = particle_type
        self.length = length


class Neutrino(object):
    def __init__(self, id, x, y, z, dx, dy, dz, E, t, Bx, By, ichan, particle_type, channel):
        self.id = id
        self.pos = Point((x, y, z))
        self.dir = Direction((dx, dy, dz))
        self.E = E
        self.time = t
        self.Bx = Bx
        self.By = By
        self.ichan = ichan
        self.particle_type = particle_type
        self.channel = channel

    def __str__(self):
        text = "Neutrino: "
        text += pdg2name(self.particle_type)
        if self.E >= 1000:
            text += ", {0} TeV".format(self.E / 1000)
        else:
            text += ", {0} GeV".format(self.E)
        text += ', CC' if int(self.channel) == 2 else ', NC'
        return text


Hit = namedtuple('Hit', 'id pmt_id pe time type n_photons track_in c_time')
Hit.__new__.__defaults__ = (None, None, None, None, None, None, None, None)

def __add_raw_hit__(self, other):
    """Add two hits by adding the ToT and preserve time and pmt_id
    of the earlier one."""
    first = self if self.time <= other.time else other
    return RawHit(first.id, first.pmt_id, self.tot+other.tot, first.time)

RawHit = namedtuple('RawHit', 'id pmt_id tot time')
RawHit.__new__.__defaults__ = (None, None, None, None)
RawHit.__add__ = __add_raw_hit__
