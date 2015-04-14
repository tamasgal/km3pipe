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

from km3pipe.tools import pdg2name, geant2pdg, angle_between

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

    @property
    def zenith(self):
        return angle_between(self, (0, 0, -1))

    def __str__(self):
        return "({0:.4}, {1:.4}, {2:.4})".format(self.x, self.y, self.z)


class Track(object):
    """Bass class for particle or shower tracks"""
    def __init__(self, id, x, y, z, dx, dy, dz, E=None, t=0, *args):
        self.id = int(id)
        # z correctio due to gen/km3 (ZED -> sea level shift)
        # http://wiki.km3net.physik.uni-erlangen.de/index.php/Simulations
        self.pos = Point((x, y, z + 405.93))
        self.dir = Direction((dx, dy, dz))
        self.E = E
        self.time = t
        self.args = args

    def __repr__(self):
        text = "Track:\n"
        text += " id: {0}\n".format(self.id)
        text += " pos: {0}\n".format(self.pos)
        text += " dir: {0}\n".format(self.dir)
        text += " energy: {0} GeV\n".format(self.E)
        text += " time: {0} ns\n".format(self.time)
        return text


class TrackIn(Track):
    """Representation of a track_in entry in an EVT file"""
    def __init__(self, *context):
        super(self.__class__, self).__init__(*context)
        self.particle_type = geant2pdg(int(self.args[0]))
        try:
            self.length = self.args[1]
        except IndexError:
            self.length = 0

    def __repr__(self):
        text = super(self.__class__, self).__repr__()
        text += " type: {0} '{1}' [PDG]\n".format(self.particle_type,
                                                  pdg2name(self.particle_type))
        text += " length: {0} [m]\n".format(self.length)
        return text


class TrackFit(Track):
    """Representation of a track_fit entry in an EVT file"""
    def __init__(self, *context):
        super(self.__class__, self).__init__(*context)
        self.speed = self.args[0]
        self.ts = self.args[1]
        self.te = self.args[2]
        self.con1 = self.args[3]
        self.con2 = self.args[4]

    def __repr__(self):
        text = super(self.__class__, self).__repr__()
        text += " speed: {0} [m/ns]\n".format(self.speed)
        text += " ts: {0} [ns]\n".format(self.ts)
        text += " te: {0} [ns]\n".format(self.te)
        text += " con1: {0}\n".format(self.con1)
        text += " con2: {0}\n".format(self.con2)
        return text


class Neutrino(object):
    """Representation of a neutrino entry in an EVT file"""
    def __init__(self, id, x, y, z, dx, dy, dz, E, t, Bx, By,
                 ichan, particle_type, channel, *args):
        self.id = id
        # z correctio due to gen/km3 (ZED -> sea level shift)
        # http://wiki.km3net.physik.uni-erlangen.de/index.php/Simulations
        self.pos = Point((x, y, z + 405.93))
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
        if self.E >= 1000000:
            text += ", {0:.3} PeV".format(self.E / 1000000)
        elif self.E >= 1000:
            text += ", {0:.3} TeV".format(self.E / 1000)
        else:
            text += ", {0:.3} GeV".format(float(self.E))
        text += ', CC' if int(self.channel) == 2 else ', NC'
        return text

# The hit entry in an EVT file
Hit = namedtuple('Hit', 'id pmt_id pe time type n_photons track_in c_time')
Hit.__new__.__defaults__ = (None, None, None, None, None, None, None, None)

# The hit_raw entry in an EVT file
def __add_raw_hit__(self, other):
    """Add two hits by adding the ToT and preserve time and pmt_id
    of the earlier one."""
    first = self if self.time <= other.time else other
    return RawHit(first.id, first.pmt_id, self.tot+other.tot, first.time)
RawHit = namedtuple('RawHit', 'id pmt_id tot time')
RawHit.__new__.__defaults__ = (None, None, None, None)
RawHit.__add__ = __add_raw_hit__
