# coding=utf-8
# cython: profile=True
# Filename: dataclasses.py
# cython: embedsignature=True
# pylint: disable=W0232,C0103,C0111
"""
...

"""
from __future__ import division, absolute_import, print_function

import ctypes
from libcpp cimport bool as c_bool  # noqa

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

from km3pipe.tools import angle_between

__all__ = ('Point', 'Position', 'Direction', 'HitSeries', 'Hit')


point_dt = np.dtype([('x', float), ('y', float), ('z', float)])

def Point_(vector, as_recarray=True):
    """A point as numpy.recarray with optional x, y and z attributes."""
    vector = np.array(vector, dtype=np.float)
    if as_recarray:
        return vector.view(point_dt, np.recarray)
    else:
        return vector


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


Position = Direction = Point  # Backwards compatibility


class Direction_(Point):
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


cdef class Hit:
    """Represents a hit on a PMT.

    Parameters
    ----------
    channel_id : int
    dir : Direction or numpy.ndarray
    dom_id : int
    id : int
    pmt_id : int
    pos : Position or numpy.ndarray
    time : int
    tot : int
    triggered : bool

    """
    cdef public int id, dom_id, time, tot, channel_id, pmt_id
    cdef public bint triggered
    cdef public np.ndarray pos
    cdef public np.ndarray dir

    def __cinit__(self,
                  int channel_id,
                  int dom_id,
                  int id,
                  int pmt_id,
                  int time,
                  int tot,
                  bint triggered,
                 ):
        self.channel_id = channel_id
        self.dom_id = dom_id
        self.id = id
        self.pmt_id = pmt_id
        self.time = time
        self.tot = tot
        self.triggered = triggered


cdef class Track:
    """Represents a particle track.

    Parameters
    ----------
    dir : Direction or numpy.ndarray
    energy : float
    id : int
    pos : Position or numpy.ndarray
    time : int
    type : int

    """
    cdef public int id, time, type
    cdef public float energy
    cdef public np.ndarray pos
    cdef public np.ndarray dir

    def __cinit__(self, dir, float energy, int id, pos, int time, int type, ):
        self.dir = dir
        self.energy = energy
        self.id = id
        self.pos = pos
        self.time = time
        self.type = type


class HitSeries(object):
    def __init__(self, hits, event_id=None):
        self.event_id = event_id
        self._channel_id = None
        self._dom_id = None
        self._hits = hits
        self._id = None
        self._index = 0
        self._pmt_id = None
        self._time = None
        self._tot = None
        self._triggered = None

    @classmethod
    def from_aanet(cls, hits, event_id=None):
        return cls([Hit(
            ord(h.channel_id),
            h.dom_id,
            h.id,
            h.pmt_id,
            h.t,
            h.tot,
            h.trig,
        ) for h in hits], event_id)

    @classmethod
    def from_evt(cls, hits, event_id=None):
        return cls([Hit(
            np.nan,     # channel_id
            np.nan,     # dom_id
            h.id,
            h.pmt_id,
            h.time,
            h.tot,
            np.nan,     # triggered
        ) for h in hits], event_id)

    @classmethod
    def from_arrays(cls, channel_ids, dom_ids, ids, pmt_ids, times, tots,
                    triggereds, event_id=None):
        args = channel_ids, dom_ids, ids, pmt_ids, times, tots, triggereds
        hits = cls([Hit(*hit_args) for hit_args in zip(*args)], event_id)
        hits._channel_id = channel_ids
        hits._dom_id = dom_ids
        hits._id = ids
        hits._pmt_id = pmt_ids
        hits._time = times
        hits._tots = tots
        hits._triggered = triggereds
        return hits

    @classmethod
    def from_table(cls, table, event_id=None):
        return cls([Hit(
            row['channel_id'],
            row['dom_id'],
            row['id'],
            row['pmt_id'],
            row['time'],
            row['tot'],
            row['triggered'],
        ) for row in table], event_id)

    def __iter__(self):
        return self

    @property
    def id(self):
        if self._id is None:
            self._id = np.array([h.id for h in self._hits])
        return self._id

    @property
    def time(self):
        if self._time is None:
            self._time = np.array([h.time for h in self._hits])
        return self._time

    @property
    def triggered(self):
        if self._triggered is None:
            self._triggered = np.array([h for h in self._hits if h.triggered])
        return self._triggered

    @property
    def tot(self):
        if self._tot is None:
            self._tot = np.array([h.tot for h in self._hits])
        return self._tot

    @property
    def dom_id(self):
        if self._dom_id is None:
            self._dom_id = np.array([h.dom_id for h in self._hits])
        return self._dom_id

    @property
    def pmt_id(self):
        if self._pmt_id is None:
            self._pmt_id = np.array([h.pmt_id for h in self._hits])
        return self._pmt_id

    @property
    def id(self):
        if self._id is None:
            self._id = np.array([h.id for h in self._hits])
        return self._id

    @property
    def channel_id(self):
        if self._channel_id is None:
            self._channel_id = np.array([h.channel_id for h in self._hits])
        return self._channel_id

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        item = self._hits[self._index]
        self._index += 1
        return item

    def __len__(self):
        return len(self._hits)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._hits[index]
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self._hits[i]

    def __str__(self):
        n_hits = len(self)
        plural = 's' if n_hits > 1 or n_hits == 0 else ''
        return("HitSeries with {0} hit{1}.".format(len(self), plural))

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return '\n'.join([str(hit) for hit in self._hits])


class TrackSeries(object):
    def __init__(self, tracks, event_id=None):
        self.event_id = event_id
        self._dir = None
        self._energy = None
        self._id = None
        self._index = 0
        self._pos = None
        self._time = None
        self._tracks = tracks
        self._type = None

    @classmethod
    def from_aanet(cls, tracks, event_id=None):
        return cls([Track(
            Direction((t.dir.x, t.dir.y, t.dir.z)),
            t.id,
            t.E,
            Position((t.pos.x, t.pos.y, t.pos.z)),
            t.t,
            t.type,
        )
                    for t in tracks], event_id)

    @classmethod
    def from_arrays(cls, directions, energies, ids, positions, times, types,
                    event_id=None):
        args = directions, energies, ids, positions, times, types
        tracks = cls([Track(*track_args) for track_args in zip(*args)], event_id)
        tracks._dir = directions
        tracks._energy = energies
        tracks._id = ids
        tracks._pos = positions
        tracks._time = times
        tracks._type = types
        return tracks

    @classmethod
    def from_table(cls, table, event_id=None):
        return cls([Track(
            row['dir'],
            row['energy'],
            row['id'],
            row['pos'],
            row['time'],
            row['type'],
        ) for row in table], event_id)

    def __iter__(self):
        return self

    def __iter__(self):
        return self

    @property
    def id(self):
        if self._id is None:
            self._id = np.array([t.id for t in self._tracks])
        return self._id

    @property
    def time(self):
        if self._time is None:
            self._time = np.array([t.time for t in self._tracks])
        return self._time

    @property
    def energy(self):
        if self._energy is None:
            self._energy = np.array([t.energy for t in self._tracks])
        return self._energy

    @property
    def type(self):
        if self._type is None:
            self._type = np.array([t.type for t in self._tracks])
        return self._type

    @property
    def pos(self):
        if self._pos is None:
            self._pos = np.array([t.pos for t in self._tracks])
        return self._pos

    @property
    def dir(self):
        if self._dir is None:
            self._dir = np.array([t.dir for t in self._tracks])
        return self._dir

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        item = self._tracks[self._index]
        self._index += 1
        return item

    def __len__(self):
        return len(self._tracks)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._tracks[index]
        elif isinstance(index, slice):
            return self._slice_generator(index)
        else:
            raise TypeError("index must be int or slice")

    def _slice_generator(self, index):
        """A simple slice generator for iterations"""
        start, stop, step = index.indices(len(self))
        for i in range(start, stop, step):
            yield self._tracks[i]

    def __str__(self):
        n_tracks = len(self)
        plural = 's' if n_tracks > 1 or n_tracks == 0 else ''
        return("TrackSeries with {0} track{1}.".format(len(self), plural))

    def __repr__(self):
        return self.__str__()

    def __insp__(self):
        return '\n'.join([str(track) for track in self._tracks])
