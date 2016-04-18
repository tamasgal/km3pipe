# coding=utf-8
# Filename: dataclasses.py
# pylint: disable=W0232,C0103,C0111
"""
...

"""
from __future__ import division, absolute_import, print_function

import numpy as np

from km3pipe.tools import angle_between

__all__ = ('Point', 'Position', 'Direction', 'HitSeries', 'Hit')


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


class HitSeries(object):
    @classmethod
    def from_hdf5(cls, dictionary):
        return cls(dictionary, Hit.from_dict)

    @classmethod
    def from_aanet(cls, dictionary):
        return cls(dictionary, Hit.from_aanet)

    @classmethod
    def from_evt(cls, dictionary):
        return cls(dictionary, Hit.from_evt)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(dictionary, Hit.from_dict)

    def __init__(self, data=None, hit_constructor=None):
        self._data = data
        self.hit_constructor = hit_constructor
        self._hits = None
        self._pos = None
        self._dir = None
        self._index = 0

        if data is None:
            self._hits = []

    def append(self, hit):
        if self._hits is None:
            self._convert_hits()
        self._hits.append(hit)
        self._pos = None
        self._dir = None

    @property
    def pos(self):
        if self._hits is None:
            self._convert_hits()
        if self._pos is None:
            self._pos = np.array([hit.pos for hit in self._hits])
        return self._pos

    @property
    def dir(self):
        if self._hits is None:
            self._convert_hits()
        if self._dir is None:
            self._dir = np.array([hit.dir for hit in self._hits])
        return self._dir

    @property
    def triggered(self):
        """Return a copy of triggered hits."""
        if self._hits is None:
            self._convert_hits()
        triggered_hits = [hit for hit in self._hits if hit.triggered]
        return HitSeries(triggered_hits, Hit.from_hit)

    def _convert_hits(self):
        self._hits = [self.hit_constructor(hit) for hit in self._data]
        self._data = None  # get rid of reference to allow GC

    def __iter__(self):
        return self

    def next(self):
        """Python 2/3 compatibility for iterators"""
        return self.__next__()

    def __next__(self):
        if self._hits is None:
            self._convert_hits()

        if self._index >= len(self):
            self._index = 0
            raise StopIteration
        item = self._hits[self._index]
        self._index += 1
        return item

    def __len__(self):
        if self._hits is None:
            self._convert_hits()
        return len(self._hits)

    def __getitem__(self, index):
        if self._hits is None:
            self._convert_hits()

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
        if self._hits is None:
            self._convert_hits()
        return '\n'.join([str(hit) for hit in self._hits])


class Hit(object):
    def __init__(self, id=None, time=None, tot=None, channel_id=None,
                 dom_id=None, pmt_id=None, triggered=None, data=None):
        self.id = id
        self.time = time
        self.t0 = None
        self.tot = tot
        self.channel_id = channel_id
        self.dom_id = dom_id
        self.pmt_id = pmt_id
        self.triggered = triggered
        self.data = data
        self.pos = None
        self.dir = None
        self.a = None  # charge <- historical

    @classmethod
    def from_hit(cls, hit):
        new_hit = Hit(hit.id, hit.time, hit.tot, hit.channel_id, hit.dom_id,
                      hit.pmt_id, hit.triggered, data=None)
        if hit.pos is not None:
            new_hit.pos = Position(hit.pos)
        if hit.dir is not None:
            new_hit.dir = Direction(hit.dir)
        new_hit.a = hit.a
        return hit

    @classmethod
    def from_dict(cls, data):
        return cls(data['id'], data['time'], data['tot'], data['channel_id'],
                   data['dom_id'], data=data)

    @classmethod
    def from_aanet(cls, data):
        try:
            return cls(data.id, data.t, data.tot, ord(data.channel_id),
                       data.dom_id, triggered=bool(data.trig), data=data)
        except TypeError:
            return cls(data.id, data.t, data.tot, data.channel_id,
                       data.dom_id, triggered=bool(data.trig), data=data)

    @classmethod
    def from_evt(cls, data):
        return cls(data.id, data.time, data.tot, pmt_id=data.pmt_id, data=data)

    def __str__(self):
        return("Hit(id={0}, time={1}, tot={2}, triggered={3})"
               .format(self.id, self.time, self.tot, self.triggered))

    def __repr__(self):
        return self.__str__()
